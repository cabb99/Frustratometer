"""Tests for static-context reduction (frustratometer.awsem.static_context).

A "static context" freezes part of the protein at its real sequence while the rest (the
"active" residues) is free to change. The frozen part is folded into a smaller model that
describes only the active residues, plus a fixed energy "offset" for the frozen part.
Evaluating this small model on the active residues (plus the offset) must give the same
energy as the full model with the frozen residues kept at their real sequence.
"""
import numpy as np
import pytest

import frustratometer
from frustratometer import frustration
from frustratometer.frustration.frustration import _AA, compute_native_energy_sparse
from frustratometer.awsem.physics import fold_static_context

test_data_path = 'tests/data'


@pytest.fixture(scope="module")
def sparse_model():
    structure = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A')
    return frustratometer.AWSEM(structure, distance_cutoff_contact=9.5,
                                min_sequence_separation_contact=2, k_electrostatics=0)


def _seq_index(seq):
    return np.array([_AA.index(c) for c in seq])


def test_native_energy_equivalence(sparse_model):
    """Shrinking the model to only the active residues must not change the energy: the small
    model's energy plus the fixed offset from the frozen part equals the full model's energy."""
    model = sparse_model
    seq_index = _seq_index(model.sequence)
    rng = np.random.default_rng(0)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    reduced, offset = fold_static_context(model.sparse_potts_model, seq_index, active)
    active_seq = ''.join(model.sequence[i] for i in active)
    reduced_energy = compute_native_energy_sparse(active_seq, reduced) + offset
    np.testing.assert_allclose(reduced_energy, model.native_energy(), rtol=1e-9, atol=1e-8)


def test_mutant_active_matches_full_sequence(sparse_model):
    """Mutating the active residues in the small model gives the same energy as making those
    same mutations in the full protein while the frozen residues keep their original sequence."""
    model = sparse_model
    seq_index = _seq_index(model.sequence)
    rng = np.random.default_rng(1)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    reduced, offset = fold_static_context(model.sparse_potts_model, seq_index, active)
    for _ in range(5):
        mutant_codes = rng.integers(1, 21, size=len(active))  # avoid gap (index 0)
        full_seq = list(model.sequence)
        for local, residue in enumerate(active):
            full_seq[residue] = _AA[mutant_codes[local]]
        full_energy = compute_native_energy_sparse(''.join(full_seq), model.sparse_potts_model)
        mutant_seq = ''.join(_AA[code] for code in mutant_codes)
        reduced_energy = compute_native_energy_sparse(mutant_seq, reduced) + offset
        np.testing.assert_allclose(reduced_energy, full_energy, rtol=1e-9, atol=1e-8)


def test_edge_cases(sparse_model):
    """Check the two extremes. If every residue is active, nothing is frozen, so the offset is
    zero and all contacts are kept. If every residue is frozen, nothing is active, so no
    contacts remain and the offset holds the whole native energy."""
    model = sparse_model
    seq_index = _seq_index(model.sequence)
    n_contacts = len(model.sparse_potts_model['contact_i'])

    # all active: zero offset, all contacts retained
    reduced, offset = fold_static_context(model.sparse_potts_model, seq_index, np.arange(model.N))
    assert offset == pytest.approx(0.0, abs=1e-9)
    assert len(reduced['contact_i']) == n_contacts

    # all static: offset equals the full native energy, no contacts remain
    reduced, offset = fold_static_context(model.sparse_potts_model, seq_index, np.array([], dtype=int))
    np.testing.assert_allclose(offset, model.native_energy(), rtol=1e-9, atol=1e-8)
    assert len(reduced['contact_i']) == 0


def test_matches_awsem_energy_selected():
    """The small model's energy matches AwsemEnergySelected, a separate trusted way of
    computing the same thing, across several random sequences of the active residues."""
    from frustratometer.optimization.optimization import AwsemEnergySelected
    structure = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A')
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5,
                                 min_sequence_separation_contact=2, k_electrostatics=0)
    seq_index = _seq_index(model.sequence)
    rng = np.random.default_rng(2)
    active = np.sort(rng.choice(model.N, size=model.N // 3, replace=False))
    reduced, offset = fold_static_context(model.sparse_potts_model, seq_index, active)

    # AwsemEnergySelected consumes the dense Potts view and a dense mask.
    model.mask = model.mask.to_dense(fill=0.0)
    selected = AwsemEnergySelected(model, selection=active, use_numba=False)
    for _ in range(4):
        mutant_codes = rng.integers(1, 21, size=len(active))
        mutant_seq = ''.join(_AA[code] for code in mutant_codes)
        reduced_energy = compute_native_energy_sparse(mutant_seq, reduced) + offset
        selected_energy = selected.energy(mutant_codes)
        np.testing.assert_allclose(reduced_energy, selected_energy, rtol=1e-7, atol=1e-7)


def test_awsem_fold_method_matches_function(sparse_model):
    """Calling fold_static_context as a method on the model gives the same reduced model and
    offset as calling the standalone function."""
    model = sparse_model
    active = np.arange(0, model.N, 2)
    reduced_method, offset_method = model.fold_static_context(active)
    reduced_func, offset_func = fold_static_context(
        model.sparse_potts_model, _seq_index(model.sequence), active)
    assert offset_method == pytest.approx(offset_func)
    np.testing.assert_array_equal(reduced_method['contact_i'], reduced_func['contact_i'])
    np.testing.assert_allclose(reduced_method['h'], reduced_func['h'])


def test_active_singleresidue_equals_full_restricted(sparse_model):
    """Single-residue frustration for a chosen set of residues matches the full-protein result
    read off at those same residues. (Single-residue frustration already keeps every other
    residue fixed, which is just what freezing the rest of the protein does.)"""
    model = sparse_model
    full = model.frustration(kind='singleresidue')
    rng = np.random.default_rng(3)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    restricted = model.frustration(kind='singleresidue', active_residues=active)
    assert restricted.shape == (len(active),)
    np.testing.assert_allclose(restricted, full[active], rtol=1e-6, atol=1e-6)


def test_active_mutational_equals_full_restricted(sparse_model):
    """Mutational frustration for a chosen set of residues matches the matching part of the
    full-protein result: the rows and columns for those residues."""
    model = sparse_model
    full = model.frustration(kind='mutational')
    rng = np.random.default_rng(4)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    restricted = model.frustration(kind='mutational', active_residues=active)
    assert restricted.shape == (len(active), len(active))
    np.testing.assert_allclose(restricted, full[np.ix_(active, active)], rtol=1e-6, atol=1e-6)


def test_active_residues_boolean_mask(sparse_model):
    """You can pick the active residues with a True/False mask, and that gives the same
    frustration as the full result read off at the True positions."""
    model = sparse_model
    mask = np.zeros(model.N, dtype=bool)
    mask[::2] = True
    restricted = model.frustration(kind='singleresidue', active_residues=mask)
    full_at_mask = model.frustration(kind='singleresidue')[np.where(mask)[0]]
    np.testing.assert_allclose(restricted, full_at_mask, rtol=1e-6, atol=1e-6)


def test_active_configurational_equals_full_restricted(sparse_model):
    """Configurational frustration for a chosen set of residues (with the decoy shuffle drawn
    from the whole protein) matches the full result restricted to those residues. The decoy
    reference is global, and each contact's energy is local to its two residues, so selecting
    a subset is the same as computing everything and reading off the active-active contacts."""
    model = sparse_model
    rng = np.random.default_rng(0)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    selected = model.frustration(kind='configurational', active_residues=active, seed=7, n_decoys=500)
    full = model.frustration(kind='configurational', active_residues=np.arange(model.N),
                             seed=7, n_decoys=500)
    restricted = full[np.ix_(active, active)]
    assert selected.shape == (len(active), len(active))
    assert np.array_equal(np.isnan(selected), np.isnan(restricted))
    finite = np.isfinite(selected)
    np.testing.assert_allclose(selected[finite], restricted[finite], rtol=1e-9, atol=1e-9)


def test_active_configurational_boolean_mask(sparse_model):
    """A True/False mask selects the same active-active contacts as the matching index list."""
    model = sparse_model
    rng = np.random.default_rng(0)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    mask = np.zeros(model.N, bool); mask[active] = True
    by_index = model.frustration(kind='configurational', active_residues=active, seed=7, n_decoys=500)
    by_mask = model.frustration(kind='configurational', active_residues=mask, seed=7, n_decoys=500)
    np.testing.assert_array_equal(np.isnan(by_mask), np.isnan(by_index))
    finite = np.isfinite(by_index)
    np.testing.assert_allclose(by_mask[finite], by_index[finite], rtol=1e-9, atol=1e-9)


def test_active_configurational_scope_active_differs_from_whole(sparse_model):
    """The decoy shuffle can be drawn from only the active residues instead of the whole
    protein. That changes the reference, so the 'active' scope gives different values than the
    'whole' scope while reporting the same set of active-active contacts."""
    model = sparse_model
    rng = np.random.default_rng(1)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    whole = model.frustration(kind='configurational', active_residues=active, seed=7,
                              n_decoys=2000, decoy_scope='whole')
    active_only = model.frustration(kind='configurational', active_residues=active, seed=7,
                                    n_decoys=2000, decoy_scope='active')
    np.testing.assert_array_equal(np.isnan(whole), np.isnan(active_only))
    finite = np.isfinite(whole)
    assert not np.allclose(whole[finite], active_only[finite])


def test_configurational_charge_field_changes_frustration(sparse_model):
    """An external charge enters the configurational energy as a single-body term (like burial),
    so placing one near an active residue changes its configurational frustration; with no
    charge the result is the standard (field-free) configurational frustration."""
    model = sparse_model
    rng = np.random.default_rng(2)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    base = model.frustration(kind='configurational', active_residues=active, seed=7, n_decoys=500)
    charge_coords = model._cb_coords[active[:1]] + np.array([3.0, 0.0, 0.0])
    charged = model.frustration(kind='configurational', active_residues=active, seed=7, n_decoys=500,
                                charge_coords=charge_coords, charges=np.array([-1.0]))
    finite = np.isfinite(base)
    assert not np.allclose(base[finite], charged[finite])


def test_pseudoconfigurational_selection_raises(sparse_model):
    """Pseudoconfigurational frustration on a selection is not supported (it has no clean
    static-context fold), so requesting it with a selection raises a clear error."""
    model = sparse_model
    active = np.arange(0, model.N, 2)
    with pytest.raises(NotImplementedError):
        model.frustration(kind='pseudoconfigurational', active_residues=active)


def test_select_residues_string_resolves(sparse_model):
    """select_residues turns any of its accepted inputs into the same sorted list of residue
    numbers: a selection string, a list of residue numbers, or a True/False mask."""
    model = sparse_model
    idx = model.select_residues('resid 10 to 40')
    assert idx.dtype.kind == 'i'
    assert 0 < len(idx) < model.N
    # index/mask inputs pass through
    np.testing.assert_array_equal(model.select_residues(idx), idx)
    mask = np.zeros(model.N, bool); mask[idx] = True
    np.testing.assert_array_equal(model.select_residues(mask), idx)


def test_active_selection_string_matches_indices(sparse_model):
    """Choosing the active residues with a selection string gives the same frustration as
    choosing them with the matching list of residue numbers."""
    model = sparse_model
    idx = model.select_residues('resid 10 to 60')
    by_selection = model.frustration(kind='singleresidue', active_selection='resid 10 to 60')
    by_indices = model.frustration(kind='singleresidue', active_residues=idx)
    np.testing.assert_allclose(by_selection, by_indices, rtol=1e-9, atol=1e-9)


def test_static_selection_is_active_complement(sparse_model):
    """Naming the frozen residues with a selection string is the same as marking everyone
    else as active."""
    model = sparse_model
    static_idx = set(model.select_residues('resid 1 to 30').tolist())
    active_idx = np.array([i for i in range(model.N) if i not in static_idx], dtype=np.intp)
    by_static = model.frustration(kind='singleresidue', static_selection='resid 1 to 30')
    by_active = model.frustration(kind='singleresidue', active_residues=active_idx)
    np.testing.assert_allclose(by_static, by_active, rtol=1e-9, atol=1e-9)


def test_external_charge_field_formula():
    """external_charge_field matches its expected formula: every external charge adds a
    potential that fades with distance, and the total is scaled by each amino acid's charge."""
    from frustratometer.awsem.physics import external_charge_field
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(0)
    n_residues, n_charges, n_aa = 6, 3, 21
    residue_coords = rng.normal(size=(n_residues, 3)) * 10
    charge_coords = rng.normal(size=(n_charges, 3)) * 10
    charges = np.array([-1.0, -1.0, 1.0])
    aa_charge = np.zeros(n_aa); aa_charge[3] = -1.0; aa_charge[9] = 1.0
    k, debye_length = 17.3636, 10.0
    field = external_charge_field(residue_coords, charge_coords, charges, aa_charge, k, debye_length)
    dist = cdist(residue_coords, charge_coords)
    potential = (np.exp(-dist / debye_length) / dist) @ charges
    np.testing.assert_allclose(field, -k * np.outer(potential, aa_charge))


def test_dna_charge_field_shifts_frustration(sparse_model):
    """Putting a charge (such as from DNA) close to a residue changes its frustration, so the
    result with the charge differs from the result without it."""
    model = sparse_model
    active = np.arange(model.N)
    base = model.frustration(kind='singleresidue', active_residues=active)
    charge_coords = model._cb_coords[:1] + np.array([3.0, 0.0, 0.0])  # a charge ~3 A from residue 0
    with_dna = model.frustration(kind='singleresidue', active_residues=active,
                                 charge_coords=charge_coords, charges=np.array([-1.0]))
    assert not np.allclose(base, with_dna)


def test_dna_charge_field_matches_explicit_potts(sparse_model):
    """Two ways of adding an external charge agree: using the built-in charge option gives the
    same frustration as building that charge field by hand and adding it to the model."""
    from frustratometer.frustration import numba as fn
    from frustratometer.frustration.frustration import compute_seq_index
    model = sparse_model
    rng = np.random.default_rng(7)
    charge_coords = model._cb_coords[[5, 20, 40]] + rng.normal(size=(3, 3))  # charges near a few residues
    charges = np.array([-1.0, -1.0, -1.0])
    dna_field = model._charge_field(charge_coords, charges, None, None)
    potts_with_field = {**model.sparse_potts_model,
                        'h': model.sparse_potts_model['h'] + dna_field}
    seq_index = compute_seq_index(model.sequence)
    expected = fn.singleresidue_frustration_potts(seq_index, potts_with_field, model.aa_freq)
    got = model.frustration(kind='singleresidue', active_residues=np.arange(model.N),
                            charge_coords=charge_coords, charges=charges)
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=1e-5)


def test_electrostatics_not_supported():
    """Freezing part of the model isn't supported yet when electrostatics are turned on, so
    it raises an error."""
    structure = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A')
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_contact=2,
                                 k_electrostatics=4.184, min_sequence_separation_electrostatics=1)
    with pytest.raises(NotImplementedError):
        model.fold_static_context(np.arange(0, model.N, 2))
