"""Tests for static-context reduction (frustratometer.awsem.static_context).

A static context pins part of the protein to its native identity; its couplings to
the active residues fold into active fields and static-static couplings into a constant
offset. The reduced model evaluated on the active subsequence (plus offset) must equal
the full model evaluated with the static positions held native.
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
    s = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A')
    return frustratometer.AWSEM(s, distance_cutoff_contact=9.5,
                                min_sequence_separation_contact=2, k_electrostatics=0)


def _seq_index(seq):
    return np.array([_AA.index(c) for c in seq])


def test_native_energy_equivalence(sparse_model):
    m = sparse_model
    si = _seq_index(m.sequence)
    rng = np.random.default_rng(0)
    active = np.sort(rng.choice(m.N, size=m.N // 2, replace=False))
    reduced, offset = fold_static_context(m.sparse_potts_model, si, active)
    active_seq = ''.join(m.sequence[i] for i in active)
    got = compute_native_energy_sparse(active_seq, reduced) + offset
    np.testing.assert_allclose(got, m.native_energy(), rtol=1e-9, atol=1e-8)


def test_mutant_active_matches_full_sequence(sparse_model):
    """Reduced energy of a mutated active subsequence == full energy with those mutations
    applied and static positions held native."""
    m = sparse_model
    si = _seq_index(m.sequence)
    rng = np.random.default_rng(1)
    active = np.sort(rng.choice(m.N, size=m.N // 2, replace=False))
    reduced, offset = fold_static_context(m.sparse_potts_model, si, active)
    for _ in range(5):
        sub = rng.integers(1, 21, size=len(active))  # avoid gap (index 0)
        full = list(m.sequence)
        for k, i in enumerate(active):
            full[i] = _AA[sub[k]]
        e_full = compute_native_energy_sparse(''.join(full), m.sparse_potts_model)
        e_red = compute_native_energy_sparse(''.join(_AA[i] for i in sub), reduced) + offset
        np.testing.assert_allclose(e_red, e_full, rtol=1e-9, atol=1e-8)


def test_edge_cases(sparse_model):
    m = sparse_model
    si = _seq_index(m.sequence)
    full_nc = len(m.sparse_potts_model['contact_i'])

    # all active: zero offset, all contacts retained
    reduced, offset = fold_static_context(m.sparse_potts_model, si, np.arange(m.N))
    assert offset == pytest.approx(0.0, abs=1e-9)
    assert len(reduced['contact_i']) == full_nc

    # all static: offset equals the full native energy, no contacts remain
    reduced, offset = fold_static_context(m.sparse_potts_model, si, np.array([], dtype=int))
    np.testing.assert_allclose(offset, m.native_energy(), rtol=1e-9, atol=1e-8)
    assert len(reduced['contact_i']) == 0


def test_matches_awsem_energy_selected():
    """Reduced model equals the canonical AwsemEnergySelected."""
    from frustratometer.optimization.optimization import AwsemEnergySelected
    s = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A')
    m = frustratometer.AWSEM(s, distance_cutoff_contact=9.5,
                             min_sequence_separation_contact=2, k_electrostatics=0)
    sparse_potts = m.sparse_potts_model
    si = _seq_index(m.sequence)
    rng = np.random.default_rng(2)
    active = np.sort(rng.choice(m.N, size=m.N // 3, replace=False))
    reduced, offset = fold_static_context(sparse_potts, si, active)

    # AwsemEnergySelected consumes the dense Potts view and a dense mask.
    m.mask = m.mask.to_dense(fill=0.0)
    sel = AwsemEnergySelected(m, selection=active, use_numba=False)
    for _ in range(4):
        sub = rng.integers(1, 21, size=len(active))
        e_red = compute_native_energy_sparse(''.join(_AA[i] for i in sub), reduced) + offset
        e_sel = sel.energy(sub)
        np.testing.assert_allclose(e_red, e_sel, rtol=1e-7, atol=1e-7)


def test_awsem_fold_method_matches_function(sparse_model):
    m = sparse_model
    active = np.arange(0, m.N, 2)
    r1, o1 = m.fold_static_context(active)
    r2, o2 = fold_static_context(m.sparse_potts_model, _seq_index(m.sequence), active)
    assert o1 == pytest.approx(o2)
    np.testing.assert_array_equal(r1['contact_i'], r2['contact_i'])
    np.testing.assert_allclose(r1['h'], r2['h'])


def test_active_singleresidue_equals_full_restricted(sparse_model):
    """Single-residue frustration with a static context equals the full-model single-residue
    frustration restricted to the active residues (single-residue decoys hold all others native)."""
    m = sparse_model
    full = m.frustration(kind='singleresidue')
    rng = np.random.default_rng(3)
    active = np.sort(rng.choice(m.N, size=m.N // 2, replace=False))
    sub = m.frustration(kind='singleresidue', active_residues=active)
    assert sub.shape == (len(active),)
    np.testing.assert_allclose(sub, full[active], rtol=1e-6, atol=1e-6)


def test_active_mutational_equals_full_restricted(sparse_model):
    """Mutational frustration over active residues equals the full (N,N) mutational restricted
    to the active-active sub-block."""
    m = sparse_model
    full = m.frustration(kind='mutational')
    rng = np.random.default_rng(4)
    active = np.sort(rng.choice(m.N, size=m.N // 2, replace=False))
    sub = m.frustration(kind='mutational', active_residues=active)
    assert sub.shape == (len(active), len(active))
    np.testing.assert_allclose(sub, full[np.ix_(active, active)], rtol=1e-6, atol=1e-6)


def test_active_residues_boolean_mask(sparse_model):
    m = sparse_model
    mask = np.zeros(m.N, dtype=bool)
    mask[::2] = True
    sub = m.frustration(kind='singleresidue', active_residues=mask)
    np.testing.assert_allclose(sub, m.frustration(kind='singleresidue')[np.where(mask)[0]],
                               rtol=1e-6, atol=1e-6)


def test_active_residues_configurational_raises(sparse_model):
    with pytest.raises(NotImplementedError):
        sparse_model.frustration(kind='configurational', active_residues=np.arange(0, sparse_model.N, 2))


def test_select_residues_string_resolves(sparse_model):
    m = sparse_model
    idx = m.select_residues('resid 10 to 40')
    assert idx.dtype.kind == 'i'
    assert 0 < len(idx) < m.N
    # index/mask inputs pass through
    np.testing.assert_array_equal(m.select_residues(idx), idx)
    mask = np.zeros(m.N, bool); mask[idx] = True
    np.testing.assert_array_equal(m.select_residues(mask), idx)


def test_active_selection_string_matches_indices(sparse_model):
    m = sparse_model
    idx = m.select_residues('resid 10 to 60')
    a = m.frustration(kind='singleresidue', active_selection='resid 10 to 60')
    b = m.frustration(kind='singleresidue', active_residues=idx)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_static_selection_is_active_complement(sparse_model):
    m = sparse_model
    static_idx = set(m.select_residues('resid 1 to 30').tolist())
    active_idx = np.array([i for i in range(m.N) if i not in static_idx], dtype=np.intp)
    a = m.frustration(kind='singleresidue', static_selection='resid 1 to 30')
    b = m.frustration(kind='singleresidue', active_residues=active_idx)
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


def test_external_charge_field_formula():
    from frustratometer.awsem.physics import external_charge_field
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(0)
    N, M, Q = 6, 3, 21
    rc = rng.normal(size=(N, 3)) * 10
    cc = rng.normal(size=(M, 3)) * 10
    q = np.array([-1.0, -1.0, 1.0])
    aac = np.zeros(Q); aac[3] = -1.0; aac[9] = 1.0
    k, L = 17.3636, 10.0
    f = external_charge_field(rc, cc, q, aac, k, L)
    d = cdist(rc, cc); ind = np.exp(-d / L) / d; phi = ind @ q
    np.testing.assert_allclose(f, -k * np.outer(phi, aac))


def test_dna_charge_field_shifts_frustration(sparse_model):
    m = sparse_model
    active = np.arange(m.N)
    base = m.frustration(kind='singleresidue', active_residues=active)
    cc = m._cb_coords[:1] + np.array([3.0, 0.0, 0.0])  # a charge ~3 A from residue 0
    with_dna = m.frustration(kind='singleresidue', active_residues=active,
                             charge_coords=cc, charges=np.array([-1.0]))
    assert not np.allclose(base, with_dna)


def test_dna_charge_field_matches_explicit_potts(sparse_model):
    """Frustration with an external charge field == frustration on a Potts model whose h has
    the same field added (independent explicit-J computation)."""
    from frustratometer.frustration import numba as fn
    from frustratometer.frustration.frustration import compute_seq_index
    m = sparse_model
    rng = np.random.default_rng(7)
    cc = m._cb_coords[[5, 20, 40]] + rng.normal(size=(3, 3))  # charges near a few residues
    q = np.array([-1.0, -1.0, -1.0])
    h_dna = m._charge_field(cc, q, None, None)
    potts2 = {**m.sparse_potts_model, 'h': m.sparse_potts_model['h'] + h_dna}
    si = compute_seq_index(m.sequence)
    ref = fn.singleresidue_frustration_potts(si, potts2, m.aa_freq)
    got = m.frustration(kind='singleresidue', active_residues=np.arange(m.N),
                        charge_coords=cc, charges=q)
    np.testing.assert_allclose(got, ref, atol=1e-6, rtol=1e-5)


def test_electrostatics_not_supported():
    s = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A')
    m = frustratometer.AWSEM(s, distance_cutoff_contact=9.5, min_sequence_separation_contact=2,
                             k_electrostatics=4.184, min_sequence_separation_electrostatics=1)
    with pytest.raises(NotImplementedError):
        m.fold_static_context(np.arange(0, m.N, 2))
