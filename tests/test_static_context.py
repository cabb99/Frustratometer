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
from frustratometer.awsem.static_context import fold_static_context

test_data_path = 'tests/data'


@pytest.fixture(scope="module")
def sparse_model():
    s = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A', sparse=True)
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
    """Reduced model equals the canonical AwsemEnergySelected on a dense model."""
    from frustratometer.optimization.optimization import AwsemEnergySelected
    s = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A', sparse=False)
    m = frustratometer.AWSEM(s, distance_cutoff_contact=9.5,
                             min_sequence_separation_contact=2, k_electrostatics=0, sparse=False)
    sparse_potts = frustration.potts_model_dense_to_sparse(m.potts_model, m.mask)
    si = _seq_index(m.sequence)
    rng = np.random.default_rng(2)
    active = np.sort(rng.choice(m.N, size=m.N // 3, replace=False))
    reduced, offset = fold_static_context(sparse_potts, si, active)

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


def test_electrostatics_not_supported():
    s = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A', sparse=True)
    m = frustratometer.AWSEM(s, distance_cutoff_contact=9.5, min_sequence_separation_contact=2,
                             k_electrostatics=4.184, min_sequence_separation_electrostatics=1)
    with pytest.raises(NotImplementedError):
        m.fold_static_context(np.arange(0, m.N, 2))
