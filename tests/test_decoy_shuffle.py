"""Each frustration kind should equal the decoy shuffle it stands for.

Configurational frustration is defined by a shuffle of contacts (random positions, identities,
and geometry). The other kinds are computed analytically, but each one still corresponds to a
specific way of shuffling the sequence: singleresidue mutates one residue, mutational mutates
the two residues of a contact. These tests build that shuffle by brute force -- making real
mutated sequences and scoring them with the energy function (independent of the analytic decoy
machinery) -- and check the analytic frustration matches it, for the whole protein and for a
selection (frustration measured "in context", with the rest of the protein held native).

Pseudoconfigurational is the configurational shuffle done on the Potts model: it samples a
coupling from the whole contact pool (any pair could be a contact) plus fields and identities,
so on the contacts it is defined on it reproduces the physical configurational frustration.
"""
import numpy as np
import pytest

import frustratometer
from frustratometer.frustration.frustration import _AA, compute_native_energy_sparse, compute_aa_freq

test_data_path = 'tests/data'


@pytest.fixture(scope="module")
def model():
    s = frustratometer.Structure(f'{test_data_path}/6u5e.pdb', 'A')
    return frustratometer.AWSEM(s, distance_cutoff_contact=9.5,
                                min_sequence_separation_contact=2, k_electrostatics=0)


def _mutate(seq, mutations):
    chars = list(seq)
    for pos, code in mutations:
        chars[pos] = _AA[code]
    return ''.join(chars)


def _shuffle_frustration(model, decoy_sites, aa_freq, n_decoys, seed):
    """Frustration from a brute-force mutation shuffle. ``decoy_sites`` is a list of residue
    tuples (one residue for singleresidue, two for a contact); for each, draw amino acids from
    ``aa_freq``, build the mutated full sequence (every other residue native), score it, and
    return (mean_decoy - native) / std_decoy -- the same index the analytic computes."""
    potts = model.sparse_potts_model
    seq = model.sequence
    native = compute_native_energy_sparse(seq, potts)
    weights = np.asarray(aa_freq, dtype=float)
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    out = np.empty(len(decoy_sites))
    for k, site in enumerate(decoy_sites):
        draws = [rng.choice(len(weights), size=n_decoys, p=weights) for _ in site]
        energies = np.array([
            compute_native_energy_sparse(_mutate(seq, list(zip(site, codes))), potts)
            for codes in zip(*draws)])
        out[k] = (energies.mean() - native) / energies.std()
    return out


def _assert_matches_shuffle(mc, analytic, atol=0.2, min_corr=0.99):
    assert np.corrcoef(mc, analytic)[0, 1] > min_corr
    np.testing.assert_allclose(mc, analytic, atol=atol)


def test_singleresidue_matches_mutation_shuffle(model):
    """Singleresidue frustration equals the shuffle that mutates one residue at a time, drawing
    amino acids from the whole protein's composition."""
    analytic = model.frustration(kind='singleresidue')
    rng = np.random.default_rng(0)
    residues = np.sort(rng.choice(model.N, size=10, replace=False))
    mc = _shuffle_frustration(model, [(int(i),) for i in residues], model.aa_freq,
                              n_decoys=3000, seed=1)
    _assert_matches_shuffle(mc, analytic[residues])


def test_mutational_matches_pair_shuffle(model):
    """Mutational frustration of a contact equals the shuffle that mutates both residues of the
    contact at once."""
    analytic = model.frustration(kind='mutational')
    potts = model.sparse_potts_model
    contact_i = np.asarray(potts['contact_i']); contact_j = np.asarray(potts['contact_j'])
    rng = np.random.default_rng(0)
    pick = rng.choice(len(contact_i), size=10, replace=False)
    pairs = [(int(contact_i[k]), int(contact_j[k])) for k in pick]
    mc = _shuffle_frustration(model, pairs, model.aa_freq, n_decoys=3000, seed=2)
    analytic_vals = np.array([analytic[i, j] for i, j in pairs])
    _assert_matches_shuffle(mc, analytic_vals)


def test_singleresidue_selection_matches_in_context_shuffle(model):
    """Singleresidue frustration on a selection equals the in-context shuffle: mutate an active
    residue while every other residue (active or static) keeps its native identity."""
    rng = np.random.default_rng(0)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    analytic = model.frustration(kind='singleresidue', active_residues=active)
    local = rng.choice(len(active), size=8, replace=False)
    mc = _shuffle_frustration(model, [(int(active[i]),) for i in local], model.aa_freq,
                              n_decoys=3000, seed=3)
    _assert_matches_shuffle(mc, analytic[local])


def test_mutational_selection_matches_in_context_shuffle(model):
    """Mutational frustration on a selection equals the in-context shuffle over active-active
    contacts: mutate both residues of an active-active contact, the rest held native."""
    rng = np.random.default_rng(0)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    analytic = model.frustration(kind='mutational', active_residues=active)
    potts = model.sparse_potts_model
    contact_i = np.asarray(potts['contact_i']); contact_j = np.asarray(potts['contact_j'])
    active_set = set(active.tolist())
    aa_contacts = [(int(i), int(j)) for i, j in zip(contact_i, contact_j)
                   if i in active_set and j in active_set]
    pick = np.random.default_rng(1).choice(len(aa_contacts), size=8, replace=False)
    pairs = [aa_contacts[k] for k in pick]
    mc = _shuffle_frustration(model, pairs, model.aa_freq, n_decoys=3000, seed=4)
    local_of = {int(r): k for k, r in enumerate(active)}
    analytic_vals = np.array([analytic[local_of[i], local_of[j]] for i, j in pairs])
    _assert_matches_shuffle(mc, analytic_vals)


def test_singleresidue_active_scope_matches_active_composition_shuffle(model):
    """With decoy_scope='active' the shuffle draws amino acids from the active residues' own
    composition instead of the whole protein's. The analytic active-scope frustration matches
    that shuffle and differs from the whole-protein scope."""
    rng = np.random.default_rng(0)
    active = np.sort(rng.choice(model.N, size=model.N // 2, replace=False))
    whole = model.frustration(kind='singleresidue', active_residues=active, decoy_scope='whole')
    active_scope = model.frustration(kind='singleresidue', active_residues=active, decoy_scope='active')
    assert not np.allclose(whole, active_scope)
    active_seq = ''.join(model.sequence[i] for i in active)
    active_freq = compute_aa_freq(active_seq, include_gaps=False)
    local = rng.choice(len(active), size=8, replace=False)
    mc = _shuffle_frustration(model, [(int(active[i]),) for i in local], active_freq,
                              n_decoys=3000, seed=5)
    _assert_matches_shuffle(mc, active_scope[local])


def test_pseudoconfigurational_matches_configurational(model):
    """Pseudoconfigurational is the configurational shuffle done on the Potts model (sample a
    coupling from the whole contact pool, plus fields and identities). On the contacts it is
    defined on, it reproduces the physical Monte-Carlo configurational frustration -- unlike the
    old per-pair mean-field version, which tracked mutational instead."""
    pseudo = model.frustration(kind='pseudoconfigurational', n_decoys=20000)
    config = model.frustration(kind='configurational')
    contact_i = np.asarray(model.sparse_potts_model['contact_i'])
    contact_j = np.asarray(model.sparse_potts_model['contact_j'])
    on_potts = np.zeros(pseudo.shape, dtype=bool)
    on_potts[contact_i, contact_j] = True
    both = on_potts & np.isfinite(config)
    assert np.corrcoef(pseudo[both], config[both])[0, 1] > 0.95


def test_pseudoconfigurational_seeded_and_dense_sparse_consistent(model):
    """Pseudoconfigurational is now a seeded Monte-Carlo estimate: the same seed reproduces the
    same values, and the sparse (per-contact) and dense (L, L) forms agree."""
    a = model.frustration(kind='pseudoconfigurational', seed=5)
    b = model.frustration(kind='pseudoconfigurational', seed=5)
    np.testing.assert_array_equal(a, b)
    sparse = model.frustration(kind='pseudoconfigurational', seed=5, dense=False)
    np.testing.assert_allclose(sparse.to_dense(fill=0.0), a)
