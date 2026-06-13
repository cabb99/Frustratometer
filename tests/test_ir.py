"""Tests for the Potts intermediate representation (frustratometer.frustration.ir).

The IR is the shared narrow waist between physics front-ends (AWSEM, DCA) and the
compute backends. These tests pin its two coupling-storage modes and the dense /
sparse-dict round-trips that downstream backends rely on.
"""
import numpy as np
import pytest

from frustratometer.frustration.ir import PottsIR, CouplingBlock


def _explicit_block(rng, Nb=5, L=8, Q=21):
    ci = rng.integers(0, L, Nb)
    cj = rng.integers(0, L, Nb)
    J = rng.normal(size=(Nb, Q, Q))
    return ci, cj, J


def test_block_requires_exactly_one_mode():
    ci = np.array([0]); cj = np.array([1])
    with pytest.raises(ValueError):
        CouplingBlock(ci, cj)  # neither J nor (coeff, G)
    with pytest.raises(ValueError):
        CouplingBlock(ci, cj, J=np.zeros((1, 2, 2)),
                      coeff=np.zeros((1, 1)), G=np.zeros((1, 2, 2)))  # both


def test_explicit_block_explicit_J_applies_scale():
    rng = np.random.default_rng(0)
    ci, cj, J = _explicit_block(rng)
    b = CouplingBlock(ci, cj, J=J, scale=2.0)
    assert b.mode == 'explicit'
    np.testing.assert_allclose(b.explicit_J(), 2.0 * J)


def test_factored_block_matches_manual_einsum():
    rng = np.random.default_rng(1)
    Nb, T, Q = 6, 3, 21
    ci = rng.integers(0, 10, Nb); cj = rng.integers(0, 10, Nb)
    coeff = rng.normal(size=(Nb, T))
    G = rng.normal(size=(T, Q, Q))
    b = CouplingBlock(ci, cj, coeff=coeff, G=G, scale=1.5)
    assert b.mode == 'factored'
    manual = 1.5 * np.einsum('bt,tij->bij', coeff, G)
    np.testing.assert_allclose(b.explicit_J(), manual)


def test_dense_roundtrip_and_sparse_dict():
    rng = np.random.default_rng(2)
    L, Q, Nb = 7, 21, 5
    h = rng.normal(size=(L, Q))
    ci, cj, J = _explicit_block(rng, Nb=Nb, L=L, Q=Q)
    d = {'h': h, 'J': J, 'contact_i': ci, 'contact_j': cj, 'L': L}

    ir = PottsIR.from_sparse_potts_dict(d)
    # dense materialization places each block's J at its (i, j) entries
    Jdense = ir.to_dense()['J']
    ref = np.zeros((L, L, Q, Q))
    np.add.at(ref, (ci, cj), J)
    np.testing.assert_allclose(Jdense, ref)

    # round-trip back to the legacy sparse dict
    out = ir.to_sparse_potts_dict()
    np.testing.assert_allclose(out['J'], J)
    np.testing.assert_array_equal(out['contact_i'], ci)


def test_multiblock_sparse_dict_concatenates():
    rng = np.random.default_rng(3)
    L, Q = 6, 21
    h = rng.normal(size=(L, Q))
    b1 = CouplingBlock(*_explicit_block(rng, Nb=4, L=L, Q=Q))
    b2c = rng.integers(0, L, 3)
    b2 = CouplingBlock(b2c, rng.integers(0, L, 3),
                       coeff=rng.normal(size=(3, 2)), G=rng.normal(size=(2, Q, Q)))
    ir = PottsIR(L=L, q=Q, h=h, blocks=[b1, b2])
    out = ir.to_sparse_potts_dict()
    assert out['J'].shape == (4 + 3, Q, Q)
    assert len(out['contact_i']) == 7
