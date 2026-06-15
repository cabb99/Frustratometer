"""Tests for the shared PottsModel representation (frustratometer.classes.potts_model)."""
import numpy as np
import pytest

from frustratometer.classes.potts_model import PottsModel, ElectrostaticsSidecar


def _sparse_dict(rng, L=8, Nc=5, Q=21):
    return {'h': rng.normal(size=(L, Q)),
            'J': rng.normal(size=(Nc, Q, Q)),
            'contact_i': rng.integers(0, L, Nc),
            'contact_j': rng.integers(0, L, Nc),
            'L': L}


def test_sparse_dict_roundtrip():
    rng = np.random.default_rng(0)
    d = _sparse_dict(rng)
    m = PottsModel.from_sparse_dict(d)
    assert m.is_sparse and m.Q == 21 and m.Nc == 5 and m.L == 8
    out = m.as_sparse_dict()
    np.testing.assert_array_equal(out['J'], d['J'])
    np.testing.assert_array_equal(out['contact_i'], d['contact_i'])
    assert out['L'] == d['L']


def test_to_dense_matches_scatter_and_existing_util():
    rng = np.random.default_rng(1)
    d = _sparse_dict(rng)
    m = PottsModel.from_sparse_dict(d)
    ref = np.zeros((d['L'], d['L'], 21, 21))
    np.add.at(ref, (d['contact_i'], d['contact_j']), d['J'])
    np.testing.assert_allclose(m.to_dense(), ref)
    # to_dense is cached (same object on second call)
    assert m.to_dense() is m.to_dense()
    # matches the existing converter while it still exists
    from frustratometer import frustration
    np.testing.assert_allclose(m.to_dense(), frustration.potts_model_sparse_to_dense(d)['J'])
    # and as_dense_dict packs h + dense J
    dd = m.as_dense_dict()
    np.testing.assert_array_equal(dd['h'], d['h'])
    np.testing.assert_allclose(dd['J'], ref)


def test_dense_mode():
    rng = np.random.default_rng(2)
    L, Q = 6, 21
    dd = {'h': rng.normal(size=(L, Q)), 'J': rng.normal(size=(L, L, Q, Q))}
    m = PottsModel.from_dense_dict(dd)
    assert not m.is_sparse
    np.testing.assert_array_equal(m.to_dense(), dd['J'])
    with pytest.raises(ValueError):
        m.as_sparse_dict()


def test_electrostatics_sidecar():
    assert ElectrostaticsSidecar.from_elec_data(None) is None
    s = ElectrostaticsSidecar.from_elec_data({'phi': 1})
    assert s.data == {'phi': 1}
