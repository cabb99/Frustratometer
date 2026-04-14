"""
Tests for PDB repair and Structure auto-repair logic.

Uses 6u5e.pdb (complete) and 2GHY.pdb (incomplete — missing residue) as test data.
"""
import pytest
from pathlib import Path
import frustratometer
from frustratometer.classes.Structure import SparseMatrix

test_data_path = Path(__file__).parent / 'data'


@pytest.mark.parametrize("pdb,chain,repair_pdb,expect_repair", [
    ("6u5e.pdb", "A", None,  False),  # complete, auto-detect -> no repair needed
    ("6u5e.pdb", "A", False, False),  # complete, skip repair -> fine
    ("6u5e.pdb", "A", True,  True),   # complete, force repair -> calls pdbfixer
    ("2GHY.pdb", "A", None,  True),   # incomplete, auto-detect -> repairs automatically
    ("2GHY.pdb", "A", True,  True),   # incomplete, force repair -> calls pdbfixer
])
def test_structure_valid(pdb, chain, repair_pdb, expect_repair, tmp_path):
    """Structure should be consistent and only call PDBFixer when expected."""
    s = frustratometer.Structure(test_data_path / pdb, chain,
                                 repair_pdb=repair_pdb, pdb_directory=tmp_path)
    dm = s.distance_matrix
    L = dm.shape if isinstance(dm, SparseMatrix) else dm.shape[0]
    assert len(s.sequence) == L
    cleaned_files = list(tmp_path.glob("*_cleaned.pdb"))
    was_repaired = len(cleaned_files) > 0
    assert was_repaired == expect_repair, (
        f"Expected PDBFixer {'to be' if expect_repair else 'not to be'} called"
    )


def test_incomplete_pdb_no_repair_raises():
    """An incomplete PDB with repair_pdb=False should raise an informative error."""
    with pytest.raises(ValueError, match="repair_pdb=True"):
        frustratometer.Structure(test_data_path / '2GHY.pdb', 'A', repair_pdb=False)


@pytest.mark.parametrize("repair_pdb", [None, True, False])
def test_from_pdb_list(repair_pdb, tmp_path):
    """Load multiple PDBs via from_pdb_list with different repair modes."""
    pdb_files = [test_data_path / '6u5e.pdb', test_data_path / '2GHY.pdb']
    if repair_pdb is False:
        # 2GHY is incomplete -> False should fail
        with pytest.raises(ValueError, match="repair_pdb=True"):
            frustratometer.Structure.from_pdb_list(pdb_files, 'A',
                                                   pdb_directory=tmp_path,
                                                   repair_pdb=False)
    else:
        structures = frustratometer.Structure.from_pdb_list(pdb_files, 'A',
                                                            pdb_directory=tmp_path,
                                                            repair_pdb=repair_pdb)
        assert len(structures) == 2
        for s in structures:
            dm = s.distance_matrix
            L = dm.shape if isinstance(dm, SparseMatrix) else dm.shape[0]
            assert len(s.sequence) == L


def test_repair_pdb_no_memory_leak(tmp_path):
    """Repairing multiple PDBs should not leak memory significantly."""
    import resource
    from frustratometer.pdb import repair_pdb

    # Warm up (first call loads OpenMM libs)
    repair_pdb(test_data_path / '6u5e.pdb', 'A', tmp_path)
    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for i in range(3):
        repair_pdb(test_data_path / '6u5e.pdb', 'A', tmp_path)

    final_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth_mb = (final_kb - baseline_kb) / 1024
    # Each uncontained leak is ~170 MB; 3 calls would be ~510 MB.
    # With multiprocessing isolation the parent process doesn't grow.
    assert growth_mb < 200, f"Memory grew by {growth_mb:.0f} MB over 3 repairs — possible leak"


def _measure_pdbfixer_leak(pdb_path, out_path, result):
    """Measure PDBFixer memory leak in an isolated process (child target)."""
    import gc, resource
    from frustratometer.pdb.fix import _repair_worker

    _repair_worker(pdb_path, 'A', out_path)
    gc.collect()
    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for _ in range(3):
        _repair_worker(pdb_path, 'A', out_path)
        gc.collect()

    final_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.value = (final_kb - baseline_kb) / 1024


@pytest.mark.memory_heavy
def test_pdbfixer_leaks_without_isolation(tmp_path):
    """Canary: calling PDBFixer in-process DOES leak memory.

    Good news: If this test starts failing, pdbfixer/OpenMM may have fixed their leak and
    the multiprocessing isolation in fix.py can be removed.

    Runs the measurement in a child process so that the rest of the test
    suite cannot inflate ``ru_maxrss`` and mask the leak.
    """
    import multiprocessing

    growth_mb = multiprocessing.Value('d', 0.0)
    p = multiprocessing.Process(
        target=_measure_pdbfixer_leak,
        args=(str(test_data_path / '6u5e.pdb'),
              str(tmp_path / 'cleaned.pdb'),
              growth_mb),
    )
    p.start()
    p.join(timeout=300)
    assert p.exitcode == 0, f"Measurement process failed (exit code {p.exitcode})"
    assert growth_mb.value >= 200, (
        f"Memory only grew {growth_mb.value:.0f} MB. PDBFixer leak may have been fixed. "
        f"Consider removing multiprocessing isolation in fix.py."
    )

def test_structure_detects_chain_breaks():
    """Structure must detect chain boundaries from a multi-chain CIF file."""

    cif_path = Path(__file__).parent / "data" / "5msm.cif"
    if not cif_path.exists():
        pytest.skip("5msm.cif not available")

    s = frustratometer.Structure(cif_path, chain=None)
    # 5msm has 6 chains (A,B,C,D,E,F) -> 5 break points
    assert s.chain_breaks is not None
    assert len(s.chain_breaks) == 5