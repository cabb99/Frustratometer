"""
Tests for PDB repair and Structure auto-repair logic.

Uses 6u5e.pdb (complete) and 2GHY.pdb (incomplete — missing residue) as test data.
"""
import pytest
from pathlib import Path
import frustratometer

test_data_path = Path(__file__).parent / 'data'


@pytest.mark.parametrize("pdb,chain,repair_pdb,expect_repair", [
    ("6u5e.pdb", "A", None,  False),  # complete, auto-detect → no repair needed
    ("6u5e.pdb", "A", False, False),  # complete, skip repair → fine
    ("6u5e.pdb", "A", True,  True),   # complete, force repair → calls pdbfixer
    ("2GHY.pdb", "A", None,  True),   # incomplete, auto-detect → repairs automatically
    ("2GHY.pdb", "A", True,  True),   # incomplete, force repair → calls pdbfixer
])
def test_structure_valid(pdb, chain, repair_pdb, expect_repair, tmp_path):
    """Structure should be consistent and only call PDBFixer when expected."""
    s = frustratometer.Structure(test_data_path / pdb, chain,
                                 repair_pdb=repair_pdb, pdb_directory=tmp_path)
    assert len(s.sequence) == s.distance_matrix.shape[0]
    cleaned_files = list(tmp_path.glob("*_cleaned.pdb"))
    was_repaired = len(cleaned_files) > 0
    assert was_repaired == expect_repair, (
        f"Expected PDBFixer {'to be' if expect_repair else 'not to be'} called"
    )


def test_incomplete_pdb_no_repair_raises():
    """An incomplete PDB with repair_pdb=False should raise an informative error."""
    with pytest.raises(ValueError, match="repair_pdb=True"):
        frustratometer.Structure(test_data_path / '2GHY.pdb', 'A', repair_pdb=False)


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


@pytest.mark.memory_heavy
def test_pdbfixer_leaks_without_isolation(tmp_path):
    """Canary: calling PDBFixer in-process DOES leak memory.
    
    Good news: If this test starts failing, pdbfixer/OpenMM may have fixed their leak and
    the multiprocessing isolation in fix.py can be removed.
    """
    import gc
    import resource
    from frustratometer.pdb.fix import _repair_worker

    pdb = str(test_data_path / '6u5e.pdb')
    out = str(tmp_path / 'cleaned.pdb')

    # Warm up
    _repair_worker(pdb, 'A', out)
    gc.collect()
    baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for _ in range(3):
        _repair_worker(pdb, 'A', out)
        gc.collect()

    final_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth_mb = (final_kb - baseline_kb) / 1024
    # Expect significant growth (~170 MB per call) when running in-process.
    # If this fails (<200 MB), the leak is fixed upstream — great news!
    assert growth_mb >= 200, (
        f"Memory only grew {growth_mb:.0f} MB. PDBFixer leak may have been fixed. "
        f"Consider removing multiprocessing isolation in fix.py."
    )
