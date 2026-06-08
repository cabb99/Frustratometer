"""A/B benchmark for `frustration.write_tcl_script`.

Compares the current (optimized) implementation against a verbatim copy of the
previous one (`_old_write_tcl_script` below) on two axes:

  1. Python generation time -- the old version ran two prody `selection.select(...)`
     queries per minimally-frustrated contact; the new one precomputes coordinates once.
  2. VMD script-load time (text mode) -- the old script emitted two `atomselect ... get`
     lookups per contact line (VMD re-parses each selection); the new one writes literal
     coordinates. (`display update off/on` was tried and measured to make no difference
     for script-sourced loading, so it was dropped.)

Per-frame movie rendering is *unchanged* (same lines, same 360 frames), so this measures
the setup costs we actually optimized, not the frame loop.

Run from the project root:
    python devtools/benchmark_tcl.py
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import prody

import frustratometer
from frustratometer.frustration import frustration

prody.confProDy(verbosity='none')

# Single-chain PDBs whose residue count matches the frustration-matrix dimension.
# (write_tcl_script assumes len(residues) == L; multi-segment PDBs like 5msm/4wnc
# violate that and break the old and new code alike -- a separate, pre-existing issue.)
PDBS = ['tests/data/6u5e.pdb', 'tests/data/3ptn.pdb']
CHAIN = 'A'


def _old_write_tcl_script(pdb_file, chain, mask, distance_matrix, distance_cutoff,
                          single_frustration, pair_frustration, tcl_script='frustration.tcl',
                          max_connections=None):
    """Verbatim pre-optimization implementation (generation portion only)."""
    fo = open(tcl_script, 'w+')
    single_frustration = np.nan_to_num(single_frustration, nan=0, posinf=0, neginf=0)
    pair_frustration = np.nan_to_num(pair_frustration, nan=0, posinf=0, neginf=0)

    structure = prody.parsePDB(str(pdb_file))
    selection = structure.select('protein', chain=chain)
    residues = np.unique(selection.getResindices())

    fo.write('[atomselect top all] set beta 0\n')
    for r, f in zip(residues, single_frustration):
        fo.write(f'[atomselect top "residue {int(r)}"] set beta {f}\n')

    r1, r2 = np.meshgrid(residues, residues, indexing='ij')
    sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel(),
                                distance_matrix.ravel(), mask.ravel()]).T
    if distance_cutoff:
        mask_dist = (sel_frustration[:, -2] <= distance_cutoff)
    else:
        mask_dist = np.ones(len(sel_frustration), dtype=bool)
    sel_frustration = sel_frustration[mask_dist & (sel_frustration[:, -1] > 0)]

    minimally_frustrated = sel_frustration[sel_frustration[:, 2] < -0.78]
    minimally_frustrated = minimally_frustrated[np.argsort(minimally_frustrated[:, 2])]
    if max_connections:
        minimally_frustrated = minimally_frustrated[:max_connections]
    fo.write('draw color green\n')
    for (r1, r2, f, d, m) in minimally_frustrated:
        r1, r2 = int(r1), int(r2)
        if abs(r1 - r2) == 1:
            continue
        pos1 = selection.select(f'resindex {r1} and (name CB or (resname GLY and name CA))').getCoords()[0]
        pos2 = selection.select(f'resindex {r2} and (name CB or (resname GLY and name CA))').getCoords()[0]
        distance = np.linalg.norm(pos1 - pos2)
        if d > 9.5 or d < 3.5:
            continue
        fo.write(f'lassign [[atomselect top "residue {r1} and name CA"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "residue {r2} and name CA"] get {{x y z}}] pos2\n')
        fo.write(f'draw line $pos1 $pos2 style {"solid" if 3.5 <= distance <= 6.5 else "dashed"} width 2\n')

    frustrated = sel_frustration[sel_frustration[:, 2] > 1]
    frustrated = frustrated[np.argsort(frustrated[:, 2])[::-1]]
    if max_connections:
        frustrated = frustrated[:max_connections]
    fo.write('draw color red\n')
    for (r1, r2, f, d, m) in frustrated:
        r1, r2 = int(r1), int(r2)
        if d > 9.5 or d < 3.5:
            continue
        fo.write(f'lassign [[atomselect top "residue {r1} and name CA"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "residue {r2} and name CA"] get {{x y z}}] pos2\n')
        fo.write(f'draw line $pos1 $pos2 style {"solid" if 3.5 <= d <= 6.5 else "dashed"} width 2\n')

    fo.write('mol delrep top 0\nmol representation NewCartoon\nmol addrep top\n')
    fo.close()
    return tcl_script


def _time(fn, n):
    best = float('inf')
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _vmd_load_seconds(tcl_path, pdb):
    """Time a headless VMD load of a script (with `exit` appended), best of 2."""
    with open(tcl_path) as f:
        body = f.read()
    runnable = tcl_path + '.run.tcl'
    with open(runnable, 'w') as f:
        f.write(body + '\nexit\n')
    best = float('inf')
    for _ in range(2):
        t0 = time.perf_counter()
        subprocess.run(['vmd', '-dispdev', 'text', '-e', runnable, pdb],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
        best = min(best, time.perf_counter() - t0)
    return best


def _count_draw(path):
    return sum(1 for ln in open(path) if ln.lstrip().startswith('draw line'))


def main():
    have_vmd = shutil.which('vmd') is not None
    tmp = Path(tempfile.mkdtemp(prefix='tclbench_'))

    for pdb in PDBS:
        if not Path(pdb).exists():
            print(f'Skipping {pdb} (not found)')
            continue

        structure = frustratometer.Structure(pdb, CHAIN, sparse=False, repair_pdb=False)
        L = len(structure.sequence)
        # Dense model: write_tcl_script needs a dense distance_matrix/mask.
        model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5,
                                     min_sequence_separation_contact=2, k_electrostatics=0)
        n_res = len(np.unique(prody.parsePDB(str(model.pdb_file))
                              .select('protein', chain=CHAIN).getResindices()))
        if n_res != model.distance_matrix.shape[0]:
            print(f'Skipping {Path(pdb).name}: {n_res} residues != matrix dim '
                  f'{model.distance_matrix.shape[0]} (write_tcl_script needs them equal)\n')
            continue

        single = -model.frustration(kind='singleresidue')
        pair = -model.frustration(kind='mutational')
        args = (model.pdb_file, model.chain, model.mask, model.distance_matrix,
                model.distance_cutoff, single, pair)

        old_tcl, new_tcl = str(tmp / 'old.tcl'), str(tmp / 'new.tcl')
        n = 20 if L < 500 else 5

        t_old = _time(lambda: _old_write_tcl_script(*args, tcl_script=old_tcl), n)
        t_new = _time(lambda: frustration.write_tcl_script(*args, tcl_script=new_tcl), n)
        n_old, n_new = _count_draw(old_tcl), _count_draw(new_tcl)

        print('=' * 70)
        print(f'  {Path(pdb).name}   (L={L})')
        print('=' * 70)
        print(f'draw lines:        old={n_old:6d}   new={n_new:6d}   (new dedups (i,j)/(j,i))')
        print(f'python generation: old={t_old*1000:8.2f}ms  new={t_new*1000:8.2f}ms  '
              f'speedup={t_old/t_new:5.1f}x')

        if have_vmd:
            try:
                v_old = _vmd_load_seconds(old_tcl, pdb)
                v_new = _vmd_load_seconds(new_tcl, pdb)
                print(f'vmd script load:   old={v_old:8.2f}s   new={v_new:8.2f}s   '
                      f'speedup={v_old/v_new:5.1f}x   (text mode, startup included)')
            except Exception as e:
                print(f'vmd script load:   skipped ({e})')
        else:
            print('vmd script load:   skipped (vmd not on PATH)')
        print()

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
