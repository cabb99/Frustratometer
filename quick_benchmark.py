import time
import numpy as np
import frustratometer
from pathlib import Path

# small protein and large protein
pdbs = ['tests/data/6u5e.pdb', 'tests/data/5msm.pdb','tests/data/9qr3.cif']

for pdb in pdbs:
    name = Path(pdb).name
    structure = frustratometer.Structure(pdb, 'A', sparse=True)
    L = len(structure.sequence)
    print(f"===========================================================")
    print(f"    Benchmarking {name} (L={L})    ")
    print(f"===========================================================")

    # Build all three models
    print('Building models...')
    t0 = time.perf_counter()
    model_py = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_contact=2, k_electrostatics=0)
    t_build_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    model_numba = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_contact=2, k_electrostatics=0, fast=True, backend='numba')
    t_build_numba = time.perf_counter() - t0

    t0 = time.perf_counter()
    model_cuda = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_contact=2, k_electrostatics=0, fast=True, backend='cuda')
    t_build_cuda = time.perf_counter() - t0

    print(f'Build times: python={t_build_py:.3f}s  numba={t_build_numba:.3f}s  cuda={t_build_cuda:.3f}s')
    print()

    # Warm up numba/cuda JIT
    _ = model_numba.frustration(kind='singleresidue')
    _ = model_numba.frustration(kind='mutational')
    _ = model_numba.native_energy()
    try:
        _ = model_cuda.frustration(kind='singleresidue')
        _ = model_cuda.frustration(kind='mutational')
        _ = model_cuda.native_energy()
        _ = model_cuda.configurational_frustration()   # warmup configurational JIT
        cuda_ok = True
    except Exception as e:
        print(f'CUDA warmup failed: {e}')
        cuda_ok = False
    print('JIT warmup done')
    print()

    N = 10 if L > 500 else 100

    # --- Native energy ---
    t0 = time.perf_counter()
    for _ in range(N):
        model_py._native_energy = None
        model_py.native_energy()
    t_py = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        model_numba._native_energy = None
        model_numba.native_energy()
    t_nb = (time.perf_counter() - t0) / N

    if cuda_ok:
        t0 = time.perf_counter()
        for _ in range(N):
            model_cuda._native_energy = None
            model_cuda.native_energy()
        t_cu = (time.perf_counter() - t0) / N
    else:
        t_cu = float('nan')

    print(f'native_energy       python={t_py*1000:8.2f}ms  numba={t_nb*1000:8.2f}ms  cuda={t_cu*1000:8.2f}ms  speedup(numba)={t_py/t_nb:.1f}x  speedup(cuda)={t_py/t_cu:.1f}x')

    # --- Singleresidue ---
    t0 = time.perf_counter()
    for _ in range(N):
        model_py._decoy_fluctuation = {}
        model_py.frustration(kind='singleresidue')
    t_py = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        model_numba.frustration(kind='singleresidue')
    t_nb = (time.perf_counter() - t0) / N

    if cuda_ok:
        t0 = time.perf_counter()
        for _ in range(N):
            model_cuda.frustration(kind='singleresidue')
        t_cu = (time.perf_counter() - t0) / N
    else:
        t_cu = float('nan')

    print(f'singleresidue       python={t_py*1000:8.2f}ms  numba={t_nb*1000:8.2f}ms  cuda={t_cu*1000:8.2f}ms  speedup(numba)={t_py/t_nb:.1f}x  speedup(cuda)={t_py/t_cu:.1f}x')

    # --- Mutational ---
    t0 = time.perf_counter()
    for _ in range(N):
        model_py._decoy_fluctuation = {}
        model_py.frustration(kind='mutational')
    t_py = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        model_numba.frustration(kind='mutational')
    t_nb = (time.perf_counter() - t0) / N

    if cuda_ok:
        t0 = time.perf_counter()
        for _ in range(N):
            model_cuda.frustration(kind='mutational')
        t_cu = (time.perf_counter() - t0) / N
    else:
        t_cu = float('nan')

    print(f'mutational          python={t_py*1000:8.2f}ms  numba={t_nb*1000:8.2f}ms  cuda={t_cu*1000:8.2f}ms  speedup(numba)={t_py/t_nb:.1f}x  speedup(cuda)={t_py/t_cu:.1f}x')

    # --- Configurational (fewer iters, it's slow) ---
    Nc = 1 if L > 500 else 3
    n_decoys = 4000

    if L > 50000:
        print("Skipping configurational.python/numba for large protein because it gets extremely slow...")
        t_py = float('inf')
        t_nb = float('inf')
    else:
        t0 = time.perf_counter()
        for _ in range(Nc):
            model_py.configurational_frustration(n_decoys=n_decoys)
        t_py = (time.perf_counter() - t0) / Nc

        t0 = time.perf_counter()
        for _ in range(Nc):
            model_numba.configurational_frustration(n_decoys=n_decoys)
        t_nb = (time.perf_counter() - t0) / Nc

    if cuda_ok:
        t0 = time.perf_counter()
        for _ in range(Nc):
            model_cuda.configurational_frustration(n_decoys=n_decoys)
        t_cu = (time.perf_counter() - t0) / Nc
    else:
        t_cu = float('nan')

    print(f'configurational     python={t_py*1000:8.2f}ms  numba={t_nb*1000:8.2f}ms  cuda={t_cu*1000:8.2f}ms  speedup(numba)={t_py/t_nb:.1f}x  speedup(cuda)={t_py/t_cu:.1f}x')
    print("\n")

