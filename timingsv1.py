"""Low-level CUDA profiler for the frustration backend.

Complements quick_benchmark.py: instead of comparing backends end-to-end, it
times the FrustrationCUDA methods and the raw _mutational_kernel launch in
isolation, which is useful when tuning the kernels themselves.
"""
import time
from pathlib import Path

from numba import cuda

import frustratometer
from frustratometer.frustration import cuda as fcuda
from frustratometer.frustration.data import FrustrationData
from frustratometer.frustration.cuda import THREADS_MUTATIONAL

test_data_path = Path('tests/data')


def time_cuda_call(label, fn, repeat=10):
    """Time a host call that launches GPU work, synchronizing on every call."""
    fn()  # warmup
    cuda.synchronize()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        cuda.synchronize()  # kernels are async; sync before measuring
        times.append(time.perf_counter() - t0)

    print(f"{label:25s} mean={sum(times) / len(times):.6f}s  min={min(times):.6f}s")


def time_kernel(label, launch):
    """Time a single kernel launch using CUDA events."""
    start = cuda.event()
    end = cuda.event()

    start.record()
    launch()
    end.record()
    end.synchronize()

    ms = cuda.event_elapsed_time(start, end)
    print(f"{label:30s}: {ms:.3f} ms")
    return ms


# 6u5e sparse AWSEM with electrostatics enabled.
structure = frustratometer.Structure(test_data_path / '6u5e.pdb', "A", sparse=True)
model = frustratometer.AWSEM(
    structure,
    distance_cutoff_contact=9.5,
    min_sequence_separation_contact=2,
    k_electrostatics=4.184,
    min_sequence_separation_electrostatics=1,
)
data = FrustrationData.from_awsem(model)
backend = fcuda.FrustrationCUDA(data)

# --- High-level method timings ---
time_cuda_call("native_energy", backend.native_energy)
time_cuda_call("singleresidue", backend.singleresidue)
time_cuda_call("mutational", backend.mutational)
time_cuda_call("configurational", backend.configurational)

# --- Raw mutational kernel + device->host copy ---
correction = 0.0
blocks = max(1, (backend.Nc + THREADS_MUTATIONAL - 1) // THREADS_MUTATIONAL)

time_kernel("mutational_kernel", lambda: fcuda._mutational_kernel[blocks, THREADS_MUTATIONAL](
    backend.seq_index,
    backend.contact_i,
    backend.contact_j,
    backend.theta,
    backend.tsw,
    backend.tsp,
    backend.burial_indicator,
    backend.gammas,
    backend.bg,
    backend.contact_freq,
    backend.charges,
    backend.elec_phi,
    backend.elec_ind_contacts,
    backend.k_contact,
    correction,
    backend.d_V,
    backend.d_mutational,
))

time_kernel("copy mutational to host", lambda: backend.d_mutational.copy_to_host())
