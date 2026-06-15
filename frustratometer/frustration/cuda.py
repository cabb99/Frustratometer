"""Refactored CUDA backend for frustration analysis.

Design goals
------------
- One explicit public object: ``FrustrationCUDA``.
- No hidden host->device conversion inside hot paths.
- Clear separation between:
  1. device math helpers,
  2. CUDA kernels,
  3. backend state / workspace,
  4. CPU-side output formatting.
- Dense output is an option on the public method, not a separate wrapper.

Expected input
--------------
The constructor expects a host-side ``FrustrationData``-like object with the
same attributes used by the original implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

ALPHABET = 21
THREADS_REDUCE = 256
THREADS_NATIVE = 256
THREADS_SINGLERESIDUE = 64
THREADS_MUTATIONAL = 64
THREADS_CONFIG = 256

# -----------------------------------------------------------------------------
# Device math helpers
# -----------------------------------------------------------------------------

@cuda.jit(device=True)
def _j_val_block(a, b, coeff, c, gammas, k_contact):
    """Generic contact coupling k·Σ_t coeff[c, t]·gammas[t, a, b] over T channels.

    ``coeff`` is (Nc, T) and ``gammas`` is (T, 21, 21); channel count is read from
    ``coeff`` so physics features add channels without touching this accessor."""
    s = 0.0
    for t in range(coeff.shape[1]):
        s += gammas[t, a, b] * coeff[c, t]
    return k_contact * s


@cuda.jit(device=True)
def _config_pair_energy(
    q1,
    q2,
    n1,
    n2,
    c,
    h_struct,
    gammas,
    conf_theta,
    conf_thetaII,
    rho_r,
    eta_sigma,
    rho_0,
    charges,
    conf_elec_ind,
    k_contact,
    k_electrostatics,
):
    """Configurational contact energy from the same burial field ``h_struct`` as the frozen
    kernels (read at the decoy positions/identities); contact uses the three water channels
    with sigma resampled from rho. Membrane configurational routes through numpy."""
    burial = -(h_struct[n1, q1] + h_struct[n2, q2])

    switch = (
        0.25
        * (1.0 - math.tanh(eta_sigma * (rho_r[n1] - rho_0)))
        * (1.0 - math.tanh(eta_sigma * (rho_r[n2] - rho_0)))
    )

    contact = -k_contact * (
        gammas[0, q1, q2] * conf_theta[c]
        + gammas[1, q1, q2] * switch * conf_thetaII[c]
        + gammas[2, q1, q2] * (1.0 - switch) * conf_thetaII[c]
    )

    electrostatic = k_electrostatics * conf_elec_ind[c] * charges[q1] * charges[q2]
    return burial + contact + electrostatic


# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------

@cuda.jit
def _clear_kernel(arr):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] = 0.0


@cuda.jit
def _build_v_kernel(seq_index, contact_i, contact_j, coeff, gammas, k_contact, V):
    """Build V[j, a] = sum_c J(seq[contact_i[c]], a, ...) over contacts hitting j."""
    c = cuda.grid(1)
    if c >= contact_i.shape[0]:
        return

    seq_a = seq_index[contact_i[c]]
    j = contact_j[c]

    for a in range(ALPHABET):
        cuda.atomic.add(V, (j, a), _j_val_block(seq_a, a, coeff, c, gammas, k_contact))


@cuda.jit
def _native_energy_kernel(
    seq_index,
    contact_i,
    contact_j,
    coeff,
    h_struct,
    gammas,
    charges,
    elec_phi,
    k_contact,
    out,
):
    """Block-reduce burial/contact/electrostatic energies into out[0:3]."""
    s_burial = cuda.shared.array(THREADS_NATIVE, dtype=np.float64)
    s_contact = cuda.shared.array(THREADS_NATIVE, dtype=np.float64)
    s_elec = cuda.shared.array(THREADS_NATIVE, dtype=np.float64)

    tid = cuda.threadIdx.x
    idx = cuda.grid(1)

    local_burial = 0.0
    local_contact = 0.0
    local_elec = 0.0

    if idx < seq_index.shape[0]:
        seq_i = seq_index[idx]
        local_burial = -h_struct[idx, seq_i]
        local_elec = -0.5 * charges[seq_i] * elec_phi[idx]

    if idx < contact_i.shape[0]:
        i = contact_i[idx]
        j = contact_j[idx]
        seq_i = seq_index[i]
        seq_j = seq_index[j]
        local_contact = -0.5 * _j_val_block(seq_i, seq_j, coeff, idx, gammas, k_contact)

    s_burial[tid] = local_burial
    s_contact[tid] = local_contact
    s_elec[tid] = local_elec
    cuda.syncthreads()

    stride = THREADS_NATIVE // 2
    while stride > 0:
        if tid < stride:
            s_burial[tid] += s_burial[tid + stride]
            s_contact[tid] += s_contact[tid + stride]
            s_elec[tid] += s_elec[tid + stride]
        cuda.syncthreads()
        stride >>= 1

    if tid == 0:
        cuda.atomic.add(out, 0, s_burial[0])
        cuda.atomic.add(out, 1, s_contact[0])
        cuda.atomic.add(out, 2, s_elec[0])


@cuda.jit
def _singleresidue_kernel(
    seq_index,
    h_struct,
    gammas,
    aa_freq,
    charges,
    elec_phi,
    k_contact,
    correction,
    V,
    result,
):
    i = cuda.grid(1)
    if i >= seq_index.shape[0]:
        return

    native_aa = seq_index[i]
    v_native = V[i, native_aa]
    phi_i = elec_phi[i]
    q_native = charges[native_aa]
    h_native = h_struct[i, native_aa]

    sum_w = 0.0
    sum_we = 0.0
    sum_we2 = 0.0

    for a in range(ALPHABET):
        delta_burial = h_native - h_struct[i, a]
        delta_contact = v_native - V[i, a]
        delta_elec = -(charges[a] - q_native) * phi_i
        delta_e = delta_burial + delta_contact + delta_elec
        weight = aa_freq[a]
        sum_w += weight
        sum_we += weight * delta_e
        sum_we2 += weight * delta_e * delta_e

    mean = sum_we / sum_w
    var = sum_we2 / sum_w - mean * mean
    if var < 0.0:
        var = 0.0

    std = math.sqrt(var)
    denom = std + correction
    result[i] = mean / denom if denom > 0.0 else 0.0


@cuda.jit
def _mutational_kernel(
    seq_index,
    contact_i,
    contact_j,
    coeff,
    h_struct,
    gammas,
    contact_freq,
    charges,
    elec_phi,
    elec_ind_contacts,
    k_contact,
    correction,
    V,
    result,
):
    c = cuda.grid(1)
    if c >= contact_i.shape[0]:
        return

    p1 = contact_i[c]
    p2 = contact_j[c]
    s1 = seq_index[p1]
    s2 = seq_index[p2]

    elec_indicator = elec_ind_contacts[c]

    phi1 = elec_phi[p1]
    phi2 = elec_phi[p2]
    q1_native = charges[s1]
    q2_native = charges[s2]

    g_native = _j_val_block(s1, s2, coeff, c, gammas, k_contact)
    const_term = V[p1, s1] + V[p2, s2] - g_native
    h1_native = h_struct[p1, s1]
    h2_native = h_struct[p2, s2]

    term_a = cuda.local.array(ALPHABET, dtype=np.float64)  # type: ignore[arg-type]
    dq_a = cuda.local.array(ALPHABET, dtype=np.float64)  # type: ignore[arg-type]
    term_b = cuda.local.array(ALPHABET, dtype=np.float64)  # type: ignore[arg-type]
    dq_b = cuda.local.array(ALPHABET, dtype=np.float64)  # type: ignore[arg-type]

    for a in range(ALPHABET):
        delta_burial = h1_native - h_struct[p1, a]
        g_a2 = _j_val_block(a, s2, coeff, c, gammas, k_contact)
        dq_a[a] = charges[a] - q1_native
        term_a[a] = delta_burial - V[p1, a] + g_a2 - dq_a[a] * phi1

    for b in range(ALPHABET):
        delta_burial = h2_native - h_struct[p2, b]
        g_1b = _j_val_block(s1, b, coeff, c, gammas, k_contact)
        dq_b[b] = charges[b] - q2_native
        term_b[b] = delta_burial - V[p2, b] + g_1b - dq_b[b] * phi2

    sum_w = 0.0
    sum_we = 0.0
    sum_we2 = 0.0

    for a in range(ALPHABET):
        for b in range(ALPHABET):
            g_ab = _j_val_block(a, b, coeff, c, gammas, k_contact)
            delta_e = (
                const_term
                + term_a[a]
                + term_b[b]
                - g_ab
                - elec_indicator * dq_a[a] * dq_b[b]
            )
            weight = contact_freq[a, b]
            sum_w += weight
            sum_we += weight * delta_e
            sum_we2 += weight * delta_e * delta_e

    mean = sum_we / sum_w
    var = sum_we2 / sum_w - mean * mean
    if var < 0.0:
        var = 0.0

    std = math.sqrt(var)
    denom = std + correction
    result[c] = mean / denom if denom > 0.0 else 0.0


@cuda.jit
def _config_native_kernel(
    seq_index,
    conf_contact_i,
    conf_contact_j,
    h_struct,
    gammas,
    conf_theta,
    conf_thetaII,
    rho_r,
    eta_sigma,
    rho_0,
    charges,
    conf_elec_ind,
    k_contact,
    k_electrostatics,
    result,
):
    c = cuda.grid(1)
    if c >= conf_contact_i.shape[0]:
        return

    n1 = conf_contact_i[c]
    n2 = conf_contact_j[c]
    q1 = seq_index[n1]
    q2 = seq_index[n2]

    result[c] = _config_pair_energy(
        q1,
        q2,
        n1,
        n2,
        c,
        h_struct,
        gammas,
        conf_theta,
        conf_thetaII,
        rho_r,
        eta_sigma,
        rho_0,
        charges,
        conf_elec_ind,
        k_contact,
        k_electrostatics,
    )


@cuda.jit
def _config_decoy_kernel(
    seq_index,
    conf_theta,
    conf_thetaII,
    conf_elec_ind,
    h_struct,
    gammas,
    rho_r,
    eta_sigma,
    rho_0,
    charges,
    k_contact,
    k_electrostatics,
    rng_states,
    result,
):
    i = cuda.grid(1)
    if i >= result.shape[0]:
        return

    n_conf = conf_theta.shape[0]
    n_res = seq_index.shape[0]

    c = int(xoroshiro128p_uniform_float64(rng_states, i) * n_conf) % n_conf
    n1 = int(xoroshiro128p_uniform_float64(rng_states, i) * n_res) % n_res
    n2 = int(xoroshiro128p_uniform_float64(rng_states, i) * n_res) % n_res
    qi1 = int(xoroshiro128p_uniform_float64(rng_states, i) * n_res) % n_res
    qi2 = int(xoroshiro128p_uniform_float64(rng_states, i) * n_res) % n_res

    q1 = seq_index[qi1]
    q2 = seq_index[qi2]

    result[i] = _config_pair_energy(
        q1,
        q2,
        n1,
        n2,
        c,
        h_struct,
        gammas,
        conf_theta,
        conf_thetaII,
        rho_r,
        eta_sigma,
        rho_0,
        charges,
        conf_elec_ind,
        k_contact,
        k_electrostatics,
    )


@cuda.jit
def _reduce_sum_sq_kernel(arr, out_sum_sq):
    """Block-reduce arr into out_sum_sq[0]=sum, out_sum_sq[1]=sum-of-squares."""
    s_sum = cuda.shared.array(THREADS_REDUCE, dtype=np.float64)
    s_sq = cuda.shared.array(THREADS_REDUCE, dtype=np.float64)

    tid = cuda.threadIdx.x
    idx = cuda.grid(1)

    value = np.float64(arr[idx]) if idx < arr.shape[0] else np.float64(0.0)
    s_sum[tid] = value
    s_sq[tid] = value * value
    cuda.syncthreads()

    stride = THREADS_REDUCE // 2
    while stride > 0:
        if tid < stride:
            s_sum[tid] += s_sum[tid + stride]
            s_sq[tid] += s_sq[tid + stride]
        cuda.syncthreads()
        stride >>= 1

    if tid == 0:
        cuda.atomic.add(out_sum_sq, 0, s_sum[0])
        cuda.atomic.add(out_sum_sq, 1, s_sq[0])


@cuda.jit
def _config_score_kernel(native_energy, decoy_stats, n_decoys, correction, result):
    c = cuda.grid(1)
    if c >= native_energy.shape[0]:
        return

    mean_decoy = decoy_stats[0] / n_decoys
    var_decoy = decoy_stats[1] / n_decoys - mean_decoy * mean_decoy
    if var_decoy < 0.0:
        var_decoy = 0.0

    std_decoy = math.sqrt(var_decoy)
    denom = std_decoy + correction
    result[c] = -(native_energy[c] - mean_decoy) / denom if denom > 0.0 else 0.0


# -----------------------------------------------------------------------------
# Public backend
# -----------------------------------------------------------------------------

class FrustrationCUDA:
    """Explicit GPU backend for frustration calculations.

    Public methods
    --------------
    - native_energy()
    - singleresidue(correction=0.0)
    - mutational(correction=0.0, dense=False)
    - configurational(n_decoys=None, seed=None, correction=0.0, dense=True)
    """

    def __init__(self, data, default_n_decoys: int = 4000, seed: int = 42):
        self.L = data.L
        self.Nc = data.Nc
        self.n_conf = data.n_conf
        self.k_contact = data.k_contact
        self.eta_sigma = data.eta_sigma
        self.rho_0 = data.rho_0
        self.k_electrostatics = data.k_electrostatics

        # Host-side metadata only for output formatting.
        self._contact_i_host = data.contact_i
        self._contact_j_host = data.contact_j
        self._conf_contact_i_host = data.conf_contact_i
        self._conf_contact_j_host = data.conf_contact_j

        # Compute minimum grid size to avoid GPU under-utilization warnings.
        # Use 4× the number of SMs so every SM gets at least 4 blocks.
        device = cuda.get_current_device()
        self._min_blocks = 4 * device.MULTIPROCESSOR_COUNT

        self._upload_inputs(data)
        self._allocate_workspace(default_n_decoys)
        self._rng_seed = seed
        self._rng_states = create_xoroshiro128p_states(default_n_decoys, seed=seed)
        self._build_v(data)

    def _blocks(self, n: int, threads: int) -> int:
        return max(self._min_blocks, (n + threads - 1) // threads)

    def _upload_inputs(self, data) -> None:
        # Core native-state inputs.
        self.seq_index = cuda.to_device(data.seq_index)
        self.contact_i = cuda.to_device(data.contact_i)
        self.contact_j = cuda.to_device(data.contact_j)
        self.coeff = cuda.to_device(data.coeff)
        self.h_struct = cuda.to_device(data.h_struct)
        self.rho_r = cuda.to_device(data.rho_r)
        self.gammas = cuda.to_device(data.gammas)
        self.aa_freq = cuda.to_device(data.aa_freq)
        self.contact_freq = cuda.to_device(data.contact_freq)
        self.charges = cuda.to_device(data.charges)
        self.elec_phi = cuda.to_device(data.elec_phi)
        self.elec_ind_contacts = cuda.to_device(data.elec_ind_contacts)

        # Configurational inputs.
        self.conf_contact_i = cuda.to_device(data.conf_contact_i)
        self.conf_contact_j = cuda.to_device(data.conf_contact_j)
        self.conf_theta = cuda.to_device(data.conf_theta)
        self.conf_thetaII = cuda.to_device(data.conf_thetaII)
        self.conf_elec_ind = cuda.to_device(data.conf_elec_ind)

    def _allocate_workspace(self, default_n_decoys: int) -> None:
        self.d_V = cuda.device_array((self.L, ALPHABET), dtype=np.float64)
        self.d_native_parts = cuda.device_array(3, dtype=np.float64)
        self.d_singleresidue = cuda.device_array(self.L, dtype=np.float64)
        self.d_mutational = cuda.device_array(self.Nc, dtype=np.float64)
        self.d_config_native = cuda.device_array(self.n_conf, dtype=np.float64)
        self.d_config_result = cuda.device_array(self.n_conf, dtype=np.float64)
        self.d_config_stats = cuda.device_array(2, dtype=np.float64)
        self.d_config_decoy = cuda.device_array(default_n_decoys, dtype=np.float64)

    def _build_v(self, data) -> None:
        _clear_kernel[self._blocks(self.d_V.size, THREADS_REDUCE), THREADS_REDUCE](self.d_V.reshape(self.d_V.size))
        if self.Nc == 0:
            return
        _build_v_kernel[self._blocks(self.Nc, THREADS_REDUCE), THREADS_REDUCE](
            self.seq_index,
            self.contact_i,
            self.contact_j,
            self.coeff,
            self.gammas,
            self.k_contact,
            self.d_V,
        )

    def _ensure_decoys(self, n_decoys: int, seed: int) -> None:
        if self.d_config_decoy.shape[0] != n_decoys:
            self.d_config_decoy = cuda.device_array(n_decoys, dtype=np.float64)
        # Re-seed on every call so repeated configurational() calls with the same seed are
        # reproducible; the decoy kernel advances the states in place, and the numba backend
        # likewise re-seeds per call via np.random.seed.
        self._rng_states = create_xoroshiro128p_states(n_decoys, seed=seed)
        self._rng_seed = seed

    @staticmethod
    def _to_dense_contact_map(values, row_index, col_index, L: int, *, symmetric: bool, fill_value: float):
        dense = np.full((L, L), fill_value, dtype=np.float64)
        dense[row_index, col_index] = values
        if symmetric:
            dense[col_index, row_index] = values
        return dense

    def native_energy(self) -> float:
        blocks = self._blocks(max(self.L, self.Nc), THREADS_NATIVE)
        _clear_kernel[self._blocks(3, THREADS_REDUCE), THREADS_REDUCE](self.d_native_parts)
        _native_energy_kernel[blocks, THREADS_NATIVE](
            self.seq_index,
            self.contact_i,
            self.contact_j,
            self.coeff,
            self.h_struct,
            self.gammas,
            self.charges,
            self.elec_phi,
            self.k_contact,
            self.d_native_parts,
        )
        parts = self.d_native_parts.copy_to_host()
        return float(parts[0] + parts[1] + parts[2])

    def singleresidue(self, correction: float = 0.0) -> np.ndarray:
        blocks = self._blocks(self.L, THREADS_SINGLERESIDUE)
        _singleresidue_kernel[blocks, THREADS_SINGLERESIDUE](
            self.seq_index,
            self.h_struct,
            self.gammas,
            self.aa_freq,
            self.charges,
            self.elec_phi,
            self.k_contact,
            float(correction),
            self.d_V,
            self.d_singleresidue,
        )
        return self.d_singleresidue.copy_to_host()

    def mutational(self, correction: float = 0.0, dense: bool = False):
        blocks = self._blocks(self.Nc, THREADS_MUTATIONAL)
        _mutational_kernel[blocks, THREADS_MUTATIONAL](
            self.seq_index,
            self.contact_i,
            self.contact_j,
            self.coeff,
            self.h_struct,
            self.gammas,
            self.contact_freq,
            self.charges,
            self.elec_phi,
            self.elec_ind_contacts,
            self.k_contact,
            float(correction),
            self.d_V,
            self.d_mutational,
        )
        values = self.d_mutational.copy_to_host()
        if dense:
            return self._to_dense_contact_map(
                values,
                self._contact_i_host,
                self._contact_j_host,
                self.L,
                symmetric=False,
                fill_value=0.0,
            )
        return values

    def configurational(
        self,
        n_decoys: Optional[int] = None,
        seed: Optional[int] = None,
        correction: float = 0.0,
        dense: bool = True,
    ):
        if n_decoys is None:
            n_decoys = self.d_config_decoy.shape[0]
        if seed is None:
            seed = self._rng_seed

        self._ensure_decoys(n_decoys, seed)

        native_blocks = self._blocks(self.n_conf, THREADS_CONFIG)
        decoy_blocks = self._blocks(n_decoys, THREADS_REDUCE)

        _config_native_kernel[native_blocks, THREADS_CONFIG](
            self.seq_index,
            self.conf_contact_i,
            self.conf_contact_j,
            self.h_struct,
            self.gammas,
            self.conf_theta,
            self.conf_thetaII,
            self.rho_r,
            self.eta_sigma,
            self.rho_0,
            self.charges,
            self.conf_elec_ind,
            self.k_contact,
            self.k_electrostatics,
            self.d_config_native,
        )

        _config_decoy_kernel[decoy_blocks, THREADS_REDUCE](
            self.seq_index,
            self.conf_theta,
            self.conf_thetaII,
            self.conf_elec_ind,
            self.h_struct,
            self.gammas,
            self.rho_r,
            self.eta_sigma,
            self.rho_0,
            self.charges,
            self.k_contact,
            self.k_electrostatics,
            self._rng_states,
            self.d_config_decoy,
        )

        _clear_kernel[self._blocks(2, THREADS_REDUCE), THREADS_REDUCE](self.d_config_stats)
        _reduce_sum_sq_kernel[decoy_blocks, THREADS_REDUCE](self.d_config_decoy, self.d_config_stats)

        _config_score_kernel[native_blocks, THREADS_CONFIG](
            self.d_config_native,
            self.d_config_stats,
            n_decoys,
            float(correction),
            self.d_config_result,
        )

        values = self.d_config_result.copy_to_host()
        if dense:
            return self._to_dense_contact_map(
                values,
                self._conf_contact_i_host,
                self._conf_contact_j_host,
                self.L,
                symmetric=True,
                fill_value=np.nan,
            )
        return values


# -----------------------------------------------------------------------------
# Explicit-J kernels (generic Potts: DCA, static-context reduced model)
# -----------------------------------------------------------------------------

@cuda.jit
def _build_v_explicit_kernel(seq_index, contact_i, contact_j, J, V):
    c = cuda.grid(1)
    if c >= contact_i.shape[0]:
        return
    si = seq_index[contact_i[c]]
    j = contact_j[c]
    for a in range(ALPHABET):
        cuda.atomic.add(V, (j, a), J[c, si, a])


@cuda.jit
def _singleresidue_explicit_kernel(seq_index, h, V, aa_freq, correction, result):
    i = cuda.grid(1)
    if i >= seq_index.shape[0]:
        return
    si = seq_index[i]
    v_nat = V[i, si]
    h_nat = h[i, si]
    sum_w = 0.0; sum_we = 0.0; sum_we2 = 0.0
    for a in range(ALPHABET):
        de = (h_nat - h[i, a]) + (v_nat - V[i, a])
        w = aa_freq[a]
        sum_w += w; sum_we += w * de; sum_we2 += w * de * de
    mean = sum_we / sum_w
    var = sum_we2 / sum_w - mean * mean
    if var < 0.0:
        var = 0.0
    denom = math.sqrt(var) + correction
    result[i] = mean / denom if denom > 0.0 else 0.0


@cuda.jit
def _mutational_explicit_kernel(seq_index, contact_i, contact_j, h, J, V,
                                contact_freq, correction, result):
    c = cuda.grid(1)
    if c >= contact_i.shape[0]:
        return
    p1 = contact_i[c]; p2 = contact_j[c]
    s1 = seq_index[p1]; s2 = seq_index[p2]
    const_term = V[p1, s1] + V[p2, s2] - J[c, s1, s2]
    h1 = h[p1, s1]; h2 = h[p2, s2]
    term_a = cuda.local.array(ALPHABET, dtype=np.float64)  # type: ignore[arg-type]
    term_b = cuda.local.array(ALPHABET, dtype=np.float64)  # type: ignore[arg-type]
    for a in range(ALPHABET):
        term_a[a] = (h1 - h[p1, a]) - V[p1, a] + J[c, a, s2]
    for b in range(ALPHABET):
        term_b[b] = (h2 - h[p2, b]) - V[p2, b] + J[c, s1, b]
    sum_w = 0.0; sum_we = 0.0; sum_we2 = 0.0
    for a in range(ALPHABET):
        for b in range(ALPHABET):
            de = const_term + term_a[a] + term_b[b] - J[c, a, b]
            w = contact_freq[a, b]
            sum_w += w; sum_we += w * de; sum_we2 += w * de * de
    mean = sum_we / sum_w
    var = sum_we2 / sum_w - mean * mean
    if var < 0.0:
        var = 0.0
    denom = math.sqrt(var) + correction
    result[c] = mean / denom if denom > 0.0 else 0.0


def _explicit_device(potts, seq_index):
    h = cuda.to_device(np.ascontiguousarray(potts['h'], dtype=np.float64))
    J = cuda.to_device(np.ascontiguousarray(potts['J'], dtype=np.float64))
    ci = cuda.to_device(np.ascontiguousarray(potts['contact_i'], dtype=np.intp))
    cj = cuda.to_device(np.ascontiguousarray(potts['contact_j'], dtype=np.intp))
    si = cuda.to_device(np.ascontiguousarray(seq_index, dtype=np.int32))
    L = int(potts.get('L', potts['h'].shape[0]))
    Nc = len(potts['contact_i'])
    V = cuda.device_array((L, ALPHABET), dtype=np.float64)
    _clear_kernel[(V.size + THREADS_REDUCE - 1) // THREADS_REDUCE, THREADS_REDUCE](V.reshape(V.size))
    if Nc:
        _build_v_explicit_kernel[(Nc + THREADS_REDUCE - 1) // THREADS_REDUCE, THREADS_REDUCE](si, ci, cj, J, V)
    return si, ci, cj, h, J, V, L, Nc


def singleresidue_frustration_potts(seq_index, potts, aa_freq, correction=0.0):
    si, ci, cj, h, J, V, L, Nc = _explicit_device(potts, seq_index)
    d_aa = cuda.to_device(np.ascontiguousarray(aa_freq, dtype=np.float64))
    result = cuda.device_array(L, dtype=np.float64)
    _singleresidue_explicit_kernel[(L + THREADS_SINGLERESIDUE - 1) // THREADS_SINGLERESIDUE,
                                   THREADS_SINGLERESIDUE](si, h, V, d_aa, float(correction), result)
    return result.copy_to_host()


def mutational_frustration_potts(seq_index, potts, contact_freq, correction=0.0):
    si, ci, cj, h, J, V, L, Nc = _explicit_device(potts, seq_index)
    d_cf = cuda.to_device(np.ascontiguousarray(contact_freq, dtype=np.float64))
    result = cuda.device_array(max(Nc, 1), dtype=np.float64)
    if Nc:
        _mutational_explicit_kernel[(Nc + THREADS_MUTATIONAL - 1) // THREADS_MUTATIONAL,
                                    THREADS_MUTATIONAL](si, ci, cj, h, J, V, d_cf, float(correction), result)
    return result.copy_to_host()[:Nc]


# -----------------------------------------------------------------------------
# Module-level API (mirrors frustration.numba)
# -----------------------------------------------------------------------------

def _as_backend(data):
    """Accept a host FrustrationData or an already-uploaded FrustrationCUDA."""
    return data if isinstance(data, FrustrationCUDA) else FrustrationCUDA(data)


def native_energy(data):
    return _as_backend(data).native_energy()


def singleresidue_frustration(data, correction=0.0):
    return _as_backend(data).singleresidue(correction=float(correction))


def mutational_frustration(data, correction=0.0):
    """Sparse per-contact mutational frustration (Nc,); see frustration.numba."""
    return _as_backend(data).mutational(correction=float(correction), dense=False)


def mutational_frustration_dense(data, correction=0.0):
    return _as_backend(data).mutational(correction=float(correction), dense=True)


def configurational_frustration(data, n_decoys=4000, seed=42, correction=0.0):
    return _as_backend(data).configurational(
        n_decoys=n_decoys, seed=seed, correction=float(correction), dense=True)
