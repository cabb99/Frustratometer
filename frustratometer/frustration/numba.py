"""
Numba-accelerated frustration kernels.

All functions accept a ``FrustrationData`` instance (from ``data.py``)
and use the unified DCA 21-letter alphabet throughout.
"""

import numpy as np
from numba import njit, prange


# ─── Inline helpers ───────────────────────────────────────────────────────────

@njit(inline='always')
def _J_val(a, b, theta_c, tsw_c, tsp_c, gammas, k):
    """Contact coupling k·(direct·θ + water·tsw + protein·tsp)."""
    return k * (gammas[0, a, b] * theta_c + gammas[1, a, b] * tsw_c + gammas[2, a, b] * tsp_c)


@njit(inline='always')
def _J_val_block(a, b, coeff_c, gammas, k):
    """Generic contact coupling k·Σ_t coeff_c[t]·gammas[t, a, b] over T channels.

    ``coeff_c`` is the per-contact channel vector (T,); ``gammas`` is (T, 21, 21).
    Channel count is read from ``coeff_c`` so physics features add channels without
    touching this accessor."""
    s = 0.0
    for t in range(coeff_c.shape[0]):
        s += gammas[t, a, b] * coeff_c[t]
    return k * s


@njit(inline='always')
def _burial_h(a, i, bg, bi, k):
    """Burial field for amino acid *a* at position *i* (positive; energy = −h)."""
    e = 0.0
    for w in range(3):
        e += bg[a, w] * bi[i, w]
    return 0.5 * k * e


@njit(inline='always')
def _frustration_index(sum_w, sum_we, sum_we2, correction):
    """Frustration index from weighted sums."""
    mean = sum_we / sum_w
    var = sum_we2 / sum_w - mean * mean
    if var < 0.0:
        var = 0.0
    std = var ** 0.5
    denom = std + correction
    return mean / denom if denom > 0.0 else 0.0


# ─── V matrix ────────────────────────────────────────────────────────────────

@njit(cache=True)
def _compute_V(seq_index, contact_i, contact_j, Nc, L,
               coeff, gammas, k_contact):
    """V[j, a] = Σ_{c: cj=j} J(seq[ci], a, c)."""
    V = np.zeros((L, 21))
    for c in range(Nc):
        si = seq_index[contact_i[c]]
        j = contact_j[c]
        cc = coeff[c]
        for a in range(21):
            V[j, a] += _J_val_block(si, a, cc, gammas, k_contact)
    return V


# ─── 1. Native energy ────────────────────────────────────────────────────────

@njit(cache=True)
def _native_energy(seq_index, contact_i, contact_j, Nc, L,
                   coeff, h_struct, gammas, k_contact,
                   charges, elec_phi):
    e_burial = 0.0
    for i in range(L):
        e_burial -= h_struct[i, seq_index[i]]

    e_contact = 0.0
    for c in range(Nc):
        e_contact -= _J_val_block(seq_index[contact_i[c]], seq_index[contact_j[c]],
                                  coeff[c], gammas, k_contact)
    e_contact *= 0.5

    e_elec = 0.0
    for i in range(L):
        e_elec += charges[seq_index[i]] * elec_phi[i]
    e_elec *= -0.5

    return e_burial + e_contact + e_elec


# ─── 2. Singleresidue frustration ────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _singleresidue(seq_index, contact_i, contact_j, Nc, L,
                   coeff, h_struct, gammas, k_contact,
                   aa_freq, charges, elec_phi, correction):
    V = _compute_V(seq_index, contact_i, contact_j, Nc, L,
                   coeff, gammas, k_contact)
    result = np.empty(L)

    for i in prange(L):
        si = seq_index[i]
        V_nat = V[i, si]
        phi_i = elec_phi[i]
        qn_i = charges[si]
        h_nat = h_struct[i, si]

        sum_w = 0.0; sum_we = 0.0; sum_we2 = 0.0
        for a in range(21):
            db = h_nat - h_struct[i, a]
            dc = V_nat - V[i, a]
            de_el = -(charges[a] - qn_i) * phi_i
            de = db + dc + de_el
            w = aa_freq[a]
            sum_w += w; sum_we += w * de; sum_we2 += w * de * de

        result[i] = _frustration_index(sum_w, sum_we, sum_we2, correction)
    return result


# ─── 3. Mutational frustration ───────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _mutational(seq_index, contact_i, contact_j, Nc, L,
                coeff, h_struct, gammas, k_contact,
                contact_freq, charges, elec_phi, elec_ind_contacts, correction):
    V = _compute_V(seq_index, contact_i, contact_j, Nc, L,
                   coeff, gammas, k_contact)
    result = np.empty(Nc)

    for c in prange(Nc):
        p1 = contact_i[c]; p2 = contact_j[c]
        s1 = seq_index[p1]; s2 = seq_index[p2]
        cc = coeff[c]
        ind_c = elec_ind_contacts[c]
        phi1 = elec_phi[p1]; phi2 = elec_phi[p2]
        qn1 = charges[s1]; qn2 = charges[s2]

        G_nat = _J_val_block(s1, s2, cc, gammas, k_contact)
        const_c = V[p1, s1] + V[p2, s2] - G_nat

        h_nat_p1 = h_struct[p1, s1]
        h_nat_p2 = h_struct[p2, s2]

        term_a = np.empty(21); dqa = np.empty(21)
        for a in range(21):
            db1 = h_nat_p1 - h_struct[p1, a]
            Ga2 = _J_val_block(a, s2, cc, gammas, k_contact)
            dqa[a] = charges[a] - qn1
            term_a[a] = db1 - V[p1, a] + Ga2 - dqa[a] * phi1

        term_b = np.empty(21); dqb = np.empty(21)
        for b in range(21):
            db2 = h_nat_p2 - h_struct[p2, b]
            G1b = _J_val_block(s1, b, cc, gammas, k_contact)
            dqb[b] = charges[b] - qn2
            term_b[b] = db2 - V[p2, b] + G1b - dqb[b] * phi2

        sum_w = 0.0; sum_we = 0.0; sum_we2 = 0.0
        for a in range(21):
            for b in range(21):
                Gab = _J_val_block(a, b, cc, gammas, k_contact)
                de = const_c + term_a[a] + term_b[b] - Gab - ind_c * dqa[a] * dqb[b]
                w = contact_freq[a, b]
                sum_w += w; sum_we += w * de; sum_we2 += w * de * de

        result[c] = _frustration_index(sum_w, sum_we, sum_we2, correction)
    return result


# ─── 4. Configurational frustration ──────────────────────────────────────────

@njit(inline='always')
def _conf_pair_energy(q1, q2, n1, n2, c, bg, burial_indicator, gammas,
                      conf_theta, conf_thetaII, rho_r, eta_sigma, rho_0,
                      charges, conf_elec_ind, k_contact, k_elec):
    """Configurational contact energy: burial at n1/n2 + contact (sigma resampled from
    rho at n1/n2) at geometry c + electrostatics, for identities q1/q2. Shared by the
    native and decoy loops."""
    eb = -(_burial_h(q1, n1, bg, burial_indicator, k_contact)
         + _burial_h(q2, n2, bg, burial_indicator, k_contact))
    sw = (0.25
          * (1.0 - np.tanh(eta_sigma * (rho_r[n1] - rho_0)))
          * (1.0 - np.tanh(eta_sigma * (rho_r[n2] - rho_0))))
    ec = -_J_val(q1, q2, conf_theta[c],
                 sw * conf_thetaII[c], (1.0 - sw) * conf_thetaII[c],
                 gammas, k_contact)
    ee = k_elec * conf_elec_ind[c] * charges[q1] * charges[q2]
    return eb + ec + ee


@njit(parallel=True, cache=True)
def _configurational(seq_index, L,
                     conf_ci, conf_cj, n_conf,
                     bg, burial_indicator, gammas,
                     conf_theta, conf_thetaII, rho_r,
                     eta_sigma, rho_0,
                     charges, conf_elec_ind,
                     k_contact, k_elec,
                     n_decoys, seed, correction):
    np.random.seed(seed)

    # native energies (parallel)
    native_e = np.empty(n_conf)
    for c in prange(n_conf):
        n1 = conf_ci[c]; n2 = conf_cj[c]
        native_e[c] = _conf_pair_energy(
            seq_index[n1], seq_index[n2], n1, n2, c, bg, burial_indicator, gammas,
            conf_theta, conf_thetaII, rho_r, eta_sigma, rho_0,
            charges, conf_elec_ind, k_contact, k_elec)

    # decoy statistics (serial for reproducible RNG)
    decoy_e = np.empty(n_decoys)
    for i in range(n_decoys):
        c = np.random.randint(0, n_conf)
        n1 = np.random.randint(0, L)
        n2 = np.random.randint(0, L)
        q1 = seq_index[np.random.randint(0, L)]
        q2 = seq_index[np.random.randint(0, L)]
        decoy_e[i] = _conf_pair_energy(
            q1, q2, n1, n2, c, bg, burial_indicator, gammas,
            conf_theta, conf_thetaII, rho_r, eta_sigma, rho_0,
            charges, conf_elec_ind, k_contact, k_elec)

    # mean / std
    mean_d = 0.0
    for i in range(n_decoys):
        mean_d += decoy_e[i]
    mean_d /= n_decoys
    sum_sq = 0.0
    for i in range(n_decoys):
        diff = decoy_e[i] - mean_d
        sum_sq += diff * diff
    std_d = (sum_sq / n_decoys) ** 0.5

    result = np.empty(n_conf)
    denom = std_d + correction
    if denom > 0.0:
        for c in prange(n_conf):
            result[c] = -(native_e[c] - mean_d) / denom
    else:
        for c in range(n_conf):
            result[c] = 0.0
    return result


# ─── Explicit-J kernels (generic Potts: DCA, static-context reduced model) ──────
# These consume a raw sparse Potts model {h (L,21), J (Nc,21,21), contacts} with the
# couplings given explicitly per contact (no AWSEM channels, no electrostatics). They
# power any front-end that already holds an explicit J — DCA and the folded
# static-context model — on the same fast kernels.

@njit(cache=True)
def _compute_V_explicit(seq_index, contact_i, contact_j, Nc, L, J):
    V = np.zeros((L, 21))
    for c in range(Nc):
        si = seq_index[contact_i[c]]
        j = contact_j[c]
        for a in range(21):
            V[j, a] += J[c, si, a]
    return V


@njit(cache=True)
def _native_energy_explicit(seq_index, contact_i, contact_j, Nc, L, h, J):
    e = 0.0
    for i in range(L):
        e -= h[i, seq_index[i]]
    ec = 0.0
    for c in range(Nc):
        ec -= J[c, seq_index[contact_i[c]], seq_index[contact_j[c]]]
    return e + 0.5 * ec


@njit(parallel=True, cache=True)
def _singleresidue_explicit(seq_index, contact_i, contact_j, Nc, L, h, J, aa_freq, correction):
    V = _compute_V_explicit(seq_index, contact_i, contact_j, Nc, L, J)
    result = np.empty(L)
    for i in prange(L):
        si = seq_index[i]
        V_nat = V[i, si]
        h_nat = h[i, si]
        sum_w = 0.0; sum_we = 0.0; sum_we2 = 0.0
        for a in range(21):
            de = (h_nat - h[i, a]) + (V_nat - V[i, a])
            w = aa_freq[a]
            sum_w += w; sum_we += w * de; sum_we2 += w * de * de
        result[i] = _frustration_index(sum_w, sum_we, sum_we2, correction)
    return result


@njit(parallel=True, cache=True)
def _mutational_explicit(seq_index, contact_i, contact_j, Nc, L, h, J, contact_freq, correction):
    V = _compute_V_explicit(seq_index, contact_i, contact_j, Nc, L, J)
    result = np.empty(Nc)
    for c in prange(Nc):
        p1 = contact_i[c]; p2 = contact_j[c]
        s1 = seq_index[p1]; s2 = seq_index[p2]
        G_nat = J[c, s1, s2]
        const_c = V[p1, s1] + V[p2, s2] - G_nat
        h1 = h[p1, s1]; h2 = h[p2, s2]

        term_a = np.empty(21)
        for a in range(21):
            term_a[a] = (h1 - h[p1, a]) - V[p1, a] + J[c, a, s2]
        term_b = np.empty(21)
        for b in range(21):
            term_b[b] = (h2 - h[p2, b]) - V[p2, b] + J[c, s1, b]

        sum_w = 0.0; sum_we = 0.0; sum_we2 = 0.0
        for a in range(21):
            for b in range(21):
                de = const_c + term_a[a] + term_b[b] - J[c, a, b]
                w = contact_freq[a, b]
                sum_w += w; sum_we += w * de; sum_we2 += w * de * de
        result[c] = _frustration_index(sum_w, sum_we, sum_we2, correction)
    return result


def _potts_arrays(potts):
    h = np.ascontiguousarray(potts['h'], dtype=np.float64)
    J = np.ascontiguousarray(potts['J'], dtype=np.float64)
    ci = np.ascontiguousarray(potts['contact_i'], dtype=np.intp)
    cj = np.ascontiguousarray(potts['contact_j'], dtype=np.intp)
    return h, J, ci, cj, len(ci), int(potts.get('L', h.shape[0]))


def native_energy_potts(seq_index, potts):
    """Native energy of an explicit sparse Potts model for the given sequence indices."""
    h, J, ci, cj, Nc, L = _potts_arrays(potts)
    return _native_energy_explicit(np.asarray(seq_index, dtype=np.int32), ci, cj, Nc, L, h, J)


def singleresidue_frustration_potts(seq_index, potts, aa_freq, correction=0.0):
    """Single-residue frustration of an explicit sparse Potts model. Returns (L,)."""
    h, J, ci, cj, Nc, L = _potts_arrays(potts)
    return _singleresidue_explicit(np.asarray(seq_index, dtype=np.int32), ci, cj, Nc, L,
                                   h, J, np.asarray(aa_freq, dtype=np.float64), float(correction))


def mutational_frustration_potts(seq_index, potts, contact_freq, correction=0.0):
    """Mutational frustration per contact (Nc,) of an explicit sparse Potts model."""
    h, J, ci, cj, Nc, L = _potts_arrays(potts)
    return _mutational_explicit(np.asarray(seq_index, dtype=np.int32), ci, cj, Nc, L,
                                h, J, np.asarray(contact_freq, dtype=np.float64), float(correction))


# ─── Public API ───────────────────────────────────────────────────────────────

def compute_V(data):
    """Compute coupling environment matrix V (L, 21)."""
    return _compute_V(data.seq_index, data.contact_i, data.contact_j,
                      data.Nc, data.L, data.coeff,
                      data.gammas, data.k_contact)


def native_energy(data):
    """Compute total native energy.

    Returns
    -------
    energy : float
    """
    return _native_energy(
        data.seq_index, data.contact_i, data.contact_j, data.Nc, data.L,
        data.coeff, data.h_struct, data.gammas, data.k_contact,
        data.charges, data.elec_phi,
    )


def singleresidue_frustration(data, correction=0.0):
    """Compute singleresidue frustration.

    Returns
    -------
    frustration : ndarray (L,)
    """
    return _singleresidue(
        data.seq_index, data.contact_i, data.contact_j, data.Nc, data.L,
        data.coeff, data.h_struct, data.gammas, data.k_contact,
        data.aa_freq, data.charges, data.elec_phi,
        float(correction),
    )


def mutational_frustration(data, correction=0.0):
    """Compute mutational frustration per contact (sparse).

    Returns
    -------
    frustration : ndarray (Nc,)
        Per-contact values; the contacts are ``data.contact_i`` / ``data.contact_j``.
        Preferred over the dense form for large proteins (avoids the (L, L) matrix).
    """
    return _mutational(
        data.seq_index, data.contact_i, data.contact_j, data.Nc, data.L,
        data.coeff, data.h_struct, data.gammas, data.k_contact,
        data.contact_freq, data.charges, data.elec_phi,
        data.elec_ind_contacts,
        float(correction),
    )


def mutational_frustration_dense(data, correction=0.0):
    """Compute mutational frustration as a dense (L, L) matrix.

    Returns
    -------
    dense : ndarray (L, L)
        Zero at non-contact positions.
    """
    frust = mutational_frustration(data, correction)
    dense = np.zeros((data.L, data.L))
    dense[data.contact_i, data.contact_j] = frust
    return dense


def configurational_frustration(data, n_decoys=4000, seed=42, correction=0.0):
    """Compute configurational frustration (MC decoys).

    Returns
    -------
    dense : ndarray (L, L)
        NaN at non-contact positions.
    """
    frust = _configurational(
        data.seq_index, data.L,
        data.conf_contact_i, data.conf_contact_j, data.n_conf,
        data.bg, data.burial_indicator, data.gammas,
        data.conf_theta, data.conf_thetaII, data.rho_r,
        data.eta_sigma, data.rho_0,
        data.charges, data.conf_elec_ind,
        data.k_contact, data.k_electrostatics,
        int(n_decoys), int(seed), float(correction),
    )
    dense = np.full((data.L, data.L), np.nan)
    dense[data.conf_contact_i, data.conf_contact_j] = frust
    dense[data.conf_contact_j, data.conf_contact_i] = frust
    return dense
