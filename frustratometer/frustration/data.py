"""
FrustrationData container and structural preparation functions.

All structural physics (rho, theta, sigma, burial, electrostatics) are
computed from raw sparse distances and AWSEM parameters.  Gamma matrices
are remapped from the AWSEM 20-letter alphabet to the unified DCA 21-letter
alphabet (gap at index 0, zeroed).
"""

from dataclasses import dataclass
import numpy as np

# ─── Alphabet & constants ─────────────────────────────────────────────────────

AA = '-ACDEFGHIKLMNPQRSTVWY'  # DCA 21-letter, gap at index 0

# DCA index → AWSEM index mapping (21 entries, gap → 0='A')
_AA_MAP = np.array(
    [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18],
    dtype=np.int32,
)

# Charges in DCA alphabet order: D(3)=-1, E(4)=-1, K(9)=+1, R(15)=+1
CHARGES = np.zeros(21, dtype=np.float64)
CHARGES[AA.index('D')] = -1.0
CHARGES[AA.index('E')] = -1.0
CHARGES[AA.index('K')] = 1.0
CHARGES[AA.index('R')] = 1.0


# ─── Gamma remapping ─────────────────────────────────────────────────────────

def remap_gamma(gamma_20x20):
    """Remap (20,20) AWSEM gamma to (21,21) DCA alphabet, zeroing the gap row/col."""
    g = gamma_20x20[np.ix_(_AA_MAP, _AA_MAP)].copy()
    g[0, :] = 0.0
    g[:, 0] = 0.0
    return np.ascontiguousarray(g, dtype=np.float64)


def remap_burial_gamma(burial_gamma_20x3):
    """Remap (20,3) AWSEM burial gamma to (21,3) DCA alphabet, zeroing gap."""
    bg = burial_gamma_20x3[_AA_MAP].copy()
    bg[0, :] = 0.0
    return np.ascontiguousarray(bg, dtype=np.float64)


# ─── Structural computations ─────────────────────────────────────────────────

def _seq_sep_ok(ri, rj, min_sep, chain_breaks=None):
    """Boolean mask for pairs satisfying minimum sequence separation."""
    if min_sep is None or min_sep <= 0:
        return np.ones(len(ri), dtype=bool)
    pos_i = ri.astype(np.float64)
    pos_j = rj.astype(np.float64)
    if chain_breaks:
        for brk in chain_breaks:
            pos_i = np.where(ri >= brk, pos_i + min_sep, pos_i)
            pos_j = np.where(rj >= brk, pos_j + min_sep, pos_j)
    return np.abs(pos_i - pos_j) >= min_sep


def compute_rho(dist_row, dist_col, dist_data, L,
                eta, r_min, r_max, min_seq_sep, chain_breaks=None):
    """Residue density (L,) from sparse distances."""
    keep = (dist_data > 0) & _seq_sep_ok(dist_row, dist_col, min_seq_sep, chain_breaks)
    d = dist_data[keep]
    vals = 0.25 * (1 + np.tanh(eta * (d - r_min))) * (1 + np.tanh(eta * (r_max - d)))
    return np.bincount(dist_row[keep], weights=vals, minlength=L).astype(np.float64)


def compute_contacts(dist_row, dist_col, dist_data, rho_r,
                     eta, r_min, r_max, r_minII, r_maxII,
                     eta_sigma, rho_0,
                     min_seq_sep, distance_cutoff, chain_breaks=None):
    """Symmetric contacts with theta, thetaII, tsw, tsp per contact.

    Returns (ci, cj, theta, thetaII, tsw, tsp).
    """
    keep = (
        (dist_data > 0)
        & _seq_sep_ok(dist_row, dist_col, min_seq_sep, chain_breaks)
        & (dist_data <= distance_cutoff)
    )
    ci = dist_row[keep].copy()
    cj = dist_col[keep].copy()
    cd = dist_data[keep]

    theta = (0.25 * (1 + np.tanh(eta * (cd - r_min)))
                   * (1 + np.tanh(eta * (r_max - cd)))).astype(np.float64)
    thetaII = (0.25 * (1 + np.tanh(eta * (cd - r_minII)))
                     * (1 + np.tanh(eta * (r_maxII - cd)))).astype(np.float64)
    sw = (0.25
          * (1 - np.tanh(eta_sigma * (rho_r[ci] - rho_0)))
          * (1 - np.tanh(eta_sigma * (rho_r[cj] - rho_0))))
    tsw = (thetaII * sw).astype(np.float64)
    tsp = (thetaII * (1.0 - sw)).astype(np.float64)

    return ci, cj, theta, thetaII, tsw, tsp


def compute_burial(rho_r, burial_kappa, burial_ro_min, burial_ro_max):
    """Burial indicator (L, 3) from residue density."""
    ro_min = np.asarray(burial_ro_min, dtype=np.float64)
    ro_max = np.asarray(burial_ro_max, dtype=np.float64)
    rho_b = rho_r[:, np.newaxis]
    return (np.tanh(burial_kappa * (rho_b - ro_min))
            + np.tanh(burial_kappa * (ro_max - rho_b))).astype(np.float64)


def compute_electrostatics(elec_row, elec_col, elec_data, L,
                           contact_i, contact_j, seq_index,
                           k_electrostatics, screening_length,
                           min_seq_sep_elec, min_seq_sep_contact=0,
                           chain_breaks=None):
    """Compute elec_phi (L,) and elec_ind_contacts (Nc,)."""
    Nc = len(contact_i)
    elec_phi = np.zeros(L, dtype=np.float64)
    elec_ind_contacts = np.zeros(Nc, dtype=np.float64)

    if k_electrostatics == 0:
        return elec_phi, elec_ind_contacts

    keep = (elec_data > 0) & _seq_sep_ok(elec_row, elec_col, min_seq_sep_elec, chain_breaks)
    e_ri = elec_row[keep]
    e_rj = elec_col[keep]
    e_d = elec_data[keep]

    with np.errstate(divide='ignore', invalid='ignore'):
        ind_vals = -k_electrostatics * np.exp(-e_d / screening_length) / e_d
        ind_vals = np.nan_to_num(ind_vals, nan=0.0, posinf=0.0, neginf=0.0)

    # frustration mask for phi (same logic as original)
    frust_sep = min(min_seq_sep_elec,
                    min_seq_sep_contact if min_seq_sep_contact > 0 else min_seq_sep_elec)
    frust_ok = _seq_sep_ok(e_ri, e_rj, frust_sep, chain_breaks)
    ind_masked = ind_vals * frust_ok.astype(np.float64)

    q_native = CHARGES[seq_index]
    np.add.at(elec_phi, e_ri, ind_masked * q_native[e_rj])

    # indicator at contact positions (sorted-key lookup)
    c_key = contact_i.astype(np.int64) * L + contact_j.astype(np.int64)
    e_key = e_ri.astype(np.int64) * L + e_rj.astype(np.int64)
    sort_e = np.argsort(e_key)
    sorted_e_key = e_key[sort_e]
    pos = np.searchsorted(sorted_e_key, c_key)
    pos = np.clip(pos, 0, max(len(sorted_e_key) - 1, 0))
    if len(sorted_e_key) > 0:
        found = sorted_e_key[pos] == c_key
        elec_ind_contacts = np.where(found, ind_vals[sort_e[pos]], 0.0)

    return elec_phi, elec_ind_contacts


def compute_configurational_contacts(dist_row, dist_col, dist_data,
                                     eta, r_min, r_max, r_minII, r_maxII,
                                     distance_cutoff, screening_length):
    """Upper-triangle contacts for configurational frustration (no seq sep filter).

    Returns (ci, cj, conf_theta, conf_thetaII, conf_elec_ind).
    """
    keep = (dist_row < dist_col) & (dist_data > 0) & (dist_data <= distance_cutoff)
    ci = dist_row[keep].copy()
    cj = dist_col[keep].copy()
    cd = dist_data[keep]

    conf_theta = (0.25 * (1 + np.tanh(eta * (cd - r_min)))
                        * (1 + np.tanh(eta * (r_max - cd)))).astype(np.float64)
    conf_thetaII = (0.25 * (1 + np.tanh(eta * (cd - r_minII)))
                          * (1 + np.tanh(eta * (r_maxII - cd)))).astype(np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        conf_elec_ind = np.exp(-cd / screening_length) / cd
        conf_elec_ind = np.nan_to_num(conf_elec_ind, nan=0.0, posinf=0.0, neginf=0.0)

    return ci, cj, conf_theta, conf_thetaII, conf_elec_ind.astype(np.float64)


def compute_frequencies(seq_index):
    """Amino acid and contact frequencies."""
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)], dtype=np.float64)
    aa_norm = aa_freq / aa_freq.sum()
    contact_freq = np.outer(aa_norm, aa_norm).astype(np.float64)
    return aa_freq, contact_freq


# ─── FrustrationData ─────────────────────────────────────────────────────────

@dataclass
class FrustrationData:
    """Pre-computed arrays for fast frustration calculations.

    All arrays use the DCA 21-letter alphabet (gap at index 0).
    Contacts are stored in symmetric COO format (both (i,j) and (j,i)).
    """
    # Sequence
    L: int
    seq_index: np.ndarray           # (L,) int32

    # AWSEM scalars
    k_contact: float
    eta_sigma: float
    rho_0: float
    k_electrostatics: float

    # Contacts (symmetric COO)
    contact_i: np.ndarray           # (Nc,)
    contact_j: np.ndarray           # (Nc,)
    theta: np.ndarray               # (Nc,)
    tsw: np.ndarray                 # (Nc,)
    tsp: np.ndarray                 # (Nc,)

    # Per-residue
    burial_indicator: np.ndarray    # (L, 3)
    rho_r: np.ndarray               # (L,)

    # Gammas (DCA 21-letter): [direct, water, protein]
    gammas: np.ndarray              # (3, 21, 21)
    bg: np.ndarray                  # (21, 3)

    # Frequencies
    aa_freq: np.ndarray             # (21,)
    contact_freq: np.ndarray        # (21, 21)

    # Electrostatics
    charges: np.ndarray             # (21,)
    elec_phi: np.ndarray            # (L,)
    elec_ind_contacts: np.ndarray   # (Nc,)

    # Configurational (upper-triangle contacts)
    conf_contact_i: np.ndarray      # (n_conf,)
    conf_contact_j: np.ndarray      # (n_conf,)
    conf_theta: np.ndarray          # (n_conf,)
    conf_thetaII: np.ndarray        # (n_conf,)
    conf_elec_ind: np.ndarray       # (n_conf,)

    @property
    def Nc(self) -> int:
        return len(self.contact_i)

    @property
    def n_conf(self) -> int:
        return len(self.conf_contact_i)

    @classmethod
    def from_sparse(
        cls,
        dist_row, dist_col, dist_data, L, sequence,
        burial_gamma, direct_gamma, water_gamma, protein_gamma,
        k_contact=4.184,
        eta=5.0,
        r_min=4.5, r_max=6.5, r_minII=6.5, r_maxII=9.5,
        eta_sigma=7.0, rho_0=2.6,
        burial_kappa=4.0,
        burial_ro_min=(0.0, 3.0, 6.0),
        burial_ro_max=(3.0, 6.0, 9.0),
        min_seq_sep_rho=2,
        min_seq_sep_contact=0,
        distance_cutoff_contact=9.5,
        k_electrostatics=0.0,
        screening_length=10.0,
        min_seq_sep_elec=1,
        chain_breaks=None,
        elec_dist_row=None, elec_dist_col=None, elec_dist_data=None,
    ):
        """Build FrustrationData from sparse distance arrays and AWSEM parameters."""
        dist_row = np.asarray(dist_row, dtype=np.intp)
        dist_col = np.asarray(dist_col, dtype=np.intp)
        dist_data = np.asarray(dist_data, dtype=np.float64)

        seq_index = np.array([AA.find(aa) for aa in sequence], dtype=np.int32)

        # Structural computations
        rho_r = compute_rho(dist_row, dist_col, dist_data, L,
                            eta, r_min, r_max, min_seq_sep_rho, chain_breaks)

        ci, cj, theta, _thetaII, tsw, tsp = compute_contacts(
            dist_row, dist_col, dist_data, rho_r,
            eta, r_min, r_max, r_minII, r_maxII,
            eta_sigma, rho_0,
            min_seq_sep_contact, distance_cutoff_contact, chain_breaks)

        burial_indicator = compute_burial(rho_r, burial_kappa,
                                          burial_ro_min, burial_ro_max)

        # Gammas → DCA 21-letter
        bg = remap_burial_gamma(burial_gamma)
        dg = remap_gamma(direct_gamma)
        wg = remap_gamma(water_gamma)
        pg = remap_gamma(protein_gamma)
        gammas = np.stack([dg, wg, pg])  # (3, 21, 21)

        # Frequencies
        aa_freq, contact_freq = compute_frequencies(seq_index)

        # Electrostatics
        e_row = dist_row if elec_dist_row is None else np.asarray(elec_dist_row, dtype=np.intp)
        e_col = dist_col if elec_dist_col is None else np.asarray(elec_dist_col, dtype=np.intp)
        e_dat = dist_data if elec_dist_data is None else np.asarray(elec_dist_data, dtype=np.float64)

        elec_phi, elec_ind_contacts = compute_electrostatics(
            e_row, e_col, e_dat, L, ci, cj, seq_index,
            k_electrostatics, screening_length,
            min_seq_sep_elec, min_seq_sep_contact, chain_breaks)

        # Configurational contacts (upper triangle, no seq sep filter)
        conf_ci, conf_cj, conf_theta, conf_thetaII, conf_elec_ind = \
            compute_configurational_contacts(
                dist_row, dist_col, dist_data,
                eta, r_min, r_max, r_minII, r_maxII,
                distance_cutoff_contact, screening_length)

        C = np.ascontiguousarray
        return cls(
            L=L,
            seq_index=C(seq_index),
            k_contact=float(k_contact),
            eta_sigma=float(eta_sigma),
            rho_0=float(rho_0),
            k_electrostatics=float(k_electrostatics),
            contact_i=C(ci), contact_j=C(cj),
            theta=C(theta), tsw=C(tsw), tsp=C(tsp),
            burial_indicator=C(burial_indicator),
            rho_r=C(rho_r),
            gammas=C(gammas), bg=bg,
            aa_freq=aa_freq, contact_freq=C(contact_freq),
            charges=CHARGES.copy(),
            elec_phi=C(elec_phi),
            elec_ind_contacts=C(elec_ind_contacts),
            conf_contact_i=C(conf_ci), conf_contact_j=C(conf_cj),
            conf_theta=C(conf_theta), conf_thetaII=C(conf_thetaII),
            conf_elec_ind=C(conf_elec_ind),
        )

    @classmethod
    def from_awsem(cls, model):
        """Build FrustrationData from a constructed AWSEM model.

        Pulls the sparse distance matrices, gamma matrices and AWSEM
        parameters straight off the model (all are set as attributes during
        ``AWSEM.__init__``), then delegates to :meth:`from_sparse`.
        """
        dm = model._sparse_distance_matrix
        elec_dm = getattr(model, '_sparse_distance_matrix_elec', None)
        elec_kw = {}
        if elec_dm is not None:
            elec_kw = dict(
                elec_dist_row=elec_dm.row,
                elec_dist_col=elec_dm.col,
                elec_dist_data=elec_dm.data,
            )
        return cls.from_sparse(
            dist_row=dm.row, dist_col=dm.col, dist_data=dm.data,
            L=model.N, sequence=model.sequence,
            burial_gamma=model.burial_gamma,
            direct_gamma=model.direct_gamma,
            water_gamma=model.water_gamma,
            protein_gamma=model.protein_gamma,
            k_contact=model.k_contact,
            eta=model.eta,
            r_min=model.r_min, r_max=model.r_max,
            r_minII=model.r_minII, r_maxII=model.r_maxII,
            eta_sigma=model.eta_sigma, rho_0=model.rho_0,
            burial_kappa=model.burial_kappa,
            burial_ro_min=model.burial_ro_min,
            burial_ro_max=model.burial_ro_max,
            min_seq_sep_rho=model.min_sequence_separation_rho,
            min_seq_sep_contact=model.min_sequence_separation_contact,
            distance_cutoff_contact=model.distance_cutoff_contact,
            k_electrostatics=model.k_electrostatics,
            screening_length=model.electrostatics_screening_length,
            min_seq_sep_elec=model.min_sequence_separation_electrostatics,
            chain_breaks=model.chain_breaks,
            **elec_kw,
        )
