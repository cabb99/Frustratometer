import json
import warnings
import numpy as np
from ..utils import _path
from .. import frustration
from ..awsem import physics
from ..frustration.frustration import _AA as _AA_21
from .Structure import Structure, SparseMatrix
from .Frustratometer import Frustratometer
from .Gamma import Gamma
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import Path
from typing import List,Optional,Union

__all__ = ['AWSEM']

# Per-identity residue charge (one-letter): used for the external (DNA/ligand) charge field.
_CHARGE = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0}


class AWSEMParameters(BaseModel):
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    """Default parameters for AWSEM energy calculations."""
    k_contact: float = Field(4.184, description="Coefficient for contact potential. (kJ/mol)")
    
    #Density
    eta: float = Field(5.0, description="Sharpness of the distance-based switching function (Angstrom^-1).")
    rho_0: float = Field(2.6, description="Density cutoff defining buried residues.")
    min_sequence_separation_rho: Optional[int] = Field(2, description="Minimum sequence separation for density calculation.")

    #Burial potential
    burial_in_context: Optional[bool] = Field(True, description="For substructure objects, this may include interactions from remainder of protein in burial term.")
    burial_kappa: float = Field(4.0, description="Sharpness of the density-based switching function for the burial potential wells")
    burial_ro_min: List[float] = Field([0.0, 3.0, 6.0], description="Minimum radii for burial potential wells. (Angstrom)")
    burial_ro_max: List[float] = Field([3.0, 6.0, 9.0], description="Maximum radii for burial potential wells. (Angstrom)")
    
    #Direct contacts
    min_sequence_separation_contact: Optional[int] = Field(0, description="Minimum sequence separation for contact calculation.")
    distance_cutoff_contact: Optional[float] = Field(9.5, description="Distance cutoff for contact calculation. (Angstrom)")
    r_min: float = Field(4.5, description="Minimum distance for direct contact potential. (Angstrom)")
    r_max: float = Field(6.5, description="Maximum distance for direct contact potential. (Angstrom)")
    
    #Mediated contacts
    gamma: Union[Path,Gamma] = Field(_path/'data'/'AWSEM_2015.json', description="File or Gamma object containing the Gamma values")
    r_minII: float = Field(6.5, description="Minimum distance for mediated contact potential. (Angstrom)")
    r_maxII: float = Field(9.5, description="Maximum distance for mediated contact potential. (Angstrom)")
    eta_sigma: float = Field(7.0, description="Sharpness of the density-based switching function between protein-mediated and water-mediated contacts.")
    

    #Membrane
    membrane: bool = Field(False, description="Enable membrane-aware energy calculation.")
    membrane_gamma: Union[Path,Gamma] = Field(_path/'data'/'AWSEM_membrane_2015.json', description="File or Gamma object containing the membrane Gamma values (for membrane proteins)")
    eta_switching: float = Field(1.0, description="Sharpness of the alpha switching function for membrane contact blending (Angstrom^-1).")
    z_m: float = Field(15.0, description="Membrane half-thickness (Angstrom).")
    z_source: str = Field("Auto", description="Atom name used for membrane Z-coordinate reference (e.g. Auto, CB or CA).")
    membrane_center: float = Field(0.0, description="Z-coordinate of membrane center (Angstrom).")
    k_membrane: float = Field(4.184, description="Strength of Zim membrane potential (kJ/mol).")
    k_m: float = Field(2.0, description="Sharpness of Zim boundary switching function (Angstrom^-1).")
    k_relative_mem: float = Field(1.0, description="Scalar multiplier for membrane gamma matrices (amplifies membrane contact interactions).")
    zim: Optional[List[float]] = Field(None, description="Per-residue hydrophobicity values (DGwoct). If None, computed from Wimley-White scale.")

    #Electrostatics
    min_sequence_separation_electrostatics: Optional[int] = Field(1, description="Minimum sequence separation for electrostatics calculation.")
    k_electrostatics: float = Field(17.3636, description="Coefficient for electrostatic interactions. (kJ/mol)")
    electrostatics_screening_length: float = Field(10, description="Screening length for electrostatic interactions. (Angstrom)")

class AWSEM(Frustratometer):
    #Mapping to DCA
    q = 20
    aa_map_awsem_list = [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18] #A gap has no energy
    aa_map_awsem_x, aa_map_awsem_y = np.meshgrid(aa_map_awsem_list, aa_map_awsem_list, indexing='ij')
    
    def __init__(self,
                 pdb_structure: Structure,
                 sequence: Union[str, None] =None,
                 expose_indicator_functions: bool=False,
                 sparse: bool=True,
                 backend: str='numpy',
                 fast: Optional[bool]=None,
                 **parameters):
        """
        Generate AWSEM object

        Parameters
        ----------
        pdb_structure : object
            Structure object generated by Structure class
        sequence :  str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code.
        expose_indicator_functions: bool
            If set to True, indicator functions of the contact and burial energy terms can be accessed by user.
        sparse : bool
            Representation of the distance data. If True (default) the model is built from the sparse
            distance matrix; if False a dense distance matrix is materialized. Either way the Potts
            model is stored sparsely (couplings only at contacts); the dense ``(N, N, Q, Q)`` coupling
            tensor is only ever an on-demand ``potts_model['J']`` view.
        backend : str
            Compute engine: ``'numpy'`` (default, reference) evaluates every query with numpy;
            ``'numba'`` / ``'cuda'`` accelerate native/single/mutational/configurational on the
            precomputed channels and require a sparse Structure.
        fast : bool, optional
            Deprecated. ``fast=True`` maps to ``backend='numba'`` (or the given numba/cuda backend),
            ``fast=False`` to ``backend='numpy'``.

        Returns
        -------
        AWSEM object
        """
        if fast is not None:
            warnings.warn(
                "AWSEM(fast=...) is deprecated; use backend='numpy'|'numba'|'cuda'.",
                DeprecationWarning, stacklevel=2)
            backend = (backend if backend in ('numba', 'cuda') else 'numba') if fast else 'numpy'
        if backend not in ('numpy', 'numba', 'cuda'):
            raise ValueError(f"backend must be 'numpy', 'numba', or 'cuda', got {backend!r}")
        self.backend = backend

        #Set attributes
        p = AWSEMParameters(**parameters)
        if p.min_sequence_separation_contact is None:
            p.min_sequence_separation_contact = 1
        if p.min_sequence_separation_rho is None:
            p.min_sequence_separation_rho = 1
        if p.min_sequence_separation_electrostatics is None:
            p.min_sequence_separation_electrostatics = 1

        for field, value in p:
            setattr(self, field, value)
        
        #Gamma parameters
        if isinstance(p.gamma, Gamma):
            gamma = p.gamma
        elif isinstance(p.gamma, (Path, str)):
            gamma_path = Path(p.gamma)
            local_candidate = Path.cwd() / p.gamma
            if local_candidate.exists():   
                gamma_path = local_candidate
            elif gamma_path.suffix == '' and gamma_path.parent == Path('.'):
                data_candidate = _path / 'data' / p.gamma.with_suffix('.json')
                if data_candidate.exists():
                    gamma_path = data_candidate
                else:
                    raise FileNotFoundError(f"Did not found the gamma file: {str(data_candidate)}")
            else:
                raise FileNotFoundError(f"Did not found the gamma file: {p.gamma}")
            gamma = Gamma(gamma_path)
        else:
            raise ValueError("Gamma parameter must be a path or a Gamma object.")
                
        self.gamma=gamma
        self.burial_gamma = gamma['Burial'].T
        self.direct_gamma = gamma['Direct'][0]
        self.protein_gamma = gamma['Protein'][0]
        self.water_gamma = gamma['Water'][0]
        self.burial_in_context=p.burial_in_context

        # Membrane gamma loading
        if p.membrane:
            if isinstance(p.membrane_gamma, Gamma):
                membrane_gamma_obj = p.membrane_gamma
            elif isinstance(p.membrane_gamma, Path):
                membrane_gamma_obj = Gamma(p.membrane_gamma)
            else:
                raise ValueError("membrane_gamma parameter must be a path or a Gamma object.")
            self.membrane_burial_gamma = membrane_gamma_obj['Burial'].T
            self.membrane_direct_gamma = membrane_gamma_obj['Direct'][0] * p.k_relative_mem
            self.membrane_protein_gamma = membrane_gamma_obj['Protein'][0] * p.k_relative_mem
            self.membrane_water_gamma = membrane_gamma_obj['Water'][0] * p.k_relative_mem

        #Structure details
        self.full_to_aligned_index_dict=pdb_structure.full_to_aligned_index_dict
        if sequence is None:
            self.sequence=pdb_structure.sequence
        else:
            self.sequence=sequence
        self.structure=pdb_structure.structure
        self.chain=pdb_structure.chain
        self.pdb_file=pdb_structure.pdb_file
        self.init_index_shift=pdb_structure.init_index_shift
        self.distance_matrix=pdb_structure.distance_matrix
        self.full_pdb_distance_matrix=pdb_structure.full_pdb_distance_matrix
        self.chain_breaks=pdb_structure.chain_breaks
        self._distance_is_sparse = pdb_structure._is_sparse
        #Sparse matrices
        self._sparse_distance_matrix = None
        self._sparse_distance_matrix_elec = None
        if self._distance_is_sparse:
            self._sparse_distance_matrix = pdb_structure._sparse_distance_matrix
            self._sparse_distance_matrix_elec = pdb_structure._sparse_distance_matrix_elec
            if not sparse:
                # Dense Potts model from sparse distances.
                from frustratometer.pdb import distance as _pdb_dist
                dense_dm = _pdb_dist.get_dense_distance_matrix(
                    pdb_file=pdb_structure.pdb_file,
                    chain=pdb_structure.chain,
                    method=pdb_structure.distance_matrix_method)
                if pdb_structure.seq_selection is not None:
                    dense_dm = dense_dm[
                        pdb_structure.init_index_shift:pdb_structure.fin_index_shift,
                        pdb_structure.init_index_shift:pdb_structure.fin_index_shift]
                self.distance_matrix = dense_dm
                self.full_pdb_distance_matrix = dense_dm
                self._distance_is_sparse = False
        else:
            self._sparse_distance_matrix = None
            self._sparse_distance_matrix_elec = None
        selection_CA = self.structure.select('name CA')
        selection_CB = self.structure.select('name CB or (resname GLY IGL and name CA)')

        resid = selection_CB['residue'].to_numpy()
        self.resid=resid
        self._cb_coords = selection_CB.get_coordinates().to_numpy()  # (N, 3) CB (CA for GLY), residue order
        self.N=len(self.resid)
        assert self.N == len(self.sequence), "The pdb is incomplete. Try setting 'repair_pdb=True' when constructing the Structure object."

        # Membrane alpha and phi_z computation
        if p.membrane:
            # OpenAWSEM has two membrane Zim behaviors:
            # - membrane_preassigned_term: CB (or CA fallback)
            # - membrane_term (simple, no preassigned file): CA
            # May need to adjust in the future to set custom z_source
            if p.z_source == "Auto":
                z_source = selection_CB if p.zim is not None else selection_CA
            elif p.z_source == "CB":
                z_source = selection_CB
            elif p.z_source == "CA":
                z_source = selection_CA
            else:
                raise ValueError(f"Invalid z_source value: {p.z_source}. Must be 'Auto', 'CB', or 'CA'.")
            z_coords = z_source.get_coordinates().to_numpy()[:, 2] - p.membrane_center
            self.z_coords = z_coords
            # alpha_i ≈ 1 inside membrane, ≈ 0 outside
            self.alpha = 0.5 * np.tanh(p.eta_switching * (z_coords + p.z_m)) + 0.5 * np.tanh(p.eta_switching * (p.z_m - z_coords))
            # phi_z for Zim potential (same shape but sharper boundary via k_m)
            self.phi_z = 0.5 * np.tanh(p.k_m * (z_coords + p.z_m)) + 0.5 * np.tanh(p.k_m * (p.z_m - z_coords))
            # Per-residue DGwoct values
            # Load Wimley-White DGwoct scale (20 values in AWSEM aa order)
            _AA = 'ARNDCQEGHILKMFPSTWYV'
            with open(_path / 'data' / 'wimley_white_dgwoct.json') as f:
                ww = json.load(f)
            self._dgwoct_scale = np.array(ww['DGwoct'], dtype=np.float64)  # (20,)
            if p.zim is not None:
                self.dgwoct = np.array(p.zim, dtype=np.float64)
            else:
                self.dgwoct = np.array([self._dgwoct_scale[_AA.index(aa)] for aa in self.sequence], dtype=np.float64)
            # Per-residue, per-aa Zim contribution to h, shape (N, 21).
            # Build paths add this to h; zim_energy() reads it at the native sequence.
            if p.zim is not None:
                # Preassigned dgwoct is per-position, independent of aa type.
                zim_col = p.k_membrane * self.dgwoct * self.phi_z  # (N,)
                self._zim_h = np.broadcast_to(zim_col[:, None],
                                              (self.N, len(self.aa_map_awsem_list))).copy()
            else:
                # dgwoct_scale (length 20) reshuffled into the 21-letter alphabet.
                zim_h_20 = p.k_membrane * self._dgwoct_scale[None, :] * self.phi_z[:, None]
                self._zim_h = zim_h_20[:, self.aa_map_awsem_list]
        else:
            self.alpha = None
            self.phi_z = None
            self.dgwoct = None
            self.z_coords = None
            self._dgwoct_scale = None
            self._zim_h = None

        self._decoy_fluctuation = {}
        self.minimally_frustrated_threshold=.78
        self._frustration_data = None
        self._fast_backend = None if backend == 'numpy' else backend

        if self._fast_backend is not None:
            if not self._distance_is_sparse:
                raise ValueError(f"backend={backend!r} requires a sparse Structure (sparse=True)")
            self._setup_fast(p, pdb_structure)
        else:
            self._build_physics(p, expose_indicator_functions, pdb_structure)

    def _build_physics(self, p, expose, pdb_structure):
        """Single physics path: structural indicators -> sparse Potts model (+ electrostatics
        sidecar and optional indicator exposure). Used for every non-fast model and to
        materialize the Potts model on demand for fast models."""
        (rho_r, ci, cj, theta_c, thetaII_c,
         sigma_water_c, sigma_protein_c, burial_indicator) = self._structural_indicators(p, pdb_structure)

        self.rho = None
        self.rho_r = rho_r
        self._burial_indicator = burial_indicator
        self._sigma_water = None
        self._sigma_protein = None

        burial_energy = physics.build_burial_energy(self, p, burial_indicator)
        self.burial_energy = burial_energy

        self.mask = self._main_mask(p)
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)

        self.sparse_potts_model = physics.build_sparse_potts(
            self, p, ci, cj, theta_c, thetaII_c, sigma_water_c, sigma_protein_c, burial_energy)
        self._potts_model = {'h': self.sparse_potts_model['h'], 'J': None}
        self._native_energy = None
        self.contact_energy = None
        self._elec_data = self._build_elec_data(p)

        if expose:
            self._expose_indicators(p, ci, cj, theta_c, thetaII_c, sigma_water_c, sigma_protein_c)

    def _structural_indicators(self, p, pdb_structure):
        """Per-residue density/burial and per-contact theta/thetaII/sigma scalars.

        Branches only on the distance representation (sparse COO vs dense matrix); both
        return the same contact-level arrays feeding the single sparse Potts builder."""
        if self._distance_is_sparse:
            dm = self._sparse_distance_matrix
            if self.burial_in_context:
                full_dm = self.full_pdb_distance_matrix
                sel_dm = full_dm if isinstance(full_dm, SparseMatrix) else dm
            else:
                sel_dm = dm
            rho_mask = frustration.compute_mask_sparse(
                sel_dm.row, sel_dm.col, sel_dm.data, sel_dm.shape,
                maximum_contact_distance=None,
                minimum_sequence_separation=p.min_sequence_separation_rho,
                chain_breaks=self.chain_breaks)
            rho_dists = sel_dm.lookup(rho_mask.row, rho_mask.col)
            rho_vals = 0.25 * (1 + np.tanh(p.eta * (rho_dists - p.r_min))) * (1 + np.tanh(p.eta * (p.r_max - rho_dists)))
            rho_r = np.bincount(rho_mask.row, weights=rho_vals, minlength=sel_dm.shape).astype(np.float64)
            if sel_dm.shape != dm.shape and self.burial_in_context:
                self.init_index_shift = pdb_structure.init_index_shift
                self.fin_index_shift = pdb_structure.fin_index_shift
                rho_r = rho_r[self.init_index_shift:self.fin_index_shift]
            contact_mask = frustration.compute_mask_sparse(
                dm.row, dm.col, dm.data, dm.shape,
                maximum_contact_distance=p.distance_cutoff_contact,
                minimum_sequence_separation=p.min_sequence_separation_contact,
                chain_breaks=self.chain_breaks)
            ci = contact_mask.row
            cj = contact_mask.col
            contact_dists = dm.lookup(ci, cj)
        else:
            selected = self.full_pdb_distance_matrix if self.burial_in_context else self.distance_matrix
            rho_mask = frustration.compute_mask(
                selected, maximum_contact_distance=None,
                minimum_sequence_separation=p.min_sequence_separation_rho,
                chain_breaks=self.chain_breaks)
            rho = (0.25 * (1 + np.tanh(p.eta * (selected - p.r_min)))
                        * (1 + np.tanh(p.eta * (p.r_max - selected))) * rho_mask)
            rho_r = rho.sum(axis=1)
            if self.full_pdb_distance_matrix.shape != self.distance_matrix.shape and self.burial_in_context:
                self.init_index_shift = pdb_structure.init_index_shift
                self.fin_index_shift = pdb_structure.fin_index_shift
                rho_r = rho_r[self.init_index_shift:self.fin_index_shift]
            contact_mask = frustration.compute_mask(
                self.distance_matrix,
                maximum_contact_distance=p.distance_cutoff_contact,
                minimum_sequence_separation=p.min_sequence_separation_contact,
                chain_breaks=self.chain_breaks)
            ci, cj = np.where(contact_mask)
            contact_dists = self.distance_matrix[ci, cj]

        theta_c = 0.25 * (1 + np.tanh(p.eta * (contact_dists - p.r_min))) * (1 + np.tanh(p.eta * (p.r_max - contact_dists)))
        thetaII_c = 0.25 * (1 + np.tanh(p.eta * (contact_dists - p.r_minII))) * (1 + np.tanh(p.eta * (p.r_maxII - contact_dists)))
        sigma_water_c = 0.25 * (1 - np.tanh(p.eta_sigma * (rho_r[ci] - p.rho_0))) * (1 - np.tanh(p.eta_sigma * (rho_r[cj] - p.rho_0)))
        sigma_protein_c = 1 - sigma_water_c

        rho_b = np.expand_dims(rho_r, 1)
        burial_indicator = (np.tanh(p.burial_kappa * (rho_b - p.burial_ro_min))
                            + np.tanh(p.burial_kappa * (p.burial_ro_max - rho_b)))
        return (rho_r, np.asarray(ci), np.asarray(cj), theta_c, thetaII_c,
                sigma_water_c, sigma_protein_c, burial_indicator)

    def _elec_aware_cutoffs(self, p):
        if p.k_electrostatics != 0:
            self.sequence_cutoff = min(p.min_sequence_separation_electrostatics, p.min_sequence_separation_contact)
            self.distance_cutoff = None
        else:
            self.sequence_cutoff = p.min_sequence_separation_contact
            self.distance_cutoff = p.distance_cutoff_contact

    def _main_mask(self, p):
        """Electrostatics-aware contact mask, from sparse COO or dense distances."""
        self._elec_aware_cutoffs(p)
        if self._distance_is_sparse:
            dm = self._sparse_distance_matrix
            return frustration.compute_mask_sparse(
                dm.row, dm.col, dm.data, dm.shape,
                maximum_contact_distance=self.distance_cutoff,
                minimum_sequence_separation=self.sequence_cutoff,
                chain_breaks=self.chain_breaks)
        return frustration.compute_mask(
            self.distance_matrix,
            maximum_contact_distance=self.distance_cutoff,
            minimum_sequence_separation=self.sequence_cutoff,
            chain_breaks=self.chain_breaks)

    def _build_elec_data(self, p):
        """Electrostatics sidecar for the sparse Potts model (None when k_electrostatics==0)."""
        if p.k_electrostatics == 0:
            return None
        if self._distance_is_sparse:
            return frustration.build_elec_data_sparse(
                self._sparse_distance_matrix_elec, self.mask,
                self.sequence, self.sparse_potts_model,
                p.k_electrostatics, p.electrostatics_screening_length,
                p.min_sequence_separation_electrostatics,
                chain_breaks=self.chain_breaks,
                mask_sequence_cutoff=self.sequence_cutoff if self.distance_cutoff is None else None,
                mask_chain_breaks=self.chain_breaks if self.distance_cutoff is None else None,
            )
        return frustration.build_elec_data(
            self.distance_matrix, self.mask, self.sequence,
            self.sparse_potts_model,
            p.k_electrostatics, p.electrostatics_screening_length,
            p.min_sequence_separation_electrostatics,
            chain_breaks=self.chain_breaks,
        )

    def _expose_indicators(self, p, ci, cj, theta_c, thetaII_c, sigma_water_c, sigma_protein_c):
        """Populate the (L, L) indicator/gamma_array contract for the optimization consumers."""
        self._start_indicator_exposure(p)
        self.indicators += physics.pair_indicators_dense(
            self, p, ci, cj, theta_c, thetaII_c, sigma_water_c, sigma_protein_c)
        self.indicator_contact_i = None
        self.indicator_contact_j = None
        if p.k_electrostatics != 0:
            electrostatics_indicator = self._electrostatics_indicator(p)
            charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            charges2 = charges[:, np.newaxis] * charges[np.newaxis, :]
            self.indicators.append(electrostatics_indicator)
            temp_gamma = 0.5 * p.k_electrostatics * charges2[self.aa_map_awsem_x, self.aa_map_awsem_y]
            temp_gamma[0, :] = 0
            temp_gamma[:, 0] = 0
            self.gamma_array.append(temp_gamma)

    def _electrostatics_indicator(self, p):
        """Dense (L, L) screened-Coulomb indicator used by the exposed gamma contract."""
        L = self.N
        if self._distance_is_sparse:
            elec_dm = self._sparse_distance_matrix_elec
            e_mask = frustration.compute_mask_sparse(
                elec_dm.row, elec_dm.col, elec_dm.data, elec_dm.shape,
                maximum_contact_distance=None,
                minimum_sequence_separation=p.min_sequence_separation_electrostatics,
                chain_breaks=self.chain_breaks)
            e_dists = elec_dm.lookup(e_mask.row, e_mask.col)
            electrostatics_indicator = np.zeros((L, L))
            with np.errstate(divide='ignore', invalid='ignore'):
                vals = np.exp(-e_dists / p.electrostatics_screening_length) / e_dists
                vals[np.isnan(vals)] = 0.0
                vals[np.isinf(vals)] = 0.0
            electrostatics_indicator[e_mask.row, e_mask.col] = vals
            return electrostatics_indicator
        electrostatics_mask = frustration.compute_mask(
            self.distance_matrix, maximum_contact_distance=None,
            minimum_sequence_separation=p.min_sequence_separation_electrostatics,
            chain_breaks=self.chain_breaks)
        return (1 / (self.distance_matrix + 1E-6)
                * np.exp(-self.distance_matrix / p.electrostatics_screening_length)
                * electrostatics_mask)

    def _setup_fast(self, p, pdb_structure):
        """Fast (numba/cuda) mode: build the FrustrationData channels and defer the Potts model.

        ``p`` and ``pdb_structure`` are retained so the sparse Potts model can be materialized
        lazily (``_ensure_potts_model``) if an inherited method needs it."""
        from ..frustration.data import FrustrationData

        self._init_params = p
        self._init_pdb_structure = pdb_structure
        self._frustration_data = FrustrationData.from_awsem(self)

        self.mask = self._main_mask(p)
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)

        self.rho = None
        self.rho_r = self._frustration_data.rho_r
        self._burial_indicator = self._frustration_data.burial_indicator
        self._sigma_water = None
        self._sigma_protein = None

        self._native_energy = None
        self._elec_data = None
        self.sparse_potts_model = None
        self._potts_model = {}
        self.contact_energy = None

        self._device_data = None
        if self._fast_backend == 'cuda':
            try:
                from ..frustration.cuda import FrustrationCUDA
                self._device_data = FrustrationCUDA(self._frustration_data)
            except Exception as e:
                raise RuntimeError(
                    f"backend='cuda' was requested but the CUDA backend could not be "
                    f"initialized: {e}. Use backend='numba' to run on CPU."
                ) from e

    def _ensure_potts_model(self):
        """Lazily build the sparse Potts model for a fast-mode model, reusing the single
        physics path so no dense (L, L, Q, Q) tensor is allocated."""
        if self.sparse_potts_model is not None or self._frustration_data is None:
            return
        if getattr(self, '_building_potts_model', False):
            return
        self._building_potts_model = True
        try:
            self._build_physics(self._init_params, expose=False,
                                pdb_structure=self._init_pdb_structure)
        finally:
            self._building_potts_model = False

    def _start_indicator_exposure(self, p):
        """Set up burial indicators and gamma arrays (common to sparse and dense)."""
        burial_indicator = self._burial_indicator

        # Burial indicators and gammas are the same regardless of membrane flag
        self.indicators = [burial_indicator[:, 0], burial_indicator[:, 1], burial_indicator[:, 2]]

        temp_burial_gamma = self.burial_gamma[self.aa_map_awsem_list].copy()
        temp_burial_gamma[0] = 0
        temp_burial_gamma *= -0.5 * p.k_contact
        self.gamma_array = [temp_burial_gamma[:, 0], temp_burial_gamma[:, 1], temp_burial_gamma[:, 2]]

        if p.membrane:
            # Add phi_z indicator and zim gamma
            zim_gamma = p.k_membrane * self._dgwoct_scale[self.aa_map_awsem_list].copy()
            zim_gamma[0] = 0
            self.indicators.append(self.phi_z)
            self.gamma_array.append(zim_gamma)

            # Water contact gammas
            for contact_gamma in [self.direct_gamma, self.protein_gamma, self.water_gamma]:
                temp_gamma = contact_gamma[self.aa_map_awsem_x, self.aa_map_awsem_y].copy()
                temp_gamma[0, :] = 0
                temp_gamma[:, 0] = 0
                temp_gamma *= -0.5 * p.k_contact
                self.gamma_array.append(temp_gamma)
            # Membrane contact gammas
            for contact_gamma in [self.membrane_direct_gamma, self.membrane_protein_gamma, self.membrane_water_gamma]:
                temp_gamma = contact_gamma[self.aa_map_awsem_x, self.aa_map_awsem_y].copy()
                temp_gamma[0, :] = 0
                temp_gamma[:, 0] = 0
                temp_gamma *= -0.5 * p.k_contact
                self.gamma_array.append(temp_gamma)
        else:
            for contact_gamma in [self.direct_gamma, self.protein_gamma, self.water_gamma]:
                temp_gamma = contact_gamma[self.aa_map_awsem_x, self.aa_map_awsem_y].copy()
                temp_gamma[0, :] = 0
                temp_gamma[:, 0] = 0
                temp_gamma *= -0.5 * p.k_contact
                self.gamma_array.append(temp_gamma)

    def _get_configurational_contact_data(self):
        """Extract upper-triangle contact distances and indices for configurational frustration.

        Works for both sparse and dense distance matrices.

        Returns
        -------
        distances : np.ndarray (C,)
            Contact distances (upper triangle, within distance_cutoff_contact).
        indices1 : np.ndarray (C,)
            Row indices of contacts.
        indices2 : np.ndarray (C,)
            Column indices of contacts.
        """
        if self._distance_is_sparse:
            dm = self._sparse_distance_matrix
            # Upper triangle: row < col
            upper = dm.row < dm.col
            uci = dm.row[upper]
            ucj = dm.col[upper]
            udists = dm.data[upper]
            valid = (udists < self.distance_cutoff_contact) & (udists > 0)
            return udists[valid], uci[valid], ucj[valid]
        else:
            dm = self.distance_matrix
            n = dm.shape[0]
            tri_upper_indices = np.triu_indices(n, k=1)
            tri_dists = dm[tri_upper_indices]
            valid = (tri_dists < self.distance_cutoff_contact) & (tri_dists > 0)
            return tri_dists[valid], tri_upper_indices[0][valid], tri_upper_indices[1][valid]

    def _compute_sigma_at_pairs(self, n1, n2):
        """Compute sigma_water and sigma_protein at specific residue pairs.

        Uses pre-computed dense sigma if available, otherwise computes from rho_r.
        """
        if self._sigma_water is not None:
            return self._sigma_water[n1, n2], self._sigma_protein[n1, n2]
        # Sparse mode: compute from rho_r
        rho1 = self.rho_r[n1]
        rho2 = self.rho_r[n2]
        sw = 0.25 * (1 - np.tanh(self.eta_sigma * (rho1 - self.rho_0))) * (1 - np.tanh(self.eta_sigma * (rho2 - self.rho_0)))
        return sw, 1 - sw

    def _configurational_pair_energy(self, c, n1, n2, q1, q2, theta, thetaII,
                                     electrostatics_indicator, charges, burial_indicator):
        """Energy of a configurational contact event: contact geometry ``c`` (theta/thetaII/
        elec indicator at that contact), local environment at positions ``n1``/``n2`` (burial
        and sigma resampled from rho), and identities ``q1``/``q2`` (AWSEM 20-letter). When
        membrane is enabled, burial blends by per-residue alpha (plus the Zim field) and the
        contact term blends by pairwise alpha. Shared by the native-energy and decoy loops."""
        burial_energy1 = (-0.5 * self.k_contact * self.burial_gamma[q1] * burial_indicator[n1]).sum(axis=0)
        burial_energy2 = (-0.5 * self.k_contact * self.burial_gamma[q2] * burial_indicator[n2]).sum(axis=0)
        if self.alpha is not None:
            m_burial1 = (-0.5 * self.k_contact * self.membrane_burial_gamma[q1] * burial_indicator[n1]).sum(axis=0)
            m_burial2 = (-0.5 * self.k_contact * self.membrane_burial_gamma[q2] * burial_indicator[n2]).sum(axis=0)
            burial_energy1 = (1 - self.alpha[n1]) * burial_energy1 + self.alpha[n1] * m_burial1
            burial_energy2 = (1 - self.alpha[n2]) * burial_energy2 + self.alpha[n2] * m_burial2
            burial_energy1 += self.k_membrane * self._dgwoct_scale[q1] * self.phi_z[n1]
            burial_energy2 += self.k_membrane * self._dgwoct_scale[q2] * self.phi_z[n2]

        sigma_water_val, sigma_protein_val = self._compute_sigma_at_pairs(n1, n2)
        direct = theta[c] * self.direct_gamma[q1, q2]
        water_mediated = sigma_water_val * thetaII[c] * self.water_gamma[q1, q2]
        protein_mediated = sigma_protein_val * thetaII[c] * self.protein_gamma[q1, q2]
        contact_energy = -self.k_contact * (direct + water_mediated + protein_mediated)
        if self.alpha is not None:
            m_direct = theta[c] * self.membrane_direct_gamma[q1, q2]
            m_water = sigma_water_val * thetaII[c] * self.membrane_water_gamma[q1, q2]
            m_protein = sigma_protein_val * thetaII[c] * self.membrane_protein_gamma[q1, q2]
            m_contact = -self.k_contact * (m_direct + m_water + m_protein)
            alpha_ij = self.alpha[n1] * self.alpha[n2]
            contact_energy = (1 - alpha_ij) * contact_energy + alpha_ij * m_contact
        electrostatics_energy = self.k_electrostatics * electrostatics_indicator[c] * charges[q1] * charges[q2]
        return burial_energy1 + burial_energy2 + contact_energy + electrostatics_energy

    def compute_configurational_decoy_statistics(self, n_decoys=4000,aa_freq=None):
        # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        _AA='ARNDCQEGHILKMFPSTWYV'
        if aa_freq is None:
            seq_index = np.array([_AA.find(aa) for aa in self.sequence])
            N=self.N
        else:
            N=self.N*10
            total = sum(aa_freq)
            probabilities = [freq / total for freq in aa_freq.ravel()]
            seq_index = np.random.choice(a=len(aa_freq), size=N, p=probabilities)
        
        distances, _, _ = self._get_configurational_contact_data()

        burial_indicator = self._burial_indicator

        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - distances))) # (c,)
        thetaII = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - distances))) #(c,)
           
        charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        electrostatics_indicator = np.exp(-distances / self.electrostatics_screening_length) / distances

        decoy_energies=np.zeros(n_decoys)
        for i in range(n_decoys):
            c=np.random.randint(0,len(distances))
            n1=np.random.randint(0,self.N)
            n2=np.random.randint(0,self.N)
            qi1=np.random.randint(0,N)
            qi2=np.random.randint(0,N)
            q1=seq_index[qi1]
            q2=seq_index[qi2]
            decoy_energies[i] = self._configurational_pair_energy(
                c, n1, n2, q1, q2, theta, thetaII, electrostatics_indicator, charges, burial_indicator)

        mean_decoy_energy = np.mean(decoy_energies)
        std_decoy_energy = np.std(decoy_energies)
        return mean_decoy_energy, std_decoy_energy
    
    def compute_configurational_energies(self):
        _AA='ARNDCQEGHILKMFPSTWYV'
        seq_index = np.array([_AA.find(aa) for aa in self.sequence])
        distances, indices1, indices2 = self._get_configurational_contact_data()
        n_contacts=len(distances)
        n = self.N

        burial_indicator = self._burial_indicator

        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - distances))) # (c,)
        thetaII = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - distances))) #(c,)
           
        charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        electrostatics_indicator = np.exp(-distances / self.electrostatics_screening_length) / distances

        configurational_energies=np.full((n,n), np.nan) # masked pairs will be left as nan
        for c in range(n_contacts):
            n1=indices1[c]
            n2=indices2[c]
            q1=seq_index[n1]
            q2=seq_index[n2]
            energy = self._configurational_pair_energy(
                c, n1, n2, q1, q2, theta, thetaII, electrostatics_indicator, charges, burial_indicator)
            configurational_energies[n1,n2]=energy
            configurational_energies[n2,n1]=energy
        return configurational_energies
    
    def select_residues(self, selection):
        """Resolve a residue selection to 0-based model residue indices.

        ``selection`` may be a molselect string evaluated on this model's structure
        (e.g. ``'chain A'``, ``'resid 10 to 40'``, ``'resname GLY'``), a boolean mask
        over ``range(N)``, or an integer index array (returned as-is)."""
        if isinstance(selection, str):
            chosen = set(np.unique(self.structure.select(selection)['residue'].to_numpy()).tolist())
            return np.array([i for i, r in enumerate(self.resid) if r in chosen], dtype=np.intp)
        sel = np.asarray(selection)
        if sel.dtype == bool:
            return np.where(sel)[0].astype(np.intp)
        return sel.astype(np.intp)

    def _resolve_active(self, active_residues=None, active_selection=None, static_selection=None):
        """Resolve active residue indices from an index/mask, an active molselect string,
        or a static molselect string (active = complement of static)."""
        if active_residues is not None:
            return self.select_residues(active_residues)
        if active_selection is not None:
            return self.select_residues(active_selection)
        if static_selection is not None:
            static = set(self.select_residues(static_selection).tolist())
            return np.array([i for i in range(self.N) if i not in static], dtype=np.intp)
        raise ValueError("Provide one of active_residues, active_selection, or static_selection.")

    def _charge_field(self, charge_coords, charges, charge_k, charge_screening):
        """(N, 21) field on h from external static point charges (e.g. DNA phosphates)."""
        from ..awsem.physics import external_charge_field
        aa_charges = np.array([_CHARGE.get(aa, 0.0) for aa in _AA_21])
        k = 17.3636 if charge_k is None else charge_k
        screening = self.electrostatics_screening_length if charge_screening is None else charge_screening
        return external_charge_field(self._cb_coords, charge_coords, charges,
                                     aa_charges, k, screening)

    def fold_static_context(self, active_residues=None, *, active_selection=None, static_selection=None,
                            charge_coords=None, charges=None, charge_k=None, charge_screening=None):
        """Reduce this model to a static-context model over the active residues.

        Residues outside the active set are held at their native identity (the "static
        context"): their couplings to the active residues fold into the active fields and
        their mutual couplings into a constant energy ``offset``. Returns
        ``(reduced_potts, offset)`` where ``reduced_potts`` is a sparse Potts model over
        the active residues. Requires a sparse Potts model (built with ``sparse=True``;
        fast models materialize one on demand).

        The active set is given by exactly one of ``active_residues`` (index array /
        boolean mask over ``range(N)``), ``active_selection`` (a molselect string), or
        ``static_selection`` (a molselect string; active = its complement).

        ``charge_coords`` (M, 3) and ``charges`` (M,) add an external static charge field
        (e.g. DNA phosphates) to the active residues via a screened Coulomb potential
        (constant ``charge_k`` kJ/mol, default 17.3636; ``charge_screening`` Angstrom,
        default the model's electrostatics screening length).

        Protein-protein electrostatics are not yet folded; use ``k_electrostatics=0``.
        """
        from ..awsem.physics import fold_static_context as _fold
        if self.k_electrostatics != 0:
            raise NotImplementedError(
                "Static-context folding does not yet include protein electrostatics; "
                "rebuild with k_electrostatics=0 (external charge_coords are still supported).")
        self._ensure_potts_model()
        if getattr(self, 'sparse_potts_model', None) is None:
            raise ValueError("fold_static_context requires a sparse Potts model (sparse=True).")
        active = self._resolve_active(active_residues, active_selection, static_selection)
        seq_index = np.array([_AA_21.index(aa) for aa in self.sequence])
        extra_field = None
        if charge_coords is not None:
            extra_field = self._charge_field(charge_coords, charges, charge_k, charge_screening)
        return _fold(self.sparse_potts_model, seq_index, active, extra_field=extra_field)

    def _get_fast_module(self):
        """Return the frustration backend module for ``self._fast_backend`` via the registry."""
        from ..frustration.backends import get_backend
        return get_backend(self._fast_backend or 'numba')

    def _get_fast_data(self):
        """Return cached FrustrationCUDA (CUDA) or FrustrationData for fast-path calls."""
        if self._fast_backend == 'cuda' and self._device_data is not None:
            return self._device_data
        return self._frustration_data

    def configurational_frustration(self,aa_freq=None, correction=0, n_decoys=4000, seed=42):
        # The fast (numba/cuda) configurational kernels evaluate only the three water
        # channels, so they do not yet reproduce the membrane blend; route membrane models
        # through the numpy path, which does. (Frozen native/single/mutational are membrane-
        # correct on the fast path.)
        if self._frustration_data is not None and self.alpha is None:
            if aa_freq is not None:
                raise NotImplementedError(
                    "Custom aa_freq is not supported by the fast (numba/cuda) configurational "
                    "backend; rebuild the model with backend='numpy' to use a custom aa_freq.")
            fmod = self._get_fast_module()
            return fmod.configurational_frustration(
                self._get_fast_data(), n_decoys=n_decoys, seed=seed, correction=float(correction))
        mean_decoy_energy, std_decoy_energy = self.compute_configurational_decoy_statistics(n_decoys=n_decoys,aa_freq=aa_freq)
        return -(self.compute_configurational_energies()-mean_decoy_energy)/(std_decoy_energy+correction)

    def _static_context_frustration(self, active_residues, kind, aa_freq, correction, dense,
                                    charge_coords=None, charges=None):
        """Frustration of the active residues with the rest held as static context.

        Folds the static context into the active fields/offset, then runs the standard
        sparse frustration machinery on the reduced model. For "singleresidue" and the
        active-active pairs this equals restricting the full-model frustration to the
        active set (single-/pair-residue decoys hold every other residue at native, which
        is exactly what the fold encodes), using the full-protein amino-acid frequencies.
        """
        if kind not in ('singleresidue', 'mutational', 'contact'):
            raise NotImplementedError(
                f"active_residues frustration supports kind in "
                f"('singleresidue', 'mutational', 'contact'), got {kind!r}.")
        reduced, _offset = self.fold_static_context(
            active_residues, charge_coords=charge_coords, charges=charges)
        active = np.asarray(active_residues)
        if active.dtype == bool:
            active = np.where(active)[0]
        active = np.sort(active)

        sub = Frustratometer()
        sub.sparse_potts_model = reduced
        sub._potts_model = {'h': reduced['h'], 'J': None}
        sub.sequence = ''.join(self.sequence[i] for i in active)
        sub.N = len(active)
        sub.mask = None
        sub._elec_data = None
        sub._decoy_fluctuation = {}
        sub.aa_freq = self.aa_freq            # full-protein frequencies (incl. context)
        sub.contact_freq = self.contact_freq
        sub.minimally_frustrated_threshold = self.minimally_frustrated_threshold
        return sub.frustration(kind=kind, aa_freq=aa_freq, correction=correction, dense=dense)

    def frustration(self, sequence=None, kind='singleresidue', mask=None, aa_freq=None, correction=0, dense=True, seed=42,
                    active_residues=None, active_selection=None, static_selection=None,
                    charge_coords=None, charges=None):
        """Frustration index for the native sequence.

        For the pair kinds ("mutational," "pseudoconfigurational," "contact") on a sparse
        model, ``dense=False`` returns a :class:`SparseMatrix` (``row``=contact_i,
        ``col``=contact_j, ``data``=frustration values, ``shape``=L) instead of the (L, L)
        matrix, avoiding the full-matrix allocation for large proteins. Expand it with
        ``.to_dense(fill=0.0)``. ``dense`` is ignored for "singleresidue" and "configurational".

        Static context: pass exactly one of ``active_residues`` (index array / boolean mask
        over ``range(N)``), ``active_selection`` (a molselect string, e.g. ``'chain A'``),
        or ``static_selection`` (a molselect string; the active set is its complement) to
        measure frustration only on the active residues while the rest are held fixed as a
        static context (their couplings fold into the active fields). Returns values over
        the active set, in ascending residue order. ``charge_coords`` (M, 3) / ``charges``
        (M,) add an external static charge field (e.g. DNA phosphates) on the active residues.
        """
        if (active_residues is not None or active_selection is not None
                or static_selection is not None or charge_coords is not None):
            active = self._resolve_active(active_residues, active_selection, static_selection) \
                if (active_residues is not None or active_selection is not None or static_selection is not None) \
                else np.arange(self.N)
            return self._static_context_frustration(active, kind, aa_freq, correction, dense,
                                                    charge_coords=charge_coords, charges=charges)
        if self._frustration_data is not None and (sequence is None or sequence == self.sequence):
            fmod = self._get_fast_module()
            data = self._get_fast_data()
            if kind == 'singleresidue':
                return fmod.singleresidue_frustration(data, correction=float(correction))
            elif kind == 'mutational':
                if not dense:
                    fd = self._frustration_data
                    values = fmod.mutational_frustration(data, correction=float(correction))
                    return SparseMatrix(fd.contact_i, fd.contact_j, data=values, shape=fd.L)
                return fmod.mutational_frustration_dense(data, correction=float(correction))
            elif kind == 'configurational':
                return self.configurational_frustration(aa_freq=aa_freq, correction=correction, seed=seed)
        return super().frustration(sequence=sequence, kind=kind, mask=mask, aa_freq=aa_freq, correction=correction, seed=seed, dense=dense)

    def native_energy(self, sequence=None, ignore_couplings_of_gaps=False, ignore_fields_of_gaps=False):
        if self._frustration_data is not None and (sequence is None or sequence == self.sequence):
            if self._native_energy is None:
                fmod = self._get_fast_module()
                self._native_energy = fmod.native_energy(self._get_fast_data())
            return self._native_energy
        return super().native_energy(sequence=sequence, ignore_couplings_of_gaps=ignore_couplings_of_gaps, ignore_fields_of_gaps=ignore_fields_of_gaps)

    def zim_energy(self) -> float:
        """Membrane Zim (insertion) potential energy for the native sequence,
        in kJ/mol. Sign matches the contribution added to ``h`` (opposite of
        :meth:`fields_energy`); ``fields_energy() + zim_energy()`` is the
        burial-only contribution. Returns 0.0 outside membrane mode."""
        if self._zim_h is None:
            return 0.0
        seq_index = np.array([_AA_21.index(aa) for aa in self.sequence])
        return float(self._zim_h[np.arange(self.N), seq_index].sum())
