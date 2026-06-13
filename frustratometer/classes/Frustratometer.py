"""Provide the primary functions."""
from .. import pdb
from .. import dca

import numpy as np
import scipy.spatial.distance as sdist
from typing import Union
from pathlib import Path
#Import other modules
from .. import frustration
import logging

__all__=['PottsModel']
##################
# PFAM functions #
##################


# Class wrapper
class Frustratometer:

    # @property
    # def sequence_cutoff(self):
    #     return self._sequence_cutoff

    # @sequence_cutoff.setter
    # def sequence_cutoff(self, value):
    #     self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
    #     self._sequence_cutoff = value
    #     self._native_energy = None
    #     self._decoy_fluctuation = {}

    # @property
    # def distance_cutoff(self):
    #     return self._distance_cutoff

    # @distance_cutoff.setter
    # def distance_cutoff(self, value):
    #     self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
    #     self._distance_cutoff = value
    #     self._native_energy = None
    #     self._decoy_fluctuation = {}

    def _ensure_potts_model(self):
        """Hook for subclasses to lazily materialize a (sparse) Potts model.

        No-op by default. AWSEM fast-mode models override this to build
        ``sparse_potts_model`` on first use, so inherited methods that need fields
        and couplings work without a Potts model being built up front.
        """
        pass

    def _ensure_dense_potts_model(self):
        """Reconstruct dense J from sparse potts model if needed."""
        self._ensure_potts_model()
        if self._potts_model.get('J') is None and getattr(self, 'sparse_potts_model', None) is not None:
            self._potts_model['J'] = frustration.potts_model_sparse_to_dense(self.sparse_potts_model)['J']

    @property
    def _is_sparse(self):
        """True when a sparse Potts model is available (materializing it if deferred)."""
        self._ensure_potts_model()
        return getattr(self, 'sparse_potts_model', None) is not None

    def _compute_native_energy_elec(self, sequence, elec_data):
        """Dispatch electrostatic native energy to sparse or dense version."""
        if elec_data.get('indicator') is None:
            return frustration.compute_native_energy_elec_sparse(sequence, elec_data)
        return frustration.compute_native_energy_elec(sequence, elec_data, self.mask)

    @property
    def potts_model(self):
        """Access the Potts model dict with auto-reconstruction of J from sparse if needed."""
        self._ensure_dense_potts_model()
        return self._potts_model

    @potts_model.setter
    def potts_model(self, value):
        self._potts_model = value

    def native_energy(self,sequence:str = None,ignore_couplings_of_gaps:bool=False,ignore_fields_of_gaps:bool = False) -> float:
        """
        Calculates the native energy of the protein sequence.

        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. If no sequence is provided as input, the original protein sequence of the protein structure object is used for the energy calculation.
        ignore_couplings_of_gaps: bool
            If set to True, the couplings terms of any gaps in the protein sequence are ignored in energy calculations.
        ignore_fields_of_gaps: bool
            If set to True, the fields terms of any gaps in the protein sequence are ignored in energy calculations.
        Returns
        -------
        energy_value : float
            Native energy of sequence
        """
        if sequence is None:
            sequence=self.sequence
        else:
            if self._is_sparse:
                energy = frustration.compute_native_energy_sparse(sequence, self.sparse_potts_model, ignore_couplings_of_gaps, ignore_fields_of_gaps)
                if getattr(self, '_elec_data', None) is not None:
                    energy += self._compute_native_energy_elec(sequence, self._elec_data)
                return energy
            return frustration.compute_native_energy(sequence, self.potts_model, self.mask,ignore_couplings_of_gaps,ignore_fields_of_gaps)
        if not self._native_energy:
            if self._is_sparse:
                self._native_energy = frustration.compute_native_energy_sparse(sequence, self.sparse_potts_model, ignore_couplings_of_gaps, ignore_fields_of_gaps)
                if getattr(self, '_elec_data', None) is not None:
                    self._native_energy += self._compute_native_energy_elec(sequence, self._elec_data)
            else:
                self._native_energy=frustration.compute_native_energy(sequence, self.potts_model, self.mask,ignore_couplings_of_gaps,ignore_fields_of_gaps)
        energy_value=self._native_energy
        return energy_value

    def sequences_energies(self, sequences:np.array, split_couplings_and_fields:bool = False):
        """
        Computes the energy of multiple protein sequences.

        .. math::
            E = \\sum_i h_i + \\frac{1}{2} \\sum_{i,j} J_{ij} \\Theta_{ij}

        Parameters
        ----------
        sequences : list
            List of amino acid sequences in string format, separated by commas. The sequences are assumed to be in one-letter code. Gaps are represented as '-'. The length of each sequence (L) should all match the dimensions of the Potts model.
        split_couplings_and_fields : bool
            If True, two lists of the sequences' couplings and fields energies are returned.
            Default is False.

        Returns
        -------
        output (if split_couplings_and_fields==False): float
            The computed energies of the protein sequences
        output (if split_couplings_and_fields==True): np.array
            Array containing computed fields and couplings energies of the protein sequences. 
        """
        if self._is_sparse:
            output=frustration.compute_sequences_energy_sparse(sequences, self.sparse_potts_model, split_couplings_and_fields)
        else:
            output=frustration.compute_sequences_energy(sequences, self.potts_model, self.mask, split_couplings_and_fields)
        return output

    def fields_energy(self, sequence:str = None, ignore_fields_of_gaps:bool = False) -> float:
        """
        Computes the fields energy of a protein sequence.
        
        .. math::
            E = \\sum_i h_i
            
        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. If no sequence is provided as input, the original protein sequence of the protein structure object is used for the energy calculation.
        ignore_fields_of_gaps : bool
            If True, fields corresponding to gaps ('-') in the sequence are set to 0 in the energy calculation.
            Default is False.

        Returns
        -------
        fields_energy : float
            The computed fields energy of the protein sequence.
        """
        if sequence is None:
            sequence=self.sequence
        fields_energy=frustration.compute_fields_energy(sequence, self._potts_model,ignore_fields_of_gaps)
        return fields_energy

    def couplings_energy(self, sequence:str = None,ignore_couplings_of_gaps:bool = False) -> float:
        """
        Computes the couplings energy of a protein sequence based on a given Potts model and an interaction mask.
        
        .. math::
            E = \\frac{1}{2} \\sum_{i,j} J_{ij} \\Theta_{ij}
            
        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. If no sequence is provided as input, the original protein sequence of the protein structure object is used for the energy calculation.
        ignore_couplings_of_gaps : bool
            If True, couplings involving gaps ('-') in the sequence are set to 0 in the energy calculation.
            Default is False.

        Returns
        -------
        couplings_energy : float
            The computed couplings energy of the protein sequence.
        """
        if sequence is None:
            sequence=self.sequence
        if self._is_sparse:
            couplings_energy=frustration.compute_couplings_energy_sparse(sequence, self.sparse_potts_model, ignore_couplings_of_gaps)
            if getattr(self, '_elec_data', None) is not None:
                couplings_energy += self._compute_native_energy_elec(sequence, self._elec_data)
        else:
            couplings_energy=frustration.compute_couplings_energy(sequence, self.potts_model, self.mask,ignore_couplings_of_gaps)
        return couplings_energy
        
    def decoy_fluctuation(self, sequence:str = None,kind:str = 'singleresidue',mask:np.array = None) -> np.array:
        """
        Computes a matrix for a sequence of length L that describes all possible changes in energy upon mutating a single or pair of residues (depending on "kind" entry used) simultaneously.
            
        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. If no sequence is provided as input, the original protein sequence of the protein structure object is used for the energy calculation.
        kind : str
            Kind of decoys generated. Options: "singleresidue," "mutational," "pseudoconfigurational," and "contact." 
        mask : np.array
            A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.

        Returns
        -------
        fluctuation : np.array
            The computed couplings energy of the protein sequence.
        """
        is_native = sequence is None or sequence == self.sequence
        if sequence is None:
            sequence=self.sequence
        if is_native and kind in self._decoy_fluctuation:
            return self._decoy_fluctuation[kind]
        if not isinstance(mask, np.ndarray):
            mask=self.mask
        # Use sparse path when available
        if self._is_sparse:
            _elec_data = getattr(self, '_elec_data', None)
            if kind == 'singleresidue':
                fluctuation = frustration.compute_singleresidue_decoy_energy_fluctuation_sparse(sequence, self.sparse_potts_model)
                if _elec_data is not None:
                    fluctuation = frustration.apply_elec_correction_singleresidue(fluctuation, _elec_data)
            elif kind == 'mutational':
                fluctuation = frustration.compute_mutational_decoy_energy_fluctuation_sparse(sequence, self.sparse_potts_model)
                if _elec_data is not None:
                    fluctuation = frustration.apply_elec_correction_mutational(fluctuation, self.sparse_potts_model, _elec_data)
            elif kind == 'pseudoconfigurational':
                from .Structure import SparseMatrix as _SM
                if isinstance(self.mask, _SM):
                    if self.distance_cutoff is None:
                        _mask_mean = frustration.mask_mean(self.mask.shape, self.sequence_cutoff, self.chain_breaks)
                    else:
                        _mask_mean = float(len(self.mask)) / (self.mask.shape * self.mask.shape)
                else:
                    _mask_mean = float(self.mask.mean())
                fluctuation = frustration.compute_pseudoconfigurational_decoy_energy_fluctuation_sparse(sequence, self.sparse_potts_model, _mask_mean)
                if _elec_data is not None:
                    fluctuation = frustration.apply_elec_correction_pseudoconfigurational(fluctuation, self.sparse_potts_model, _elec_data)
            elif kind == 'contact':
                fluctuation = frustration.compute_contact_decoy_energy_fluctuation_sparse(sequence, self.sparse_potts_model)
                if _elec_data is not None:
                    fluctuation = frustration.apply_elec_correction_contact(fluctuation, self.sparse_potts_model, _elec_data)
            else:
                raise Exception("Wrong kind of decoy generation selected")
        else:
            if kind == 'singleresidue':
                fluctuation = frustration.compute_singleresidue_decoy_energy_fluctuation(sequence, self.potts_model, mask)
            elif kind == 'mutational':
                fluctuation = frustration.compute_mutational_decoy_energy_fluctuation(sequence, self.potts_model, mask)
            elif kind == 'pseudoconfigurational':
                fluctuation = frustration.compute_pseudoconfigurational_decoy_energy_fluctuation(sequence, self.potts_model, mask)
            elif kind == 'contact':
                fluctuation = frustration.compute_contact_decoy_energy_fluctuation(sequence, self.potts_model, mask)
            else:
                raise Exception("Wrong kind of decoy generation selected")
        if is_native:
            self._decoy_fluctuation[kind] = fluctuation
        return fluctuation

    def decoy_energy(self, kind:str = 'singleresidue',sequence: str =None) ->np.array:
        """
        Computes all possible decoy energies.
        
        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
        kind : str
            Kind of decoys generated. Options: "singleresidue," "mutational," "pseudoconfigurational," and "contact." 

        Returns
        -------
        decoy_energy: np.array
            Matrix describing all possible decoy energies.
        """
        if sequence is None:
            sequence=self.sequence
        decoy_energy=self.native_energy(sequence=sequence) + self.decoy_fluctuation(kind=kind,sequence=sequence)
        return decoy_energy

    def scores(self):
        """
        Computes accuracy of DCA predicted contacts by calculating contact scores based on the Frobenius norm.

        Returns
        -------
        corr_norm : np.array
            Contact score matrix (N x N)
        """
        self._ensure_dense_potts_model()
        return frustration.compute_scores(self.potts_model)

    def frustration(self, sequence:str = None, kind:str = 'singleresidue', mask:np.array = None, aa_freq:np.array = None, correction:int = 0, seed:int = 42, dense:bool = True) -> np.array:
        """
        Calculates frustration index values.

        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
        kind : str
            Kind of decoys generated. Options: "singleresidue," "mutational," "pseudoconfigurational," "configurational," and "contact."
        mask : np.array
            A 2D Boolean array that determines which residue pairs should be considered in the energy computation. The mask should have dimensions (L, L), where L is the length of the sequence.
        aa_freq: np.array
            Array of frequencies of all 21 possible amino acids within sequence
        dense : bool
            For the pair kinds ("mutational," "pseudoconfigurational," "contact") on a sparse
            model, ``dense=False`` returns a :class:`SparseMatrix` of per-contact values
            (``row``=contact_i, ``col``=contact_j, ``data``=values, ``shape``=L) instead of the
            full (L, L) matrix, avoiding the dense allocation for large proteins. Expand it with
            ``.to_dense(fill=0.0)``. Ignored for "singleresidue" and for dense models, which always
            return the (L,) / (L, L) array.

        Returns
        -------
        frustration_values: np.array or SparseMatrix
            Frustration index values.
        """
        if sequence is None:
            sequence=self.sequence
        if not isinstance(mask, np.ndarray):
            mask=self.mask
        # Configurational frustration has its own dedicated method (e.g. in AWSEM);
        # dispatch before computing decoy_fluctuation which doesn't handle 'configurational'.
        if kind == 'configurational':
            if 'configurational_frustration' in dir(self):
                #TODO: Correct this function for different aa_freq than WT
                return self.configurational_frustration(aa_freq, correction, seed=seed)
            raise ValueError(
                "kind='configurational' (Monte-Carlo configurational frustration) is only "
                "available on models that implement configurational_frustration (e.g. AWSEM). "
                "For this model use kind='pseudoconfigurational' (analytic approximation)."
            )
        decoy_fluctuation = self.decoy_fluctuation(sequence=sequence,kind=kind, mask=mask)
        if kind == 'singleresidue':
            if aa_freq is None:
                aa_freq = self.aa_freq
            frustration_values=frustration.compute_single_frustration(decoy_fluctuation, aa_freq, correction)
            return frustration_values
        elif kind in ['mutational', 'pseudoconfigurational', 'contact']:
            if aa_freq is None:
                aa_freq = self.contact_freq
            # Sparse decoy fluctuation has shape (N_contacts, 21, 21) — use sparse pair frustration.
            _use_sparse = self._is_sparse
            if _use_sparse and decoy_fluctuation.ndim == 3:
                sparse_frust = frustration.compute_pair_frustration_sparse(decoy_fluctuation, aa_freq, correction)
                contact_i = self.sparse_potts_model['contact_i']
                contact_j = self.sparse_potts_model['contact_j']
                L = self.sparse_potts_model['L']
                if not dense:
                    from .Structure import SparseMatrix as _SM
                    return _SM(contact_i, contact_j, data=sparse_frust, shape=L)
                frustration_values = frustration.sparse_frustration_to_dense(
                    sparse_frust, contact_i, contact_j, L,
                )
            else:
                frustration_values=frustration.compute_pair_frustration(decoy_fluctuation, aa_freq, correction)
            return frustration_values

    def plot_decoy_energy(self, sequence:str = None, kind:str = 'singleresidue', method:str = 'clustermap'):
        """
        Plot comparison of single residue decoy energies, relative to the native energy

        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
        kind : str
            Kind of decoys generated. Options: "singleresidue," "mutational," "pseudoconfigurational," and "contact." 
        method : str
            Options: "clustermap", "heatmap"
        """
        if sequence is None:
            sequence=self.sequence
        native_energy = self.native_energy(sequence=sequence)
        decoy_energy = self.decoy_energy(kind=kind,sequence=sequence)
        if kind == 'singleresidue':
            g = frustration.plot_singleresidue_decoy_energy(decoy_energy, native_energy, method)
            return g

    def roc(self):
        """
        Computes Receiver Operating Characteristic (ROC) curve of 
        contacts predicted by DCA and true contacts, as identified from the distance matrix.
        """
        return frustration.compute_roc(self.scores(), self.distance_matrix, self.distance_cutoff)

    def plot_roc(self):
        """
        Plots the curve of the receiver-operating characteristic.
        """
        frustration.plot_roc(self.roc())

    def auc(self):
        """
        Computes area under the curve of the receiver-operating characteristic.
        A higher AUC value (maximum=1) indicates that the TPR is always high, regardless of FPR. 
        """
        return frustration.compute_auc(self.roc())

    def vmd(self, sequence: str = None, single:Union[str,np.array] = 'singleresidue', pair:Union[str,np.array] = 'mutational',
             aa_freq:np.array = None, correction:int = 0, max_connections:Union[int,None] = None, movie_name=None, still_image_name=None):
        """
        Calculates frustration indices and superimposes frustration patterns onto PDB structure using the VMD software.

        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
        pair : str
            Kind of pair frustration calculated. Options: "mutational," "pseudoconfigurational," "configurational," and "contact." 
        aa_freq: np.array
            Array of frequencies of all 21 possible amino acids within sequence
        max_connections : int
            Maximum number of pair frustration values visualized in tcl file
        movie_name : Path or str
            Output tcl script file with rotating structure
        """

        if sequence is None:
            sequence=self.sequence
        elif sequence.strip() != self.sequence.strip(): 
            logging.warning("The value of the self.sequence property of your Frustratometer object differs\n\
                    from the sequence that was passed to this vmd function. Proceeding further may not\n\
                    perform the computation that you intend to perform.")
        

        from .Structure import SparseMatrix as _SM
        want_sparse = isinstance(self.mask, _SM)
        single_frustration = -self.frustration(kind=single, sequence=sequence, aa_freq=aa_freq)
        pair_frustration = self.frustration(kind=pair, sequence=sequence, aa_freq=aa_freq, dense=not want_sparse)
        if isinstance(pair_frustration, _SM):
            pair_frustration = _SM(pair_frustration.row, pair_frustration.col,
                                   data=-np.asarray(pair_frustration.data), shape=pair_frustration.shape)
        else:
            pair_frustration = -pair_frustration

        tcl_script = frustration.write_tcl_script(self.pdb_file, self.chain, self.mask, self.distance_matrix, self.distance_cutoff,
                                      single_frustration, pair_frustration,
                                      max_connections=max_connections, movie_name=movie_name, still_image_name=still_image_name)
        frustration.call_vmd(self.pdb_file, tcl_script)

    def view_pair_frustration(self, sequence:str = None, pair:str = 'mutational', aa_freq:np.array = None):
        """
        Calculates pair frustration indices and superimposes frustration patterns onto PDB structure, using Pymol
        for local visualization.

        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
        pair : str
            Kind of pair frustration calculated. Options: "mutational," "pseudoconfigurational," "configurational," and "contact." 
        aa_freq: np.array
            Array of frequencies of all 21 possible amino acids within sequence
        """
        import py3Dmol

        if sequence is None:
            sequence=self.sequence
        
        pdb_filename = self.pdb_file
        shift=self.init_index_shift+1

        from .Structure import SparseMatrix as _SM
        if isinstance(self.mask, _SM):
            sm = self.frustration(sequence=sequence, kind=pair, dense=False)
            if isinstance(sm, _SM):
                # Sparse-native: take the upper-triangle contacts directly (no (L, L) matrix).
                up = np.asarray(sm.row) < np.asarray(sm.col)
                sel_frustration = np.array([np.asarray(sm.row)[up],
                                            np.asarray(sm.col)[up],
                                            np.asarray(sm.data)[up]]).T
            else:
                # e.g. configurational returns a dense (L, L)
                pair_frustration = sm * np.triu(self.mask.to_dense(fill=0.0))
                residues = np.arange(len(sequence))
                r1, r2 = np.meshgrid(residues, residues, indexing='ij')
                sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel()]).T
        else:
            pair_frustration=self.frustration(sequence=sequence, kind=pair)*np.triu(self.mask)
            residues=np.arange(len(sequence))
            r1, r2 = np.meshgrid(residues, residues, indexing='ij')
            sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel()]).T
        minimally_frustrated = sel_frustration[sel_frustration[:, -1] > 1]
        frustrated = sel_frustration[sel_frustration[:, -1] < -self.minimally_frustrated_threshold]
        
        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
        view.addModel(open(pdb_filename,'r').read(),'pdb')

        view.setBackgroundColor('white')
        view.setStyle({'cartoon':{'color':'white'}})
        
        for i,j,f in frustrated:
            view.addLine({'start':{'chain':self.chain,'resi':[str(i+shift)]},'end':{'chain':self.chain,'resi':[str(j+shift)]},
                        'color':'red', 'dashed':False,'linewidth':3})
        
        for i,j,f in minimally_frustrated:
            view.addLine({'start':{'chain':self.chain,'resi':[str(i+shift)]},'end':{'chain':self.chain,'resi':[str(j+shift)]},
                        'color':'green', 'dashed':False,'linewidth':3})

        view.zoomTo(viewer=(0,0))

        return view

    def view_single_frustration(self,  aa_freq:np.array = None, only_frustrated_contacts:bool=False):
        """
        Calculates single residue frustration indices and superimposes frustration patterns onto PDB structure, using Pymol
        for local visualization.

        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
        aa_freq : np.array
            Array of frequencies of all 21 possible amino acids within sequence
        only_frustrated_contacts : bool
            If set to True, minimally frustrated contacts are also hilighted.
        """
        import py3Dmol
        pdb_filename = self.pdb_file
        shift=self.init_index_shift+1
        single_frustration=self.frustration(kind="singleresidue")
        residues=np.arange(len(self.sequence))

        sel_frustration = np.array([residues,single_frustration]).T
        minimally_frustrated = sel_frustration[sel_frustration[:, -1] > self.minimally_frustrated_threshold]
        frustrated = sel_frustration[sel_frustration[:, -1] < -1]
        
        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
        view.addModel(open(pdb_filename,'r').read(),'pdb')

        view.setBackgroundColor('white')
        view.setStyle({'cartoon':{'color':'white'}})
        
        for i,f in frustrated:
            view.setStyle({'model': -1, 'resi': i+shift}, {"cartoon": {'color': 'red'}})
            # view.addLine({'start':{'chain':self.chain,'resi':[str(i+shift)]},'end':{'chain':self.chain,'resi':[str(j+shift)]},
            #             'color':'red', 'dashed':False,'linewidth':3})

        if only_frustrated_contacts==False:
            for i,f in minimally_frustrated:
                view.setStyle({'model': -1, 'resi': i+shift}, {"cartoon": {'color': 'green'}})
                # view.addLine({'start':{'chain':self.chain,'resi':[str(i+shift)]},'end':{'chain':self.chain,'resi':[str(j+shift)]},
                #             'color':'green', 'dashed':False,'linewidth':3})

        view.zoomTo(viewer=(0,0))

        return view

    def generate_frustration_pair_distribution(self,sequence: str =None, kind:str ="singleresidue",bins: int =30,maximum_shell_radius: int=20):
        """
        Calculates frustration pair distributions. This helps identify spatial proximity of similarly frustrated residues or contacts from one another.
        
        For mutational, pseudoconfigurational, configurational, and contact frustration pair distributions, the distances between midpoints of Cb-Cb (or Ca in the case of glycine) 
        atom pairs are measured. 
        For single residue frustration, the distances of Cb (or Ca in the case of glycine) atoms are measured.  

        Parameters
        ----------
        sequence : str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. Gaps are represented as '-'. The length of the sequence (L) should match the dimensions of the Potts model.
        aa_freq : np.array
            Array of frequencies of all 21 possible amino acids within sequence
        kind : str
            Kind of decoys generated. Options: "singleresidue," "mutational," "pseudoconfigurational," "configurational," and "contact." 
        bins : int 
            Number of bins
        maximum_shell_radius : int
            Maximum shell radius to evaluate

        Returns
        ----------
        minimally_frustrated_gr : np.array
            Pair distribution function of minimally frustrated contacts
        frustrated_gr : np.array
            Pair distribution function of frustrated contacts
        neutral_gr : np.array
            Pair distribution function of neutral contacts
        r_m : np.array
            Array of midpoints between evaluated spherical shells
        """
        if sequence==None:
            sequence=self.sequence
        frustration_values=self.frustration(sequence=sequence,kind=kind)
        
        residue_cb_coordinates=(self.structure.select('(protein and (name CB) or (resname GLY and name CA))').getCoords())
        
        if "-" in sequence:
            original_residue_cb_coordinates=residue_cb_coordinates
            mapped_residues=list(self.structure.full_to_aligned_index_dict.values())
            residue_cb_coordinates=original_residue_cb_coordinates[mapped_residues,:]

        if kind=="singleresidue":
            sel_frustration = np.column_stack((residue_cb_coordinates,np.expand_dims(frustration_values, axis=1)))
            
        elif kind in ["configurational","pseudoconfigurational","mutational"]:
            i,j=np.meshgrid(range(0,len(self.sequence)),range(0,len(self.sequence)))
            midpoint_coordinates=(residue_cb_coordinates[i.flatten(),:]+ residue_cb_coordinates[j.flatten(),:])/2
            sel_frustration = np.column_stack((midpoint_coordinates, frustration_values.ravel()))

        r=np.linspace(1,maximum_shell_radius,bins)
        r_m=(r[1:]+r[:-1])/2

        maximum_shell_volume=4/3 * np.pi * (maximum_shell_radius**3)
        shell_vol = 4 * np.pi * (r[1:]-r[:-1]) * (r[1:]**2)
        ###
        #Calculate contact number densities
        minimally_frustrated_count=len(sel_frustration[sel_frustration[:, -1] > self.minimally_frustrated_threshold])
        frustrated_count=len(sel_frustration[sel_frustration[:, -1] < -1])
        neutral_count=len(sel_frustration[(sel_frustration[:, -1] > -1) & (sel_frustration[:, -1] < self.minimally_frustrated_threshold)])

        minimally_frustrated_density=(minimally_frustrated_count*(minimally_frustrated_count-1))/(2*maximum_shell_volume)
        frustrated_density=(frustrated_count*(frustrated_count-1))/(2*maximum_shell_volume)
        neutral_density=(neutral_count*(neutral_count-1))/(2*maximum_shell_volume)
        ###
        #Calculate relative distances of contacts
        minimally_frustrated_contacts=(sdist.pdist(sel_frustration[sel_frustration[:, -1] > self.minimally_frustrated_threshold][:,:-1]))
        frustrated_contacts=(sdist.pdist(sel_frustration[sel_frustration[:, -1] <-1][:,:-1]))
        neutral_contacts=(sdist.pdist(sel_frustration[(sel_frustration[:, -1] > -1) & (sel_frustration[:, -1] < self.minimally_frustrated_threshold)][:,:-1]))
        ###
        #Calculate contact histograms
        minimally_frustrated_hist,_ = np.histogram(minimally_frustrated_contacts,bins=r)
        minimally_frustrated_gr=np.divide(minimally_frustrated_hist,(shell_vol*minimally_frustrated_density))

        frustrated_hist,_= np.histogram(frustrated_contacts,bins=r)
        frustrated_gr=np.divide(frustrated_hist,(shell_vol*frustrated_density))

        neutral_hist,_=np.histogram(neutral_contacts,bins=r)
        neutral_gr=np.divide(neutral_hist,(shell_vol*neutral_density))
        
        return minimally_frustrated_gr,frustrated_gr,neutral_gr,r_m
   

    def total_frustration(self,
                      n_decoys: int = 1000,
                      config_decoys: bool = False,
                      msa_mask: Union[int, np.array] = 1,
                      fragment_pos: Union[None, np.array] = None,
                      fragment_in_context: bool = False,
                      output_kind: str = 'frustration') -> Union[float, np.array] :

        """
        Calculates the total frustration of a protein fragment.

        Parameters
        ----------
        n_decoys: int
            Number of sequence decoys to create
        config_decoys: bool
            If True, use the pseudoconfigurational decoys approximation, shuffling index positions for pseudoconfigurational decoys energy calculation. If False, mutational decoys.
        msa_mask: np.array
            Extra mask to use a Multiple Sequence Alignment that do not cover completely the reference PDB
        fragment_pos: np.array
            Fragment positions. If None, use the complete model
        fragment_in_context: bool
            If True, the energetics calculations take into account the interactions between the fragment and other sequence positions
        output_kind: str
            If 'frustration', returns frustration. If not, returns native energy, decoy energy average and decoy energy standard deviation.
        Return
        -------
        total_frustration : float
            Total frustration of the fragment or complete protein
        native_energy: float
            Native energy of the given sequence
        decoy_energy_average: float
            Average of the decoy energy distribution
        decoy_energy_std: float
            Standard deviation of the decoy energy distribution
        """

        from .Structure import SparseMatrix as _SM
        simple = (not config_decoys and not fragment_in_context and fragment_pos is None
                  and np.isscalar(msa_mask) and msa_mask == 1)
        if self._is_sparse and simple:
            # Sparse-native: full protein, mutational decoys — no dense (L, L, Q, Q) / (N, L, L).
            return frustration.compute_total_frustration_sparse(
                self.sequence, self.sparse_potts_model, n_decoys, output_kind)

        # Dense path (fragments / pseudoconfigurational decoys / MSA mask). Densify a sparse
        # mask so compute_fragment_mask can index it.
        mask = self.mask.to_dense(fill=0.0) if isinstance(self.mask, _SM) else self.mask
        return frustration.compute_total_frustration(self.sequence,
                                                     self.potts_model,
                                                     mask,
                                                     n_decoys,
                                                     config_decoys,
                                                     msa_mask,
                                                     fragment_pos,
                                                     fragment_in_context,
                                                     output_kind)


    def sliding_window(self,
                       win_size: int = 5,
                       ndecoys: int = 1000,
                       config_decoys: bool = False) -> dict:

        """
        Computes the total frustration, the native energy, the decoy average energy and the decoy standard deviation for fragments on a sliding window

        Parameters
        ----------
        win_size: int
            Size of the sliding window
        ndecoys: int
            Number of decoy sequences to use
        config_decoys: bool
            If True, use the configurational decoys approximation, shuffling index positions for configurational decoys energy calculation. If False, mutational decoys.

        Returns
        -------
        results: dict
            Dictionary with the results, containing
            'fragment_center': center position of each window 
            'win_size': size of the sliding windows
            'native_energy': native energy for each window
            'decoy_energy_av': decoy energy average for each window
            'decoy_energy_std': decoy energy standard deviation for each window
            'frustration': total frustration index for each window

        """

        from .Structure import SparseMatrix as _SM
        if self._is_sparse and not config_decoys:
            # Sparse-native: per-residue field (L,) + per-contact coupling (Nc,), no (N, L, L).
            return frustration.compute_energy_sliding_window_sparse(
                self.sequence, self.sparse_potts_model, win_size, ndecoys)

        # Dense path (pseudoconfigurational decoys); densify a sparse mask for compute_fragment_mask.
        mask = self.mask.to_dense(fill=0.0) if isinstance(self.mask, _SM) else self.mask
        return frustration.compute_energy_sliding_window(self.sequence,
                                                         self.potts_model,
                                                         mask,
                                                         win_size,
                                                         ndecoys,
                                                         config_decoys)
