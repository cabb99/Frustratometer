"""The numpy reference engine.

Evaluates every query from the Potts model with numpy, handling both the sparse and dense
representations and the electrostatics sidecar. This is where the dense/sparse dispatch lives now
(it used to be scattered across ``Frustratometer``); the dense branch is removed once the codebase
is sparse-canonical.
"""
from .. import frustration
from .base import Backend

__all__ = ['NumpyEngine']


class NumpyEngine(Backend):
    name = 'numpy'
    supports_configurational = False  # AWSEM provides its own configurational; DCA has none

    def native_energy(self, fm, sequence, ignore_couplings_of_gaps=False, ignore_fields_of_gaps=False):
        if fm._is_sparse:
            energy = frustration.compute_native_energy_sparse(
                sequence, fm.sparse_potts_model, ignore_couplings_of_gaps, ignore_fields_of_gaps)
            if getattr(fm, '_elec_data', None) is not None:
                energy += fm._compute_native_energy_elec(sequence, fm._elec_data)
            return energy
        return frustration.compute_native_energy(
            sequence, fm.potts_model, fm.mask, ignore_couplings_of_gaps, ignore_fields_of_gaps)

    def couplings_energy(self, fm, sequence, ignore_couplings_of_gaps=False):
        if fm._is_sparse:
            couplings_energy = frustration.compute_couplings_energy_sparse(
                sequence, fm.sparse_potts_model, ignore_couplings_of_gaps)
            if getattr(fm, '_elec_data', None) is not None:
                couplings_energy += fm._compute_native_energy_elec(sequence, fm._elec_data)
            return couplings_energy
        return frustration.compute_couplings_energy(
            sequence, fm.potts_model, fm.mask, ignore_couplings_of_gaps)

    def sequences_energies(self, fm, sequences, split_couplings_and_fields=False):
        if fm._is_sparse:
            return frustration.compute_sequences_energy_sparse(
                sequences, fm.sparse_potts_model, split_couplings_and_fields)
        return frustration.compute_sequences_energy(
            sequences, fm.potts_model, fm.mask, split_couplings_and_fields)

    def decoy_fluctuation(self, fm, sequence, kind, mask):
        if fm._is_sparse:
            _elec_data = getattr(fm, '_elec_data', None)
            if kind == 'singleresidue':
                fluctuation = frustration.compute_singleresidue_decoy_energy_fluctuation_sparse(
                    sequence, fm.sparse_potts_model)
                if _elec_data is not None:
                    fluctuation = frustration.apply_elec_correction_singleresidue(fluctuation, _elec_data)
            elif kind == 'mutational':
                fluctuation = frustration.compute_mutational_decoy_energy_fluctuation_sparse(
                    sequence, fm.sparse_potts_model)
                if _elec_data is not None:
                    fluctuation = frustration.apply_elec_correction_mutational(
                        fluctuation, fm.sparse_potts_model, _elec_data)
            elif kind == 'pseudoconfigurational':
                from ..classes.Structure import SparseMatrix as _SM
                if isinstance(fm.mask, _SM):
                    if fm.distance_cutoff is None:
                        _mask_mean = frustration.mask_mean(fm.mask.shape, fm.sequence_cutoff, fm.chain_breaks)
                    else:
                        _mask_mean = float(len(fm.mask)) / (fm.mask.shape * fm.mask.shape)
                else:
                    _mask_mean = float(fm.mask.mean())
                fluctuation = frustration.compute_pseudoconfigurational_decoy_energy_fluctuation_sparse(
                    sequence, fm.sparse_potts_model, _mask_mean)
                if _elec_data is not None:
                    fluctuation = frustration.apply_elec_correction_pseudoconfigurational(
                        fluctuation, fm.sparse_potts_model, _elec_data)
            elif kind == 'contact':
                fluctuation = frustration.compute_contact_decoy_energy_fluctuation_sparse(
                    sequence, fm.sparse_potts_model)
                if _elec_data is not None:
                    fluctuation = frustration.apply_elec_correction_contact(
                        fluctuation, fm.sparse_potts_model, _elec_data)
            else:
                raise Exception("Wrong kind of decoy generation selected")
        else:
            if kind == 'singleresidue':
                fluctuation = frustration.compute_singleresidue_decoy_energy_fluctuation(
                    sequence, fm.potts_model, mask)
            elif kind == 'mutational':
                fluctuation = frustration.compute_mutational_decoy_energy_fluctuation(
                    sequence, fm.potts_model, mask)
            elif kind == 'pseudoconfigurational':
                fluctuation = frustration.compute_pseudoconfigurational_decoy_energy_fluctuation(
                    sequence, fm.potts_model, mask)
            elif kind == 'contact':
                fluctuation = frustration.compute_contact_decoy_energy_fluctuation(
                    sequence, fm.potts_model, mask)
            else:
                raise Exception("Wrong kind of decoy generation selected")
        return fluctuation
