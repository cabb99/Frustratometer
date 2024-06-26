"""PDB functions"""

from .pdb import *
try:
    from .fix import repair_pdb
except ImportError as e:
    error_message=str(e)
    if 'pdbfixer' in str(e):
        def repair_pdb(*args, **kwargs):
            warn_pdbfixer_not_installed()
            raise ImportError("openmm and pdbfixer must be installed to use the repair_pdb function.")

        def warn_pdbfixer_not_installed():
            import warnings
            warnings.warn(
                "pdbfixer or openmm are not installed but are needed for this function.\n"
                "To install pdbfixer and openmm, please run the following command:\n"
                "conda install -c conda-forge openmm pdbfixer",
                ImportWarning
            )
    else:
        raise e
    
__all__ = ['download', 'get_sequence', 'get_distance_matrix', 'full_to_filtered_aligned_mapping', 'repair_pdb']