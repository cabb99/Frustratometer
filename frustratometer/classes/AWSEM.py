import numpy as np
from ..utils import _path
from .. import frustration
from .Frustratometer import Frustratometer
from .Gamma import Gamma
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import Path
from typing import List,Optional,Union,Generator
import os

__all__ = ['AWSEM','AWSEMIndicators','DecoyEnsemble', 'AWSEMVariancePotts']

class Parameters(BaseModel):
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    """Default parameters for AWSEM energy calculations."""
    k_contact: float = Field(4.184, description="""
        Scale factor for contact potential.
        Many parameters used to be given in kcal/mol,
        but we want our results in kJ/mol, so this is
        set to the appropriate conversion factor by default.
        Note that the electrostatic parameter is not scaled
        by k_contact.""")
    
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
    membrane_gamma: Union[Path,Gamma] = Field(_path/'data'/'AWSEM_membrane_2015.json', description="File or Gamma object containing the membrane Gamma values (for membrane proteins)")
    eta_switching: int = Field(10, description="Switching distance for the membrane switching function")

    #Electrostatics
    min_sequence_separation_electrostatics: Optional[int] = Field(1, description="Minimum sequence separation for electrostatics calculation.")
    k_electrostatics: float = Field(17.3636, description="Coefficient for electrostatic interactions. (kJ/mol)")
    electrostatics_screening_length: float = Field(10, description="Screening length for electrostatic interactions. (Angstrom)")
    charges: np.array = Field(np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), description="Charge on each residue type")
    # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    #charges: np.array = Field(np.array([0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), description="Charge on each residue type")
    #['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

class AWSEMBase(Frustratometer):

    def __init__(self, 
                 sequence: str,
                 expose_indicator_functions: bool=False,
                 potts: bool=True,
                 **parameters)->object:
        """
        Generate AWSEM object

        Parameters
        ----------
        sequence: str
            The amino acid sequence
        expose_indicator_functions: bool
            If set to True, indicator functions of the contact and burial energy terms can be accessed by user.
        potts: bool
            Whether to set up the potts model (can be RAM-intensive and time-intensive),
            which is unnecessary if all you want to get is the indicator functions.
        
        Returns
        -------
        AWSEM object
        """

        # set sequence based on argument
        self.N = len(sequence)
        self.sequence = sequence

        # set indicator function exposure based on argument
        #     i guess not exposing indicator functions saves memory?
        self.expose_indicator_functions = expose_indicator_functions

        # whether to compute potts model
        self.potts = potts

        # parse other arguments
        p = Parameters(**parameters)
        if p.min_sequence_separation_contact is None:
            p.min_sequence_separation_contact = 1
        if p.min_sequence_separation_rho is None:
            p.min_sequence_separation_rho = 1
        if p.min_sequence_separation_electrostatics is None:
            p.min_sequence_separation_electrostatics = 1
        for field, value in p:
            setattr(self, field, value)
        self.p = p

        # set gamma
        if isinstance(self.p.gamma, Gamma):
            gamma = self.p.gamma
        elif isinstance(self.p.gamma, Path):
            gamma = Gamma(self.p.gamma)
            self.p.gamma = gamma
        else:
            raise ValueError("Gamma parameter must be a path or a Gamma object.")
        """
        # CARLOS: if you really want to reorder, we can do something like this,
                  but it shouldn't be necessary--we always have access to the
                  order in self.gamma.alphabet
        ordered_alphabet = ['A','C','D','E','F','G','H','I','K','L',
                            'M','N','P','Q','R','S','T','V','W','Y']
        for aa in ordered_alphabet:
            assert aa in gamma.alphabet, f'{aa} missing from gamma.alphabet!'
        if len(gamma.alphabet) == 20: # alphabet is exactly the canonical AAs
            gamma = gamma.reorder(ordered_alphabet)
        elif len(gamma.alphabet) > 20: # includes noncanonical AA(s) (or a "gap")
            ncAA = []
            for aa in gamma.alphabet:
                if aa not in ordered_alphabet:
                    ncAA.append(AA)
            ordered_alphabet = ncAA + ordered_alphabet # insert at the beginning
            gamma = gamma.reorder(ordered_alphabet)
        else:
            raise ValueError(f"gamma file alphabet {gamma.alphabet} was too short")
        """
        #    burial gamma
        self.q = len(gamma.alphabet) # most likely 20, but could be different
        gb = gamma['Burial']
        if gb.shape == (3,self.q):
            self.burial_gamma = gb.T
        elif gb.shape == (self.q,3):
            self.burial_gamma = gb
        else:
            raise ValueError(f"""Don't know how to parse burial gamma with shape {gb.shape}.
            Expected ({self.q},3) or (3,{self.q}).""")
        #     pairwise gamma: squeeze to remove extra axis that is commonly present
        self.direct_gamma = np.squeeze(gamma['Direct']) 
        self.protein_gamma = np.squeeze(gamma['Protein'])
        self.water_gamma = np.squeeze(gamma['Water'])
        assert self.direct_gamma.shape == self.protein_gamma.shape == self.water_gamma.shape == (self.q,self.q)
        self.gamma = self.p.gamma

        # set other attributes
        self.burial_in_context = self.p.burial_in_context
        self.aa_freq = frustration.compute_aa_freq(self.sequence, AA=self.gamma.alphabet)
        self.contact_freq = frustration.compute_contact_freq(self.sequence, AA=self.gamma.alphabet)
        charges2 = self.p.charges[:,np.newaxis] * self.p.charges[np.newaxis,:]
        if self.p.k_electrostatics != 0:
            self.sequence_cutoff=min(self.p.min_sequence_separation_electrostatics, self.p.min_sequence_separation_contact)
            self.distance_cutoff=None # the distance matrix isn't guaranteed to exist in all subclasses,
                                      # but it doesn't hurt to define the distance_cutoff attribute--
                                      # it's just like any other parameter, such as sequence_cutoff,
                                      # that only matters if we need to compute a mask from a distance matrix 
            self.electrostatics_gamma = -self.p.k_electrostatics * charges2[np.newaxis, np.newaxis, :, :]
        else:
            self.sequence_cutoff=self.p.min_sequence_separation_contact
            self.distance_cutoff=self.p.distance_cutoff_contact # the distance matrix isn't guaranteed to exist in all subclasses,
                                                                # but it doesn't hurt to define the distance_cutoff attribute--
                                                                # it's just like any other parameter, such as sequence_cutoff,
                                                                # that only matters if we need to compute a mask from a distance matrix
        self.charges2 = charges2 
        self._decoy_fluctuation = {} # used for mutational calculation, possibly others
        self.minimally_frustrated_threshold=.78 # this should be a class variable or an argument to __init__

    # carlos wanted to have gamma_array with gammas multiplied by lambda and coefficients
    @property
    def coefficient_lambda_gamma_array(self):
        _coefficient_lambda_gamma_array = []
        _coefficient_lambda_gamma_array.append(-0.5 * self.p.k_contact * self.burial_gamma[:,0])
        _coefficient_lambda_gamma_array.append(-0.5 * self.p.k_contact * self.burial_gamma[:,1])
        _coefficient_lambda_gamma_array.append(-0.5 * self.p.k_contact * self.burial_gamma[:,2])
        _coefficient_lambda_gamma_array.append(-0.5 * self.p.k_contact * self.direct_gamma)
        _coefficient_lambda_gamma_array.append(-0.5 * self.p.k_contact * self.protein_gamma)
        _coefficient_lambda_gamma_array.append(-0.5 * self.p.k_contact * self.water_gamma)
        _coefficient_lambda_gamma_array.append(0.5 * self.p.k_electrostatics * self.charges2)
        #  not a typo, supposed to be positive ^^^
        # charges2 is our electrostatic "gamma"
        return _coefficient_lambda_gamma_array
    @coefficient_lambda_gamma_array.setter
    def coefficient_lambda_gamma_array(self):
        raise AttributeError("""Setting AWSEM.coefficient_lambda_gamma_array 
                                directly is not allowed. Modify AWSEM.k_contact, 
                                AWSEM.burial_gamma, AWSEM.direct_gamma, 
                                AWSEM.protein_gamma, or AWSEM.water_gamma instead.""")

    def subclass_setup_helper(self):
        """
        This method calls methods to calculate native indicator functions, 
        masks (based on the native distance matrix), and native energy,
        then optionally sets up the potts model.

        This method is intended to be called as the last step of __init__
        in each subclass of AWSEMBase. The subclasses may differ in how
        they load in the structural information (the part of __init__ 
        preceding the call to this method) and how they implement
        the calculate_indicators and calculate_masks methods called 
        by subclass_setup_helper
        """
        self.calculate_masks() # subclasses should (re)define this method as needed
        self.calculate_indicators() # subclasses should (re)define this method as needed
        if self.potts:
            self.calculate_energy_and_potts()
        else:
            if 'potts_model' in dir(self) or 'burial_energy' in dir(self)\
              or 'contact_energy' in dir(self) or '_native_energy' in dir(self):
                # if one has been defined, they should all have been defined
                assert 'potts_model' in dir(self), dir(self)
                assert 'burial_energy' in dir(self), dir(self)
                assert 'contact_energy' in dir(self), dir(self)
                assert '_native_energy' in dir(self), dir(self)
                # potts model and energies will be inaccurate once indicators are modified;
                # if we don't care about the potts model, then we should delete the old
                # data so it can't be accidentally misused in the future
                del self.potts_model
                del self.burial_energy
                del self.contact_energy
                del self._native_energy
             
    def calculate_indicators(self):
        raise NotImplementedError("Subclasses must implement this method")

    def calculate_masks(self):
        # calculate masks
        if self.burial_in_context==True:
            selected_matrix=self.full_pdb_distance_matrix
        else:
            selected_matrix=self.distance_matrix
        self.sequence_mask_rho = frustration.compute_mask(selected_matrix, 
                                                     maximum_contact_distance=None, 
                                                     minimum_sequence_separation = self.p.min_sequence_separation_rho)
        self.sequence_mask_contact = frustration.compute_mask(self.distance_matrix, 
                                                     maximum_contact_distance=self.p.distance_cutoff_contact, 
                                                     minimum_sequence_separation = self.p.min_sequence_separation_contact)
        self.electrostatics_mask = frustration.compute_mask(self.distance_matrix, 
                                                     maximum_contact_distance=None, 
                                                     minimum_sequence_separation=self.p.min_sequence_separation_electrostatics)
        #with open('my_data.txt','w') as f:
        #    f.write(f"self.distance_cutoff: {self.distance_cutoff}\n")
        #    f.write(f"self.sequence_cutoff: {self.sequence_cutoff}\n")
        #np.save('my_distance_matrix.npy',self.distance_matrix)
        self.mask = frustration.compute_mask(self.distance_matrix, 
                                             maximum_contact_distance=self.distance_cutoff, 
                                             minimum_sequence_separation = self.sequence_cutoff)
        #np.save('my_mask_new.npy',self.mask)
        self.selected_matrix = selected_matrix # we'll need this in the calculate_indicators function 

    def calculate_energy_and_potts(self):

        J_index = np.meshgrid(range(self.N), range(self.N), range(self.q), range(self.q), indexing='ij', sparse=False)
        h_index = np.meshgrid(range(self.N), range(self.q), indexing='ij', sparse=False)
        
        # compute burial and contact energies
        self.burial_energy = 0.5 * self.p.k_contact * self.burial_gamma[h_index[1]] * self.burial_indicator[:, np.newaxis, :] 
        direct = self.direct_indicator * self.direct_gamma[J_index[2], J_index[3]]
        water_mediated = self.water_indicator * self.water_gamma[J_index[2], J_index[3]]
        protein_mediated = self.protein_indicator  * self.protein_gamma[J_index[2], J_index[3]]
        contact_energy = self.p.k_contact * np.array([direct, water_mediated, protein_mediated]) * self.sequence_mask_contact[np.newaxis, :, :, np.newaxis, np.newaxis]
        
        # Compute electrostatics and add to contact energy
        if self.p.k_electrostatics!=0:
            electrostatics_energy = self.electrostatics_gamma * self.electrostatics_indicator[:,:,np.newaxis,np.newaxis]
            contact_energy = np.append(contact_energy, electrostatics_energy[np.newaxis,:,:,:,:], axis=0)

        self.contact_energy = contact_energy

        # Compute potts model
        self.potts_model = {}
        self.potts_model['h'] = self.burial_energy.sum(axis=-1)[:, :]#self.aa_map_awsem_list]
        assert self.potts_model['h'].shape == (self.N, self.q), self.potts_model['h'].shape
        self.potts_model['J'] = self.contact_energy.sum(axis=0)[:, :, :, :]#self.aa_map_awsem_x, self.aa_map_awsem_y]
        assert self.potts_model['J'].shape == (self.N, self.N, self.q, self.q), self.potts_model['J'].shape 
        # Set the gap energy to zero
        #self.potts_model['h'][:, 0] = 0
        #self.potts_model['J'][:, :, 0, :] = 0
        #self.potts_model['J'][:, :, :, 0] = 0
        self._native_energy=None # don't know what this does


    def compute_configurational_decoy_statistics(self):
        raise NotImplementedError("Subclasses must define this method")

    def compute_configurational_energies(self):
        raise NotImplementedError("Subclasses must define this method")

    def configurational_frustration(self,aa_freq=None, correction=0, n_decoys=4000):
        mean_decoy_energy, std_decoy_energy = self.compute_configurational_decoy_statistics(n_decoys=n_decoys,aa_freq=aa_freq)
        return -(self.compute_configurational_energies()-mean_decoy_energy)/(std_decoy_energy+correction)


class AWSEM(AWSEMBase):

    def __init__(self, 
                 pdb_structure: object | tuple, # tuple is an object, but this clarifies what we expect
                 sequence: str =None,
                 expose_indicator_functions: bool=False,
                 potts: bool=True,
                 alt_sigma_wat: bool=False,
                 **parameters)->object:
        # assume the user wanted the sequence from the pdb structure if not given
        if not sequence:
            try:
                sequence = pdb_structure.sequence
            except:
                if isinstance(pdb_structure,tuple):
                    raise ValueError("""It seems that you are trying to use 
                                        the tuple pdb_structure format, which
                                        specifies a conformation but not a sequence.
                                        In this case, you must provide the sequence
                                        as a separate argument to this class.""")
                else:
                    raise
        # load structure-independent parameters and methods
        super().__init__(sequence, expose_indicator_functions, potts, **parameters) 
        self.alt_sigma_wat = alt_sigma_wat
        # set up strucure
        self.setup_structure(pdb_structure)
        self.subclass_setup_helper()

    def setup_structure(self, pdb_structure):
        if not isinstance(pdb_structure, tuple): # alt_conf should be our custom Structure object
                                                 # maybe our type check here should be more restrictive,
                                                 # but the __init__ only requires pdb_structure to be an object,
                                                 # so I'll take my cue from that
            # check structure
            selection_CB = pdb_structure.structure.select('name CB or (resname GLY IGL and name CA)')
            resid = selection_CB.getResindices()
            N=len(resid)
            self.resid = resid
            self.N = N
            # set structure-dependent properties
            self._pdb_structure = pdb_structure
            self.structure=pdb_structure.structure
            self.chain=pdb_structure.chain
            self.pdb_file=pdb_structure.pdb_file
            self.init_index_shift=pdb_structure.init_index_shift
            self.full_to_aligned_index_dict=pdb_structure.full_to_aligned_index_dict
            self.distance_matrix=pdb_structure.distance_matrix
            self.full_pdb_distance_matrix=pdb_structure.full_pdb_distance_matrix
            self.midpoint_matrix = pdb_structure.midpoint_matrix 
            #      midpoint matrix is used to map interacting pairs to a single point in space
        elif isinstance(pdb_structure, tuple): # pdb_structure is defined by a few distance matrices
            if len(pdb_structure)==3\
              and isinstance(pdb_structure[0],np.ndarray)\
              and isinstance(pdb_structure[1],np.ndarray)\
              and isinstance(pdb_structure[2],np.ndarray) or pdb_structure[2] is None:
                # pdb_structure is a full_pdb_distance_matrix 
                # followed by a distance_matrix
                # followed by a midpoint matrix (or None)
                self._pdb_structure = None # we're getting our conformer from within python, not a pdb file
                self.structure = None # we're getting our conformer from within python, not a pdb file
                self.full_pdb_distance_matrix = pdb_structure[0]
                self.distance_matrix = pdb_structure[1]
                self.midpoint_matrix = pdb_structure[2] 
                #      midpoint matrix is used to map interacting pairs to a single point in space;
                #      usually not necessary, so it will usually be None
                #
                # the rest of the attributes that are set in the case that pdb_structure is a Structure
                # either remain the same (if this method has been previously called with a Structure)
                # or go undefined (if we are passing a list of arrays the first time that we are calling 
                # this method)
        else:
            raise AssertionError("unexpected else block")

    @property
    def pdb_structure(self):
        return self._pdb_structure
    @pdb_structure.setter
    def pdb_structure(self,pdb_structure):
        # reset structural attributes
        self.setup_structure(pdb_structure)
        # check that our new structure is compatible with our old one
        if self.N != len(self.sequence):
            breakpoint()
            raise ValueError("The pdb is incomplete. Try setting 'repair_pdb=True' when constructing the Structure object.")
        self.subclass_setup_helper()
    def change_conformation(self,alt_conf):
        # this method is an alias for the setter
        self.pdb_structure = alt_conf

    def calculate_indicators(self):
        # Calculate rho
        rho = 0.25 
        rho *= (1 + np.tanh(self.p.eta * (self.selected_matrix - self.p.r_min)))
        rho *= (1 + np.tanh(self.p.eta * (self.p.r_max - self.selected_matrix)))
        rho *= self.sequence_mask_rho
        self.rho=rho
        #Calculate sigma water
        rho_r = (rho).sum(axis=1)
        if self.full_pdb_distance_matrix.shape!=self.distance_matrix.shape:
            if self.burial_in_context==True:
                self.init_index_shift=self.pdb_structure.init_index_shift
                self.fin_index_shift=self.pdb_structure.fin_index_shift
                rho_r=rho_r[self.init_index_shift:self.fin_index_shift]
        self.rho_r=rho_r
        rho_b = np.expand_dims(rho_r, 1)
        rho1 = np.expand_dims(rho_r, 0)
        rho2 = np.expand_dims(rho_r, 1)
        sigma_water = 0.25 * (1 - np.tanh(self.p.eta_sigma * (rho1 - self.p.rho_0))) * (1 - np.tanh(self.p.eta_sigma * (rho2 - self.p.rho_0)))
        if self.alt_sigma_wat:
            sigma_water = -sigma_water + 0.5*( (1 - np.tanh(self.p.eta_sigma * (rho1 - self.p.rho_0))) + (1 - np.tanh(self.p.eta_sigma * (rho2 - self.p.rho_0))))
        sigma_protein = 1 - sigma_water
        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.p.eta * (self.distance_matrix - self.p.r_min))) * (1 + np.tanh(self.p.eta * (self.p.r_max - self.distance_matrix)))
        thetaII = 0.25 * (1 + np.tanh(self.p.eta * (self.distance_matrix - self.p.r_minII))) * (1 + np.tanh(self.p.eta * (self.p.r_maxII - self.distance_matrix)))
        burial_indicator = np.tanh(self.p.burial_kappa * (rho_b - self.p.burial_ro_min)) + np.tanh(self.p.burial_kappa * (self.p.burial_ro_max - rho_b))
        direct_indicator = theta[:, :, np.newaxis, np.newaxis]
        water_indicator = thetaII[:, :, np.newaxis, np.newaxis] * sigma_water[:, :, np.newaxis, np.newaxis]
        protein_indicator = thetaII[:, :, np.newaxis, np.newaxis] * sigma_protein[:, :, np.newaxis, np.newaxis]
        # store indicators and gammas for our particular sequence as attributes
        self.indicators=[]
        self.indicators.append(burial_indicator[:,0])
        self.indicators.append(burial_indicator[:,1])
        self.indicators.append(burial_indicator[:,2])
        self.indicators.append(direct_indicator[:,:,0,0]*self.sequence_mask_contact)
        self.indicators.append(protein_indicator[:,:,0,0]*self.sequence_mask_contact)
        self.indicators.append(water_indicator[:,:,0,0]*self.sequence_mask_contact)
        self.burial_indicator = burial_indicator 
        self.direct_indicator = direct_indicator 
        self.water_indicator = water_indicator   
        self.protein_indicator = protein_indicator
        #breakpoint()
        if True:#self.p.k_electrostatics != 0:
            electrostatics_indicator = 1 / (self.distance_matrix + 1E-6) * np.exp(-self.distance_matrix / self.p.electrostatics_screening_length) * self.electrostatics_mask
            self.indicators.append(electrostatics_indicator)
            self.electrostatics_indicator = electrostatics_indicator 

    def calculate_energy_and_potts(self):
        super().calculate_energy_and_potts()
        if not self.expose_indicator_functions:
            del self.burial_indicator
            del self.direct_indicator
            del self.water_indicator
            del self.protein_indicator
            if "electrostatics_indicator" in dir(self):
                # won't exist if electrostatics are turned off
                del self.electrostatics_indicator
            del self.indicators

    def compute_configurational_decoy_statistics(self, n_decoys=4000,aa_freq=None):
        # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        _AA = self.gamma.alphabet #'ARNDCQEGHILKMFPSTWYV'
        if aa_freq is None:
            seq_index = np.array([_AA.index(aa) for aa in self.sequence])
            N=self.N
        else:
            N=self.N*10
            total = sum(aa_freq)
            probabilities = [freq / total for freq in aa_freq.ravel()]
            seq_index = np.random.choice(a=len(aa_freq), size=N, p=probabilities)
        
        distances = np.triu(self.distance_matrix)
        distances = distances[(distances<self.distance_cutoff_contact) & (distances>0)]

        rho_b = np.expand_dims(self.rho_r, 1) #(n,1)
        rho1 = np.expand_dims(self.rho_r, 0) #(1,n)
        rho2 = np.expand_dims(self.rho_r, 1) #(n,1)

        sigma_water = 0.25 * (1 - np.tanh(self.eta_sigma * (rho1 - self.rho_0))) * (1 - np.tanh(self.eta_sigma * (rho2 - self.rho_0))) #(n,n)
        sigma_protein = 1 - sigma_water #(n,n)

        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - distances))) # (c,)
        thetaII = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - distances))) #(c,)
        burial_indicator = np.tanh(self.burial_kappa * (rho_b - self.burial_ro_min)) + np.tanh(self.burial_kappa * (self.burial_ro_max - rho_b)) #(n,3)
           
        charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        electrostatics_indicator = np.exp(-distances / self.electrostatics_screening_length) / distances

        decoy_energies=np.zeros(n_decoys)
        #decoy_data=[None]*n_decoys
        #decoy_data_columns=['decoy_i','rand_i_resno','rand_j_resno','ires_type','jres_type','i_resno','j_resno','rij','rho_i','rho_j','water_energy','burial_energy_i','burial_energy_j','electrostatic_energy','tert_frust_decoy_energies']
        for i in range(n_decoys):
            c=np.random.randint(0,len(distances))
            n1=np.random.randint(0,self.N)
            n2=np.random.randint(0,self.N)
            qi1=np.random.randint(0,N)
            qi2=np.random.randint(0,N)
            q1=seq_index[qi1]
            q2=seq_index[qi2]

            
            burial_energy1 = (-0.5 * self.k_contact * self.burial_gamma[q1] * burial_indicator[n1]).sum(axis=0)
            burial_energy2 = (-0.5 * self.k_contact * self.burial_gamma[q2] * burial_indicator[n2]).sum(axis=0)
            
            direct = theta[c] * self.direct_gamma[q1, q2]
            water_mediated = sigma_water[n1,n2] * thetaII[c] * self.water_gamma[q1,q2]
            protein_mediated = sigma_protein[n1,n2] * thetaII[c] * self.protein_gamma[q1,q2]
            contact_energy = -self.k_contact * (direct+water_mediated+protein_mediated)
            electrostatics_energy = self.k_electrostatics * electrostatics_indicator[c]*charges[q1]*charges[q2]

            decoy_energies[i]=(burial_energy1+burial_energy2+contact_energy+electrostatics_energy)
            #decoy_data[i]=[i, qi1, qi2, q1, q2, n1, n2, distances[c], self.rho_r[n1], self.rho_r[n2], contact_energy/4.184, burial_energy1/4.184, burial_energy2/4.184, electrostatics_energy/4.184, decoy_energies[i]]
            
        mean_decoy_energy = np.mean(decoy_energies)
        std_decoy_energy = np.std(decoy_energies)
        return mean_decoy_energy, std_decoy_energy
    
    def compute_configurational_energies(self):
        _AA= self.gamma.alphabet #'ARNDCQEGHILKMFPSTWYV'
        seq_index = np.array([_AA.index(aa) for aa in self.sequence])
        distances = np.triu(self.distance_matrix)
        distances = distances[(distances<self.distance_cutoff_contact) & (distances>0)]
        n_contacts=len(distances)

        n = self.distance_matrix.shape[0]  # Assuming self.distance_matrix is defined and square
        tri_upper_indices = np.triu_indices(n, k=1)  # k=1 excludes the diagonal
        valid_pairs = (self.distance_matrix[tri_upper_indices] < self.distance_cutoff_contact) & \
                      (self.distance_matrix[tri_upper_indices] > 0)
        indices1,indices2 = (tri_upper_indices[0][valid_pairs], tri_upper_indices[1][valid_pairs])

        # for n1,n2,c in zip(indices1,indices2,range(n_contacts)):
        #     assert self.distance_matrix[n1,n2] == distances[c]
        
        rho_b = np.expand_dims(self.rho_r, 1) #(n,1)
        rho1 = np.expand_dims(self.rho_r, 0) #(1,n)
        rho2 = np.expand_dims(self.rho_r, 1) #(n,1)

        sigma_water = 0.25 * (1 - np.tanh(self.eta_sigma * (rho1 - self.rho_0))) * (1 - np.tanh(self.eta_sigma * (rho2 - self.rho_0))) #(n,n)
        sigma_protein = 1 - sigma_water #(n,n)

        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - distances))) # (c,)
        thetaII = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - distances))) #(c,)
        burial_indicator = np.tanh(self.burial_kappa * (rho_b - self.burial_ro_min)) + np.tanh(self.burial_kappa * (self.burial_ro_max - rho_b)) #(n,3)
           
        charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        electrostatics_indicator = np.exp(-distances / self.electrostatics_screening_length) / distances

        # decoy_data_columns=['decoy_i','i_resno','j_resno','ires_type','jres_type','aa1','aa2','rij','rho_i','rho_j','water_energy','burial_energy_i','burial_energy_j','electrostatic_energy','total_energies']
        # decoy_data=[]
        configurational_energies=np.zeros((n,n))
        for c in range(n_contacts):
            n1=indices1[c]
            n2=indices2[c]
            q1=seq_index[n1]
            q2=seq_index[n2]

            burial_energy1 = (-0.5 * self.k_contact * self.burial_gamma[q1] * burial_indicator[n1]).sum(axis=0)
            burial_energy2 = (-0.5 * self.k_contact * self.burial_gamma[q2] * burial_indicator[n2]).sum(axis=0)
            
            direct = theta[c] * self.direct_gamma[q1, q2]
            water_mediated = sigma_water[n1,n2] * thetaII[c] * self.water_gamma[q1,q2]
            protein_mediated = sigma_protein[n1,n2] * thetaII[c] * self.protein_gamma[q1,q2]
            contact_energy = -self.k_contact * (direct+water_mediated+protein_mediated)
            electrostatics_energy = self.k_electrostatics * electrostatics_indicator[c]*charges[q1]*charges[q2]

            energy=(burial_energy1+burial_energy2+contact_energy+electrostatics_energy)
            configurational_energies[n1,n2]=energy
            configurational_energies[n2,n1]=energy
            # decoy_data+=[[c, n1, n2, q1, q2, _AA[q1],_AA[q2], distances[c], self.rho_r[n1], self.rho_r[n2], contact_energy/4.184, burial_energy1/4.184, burial_energy2/4.184, electrostatics_energy/4.184, energy/4.184]]
        # import pandas as pd
        return configurational_energies #, pd.DataFrame(decoy_data, columns=decoy_data_columns)
    

class AWSEMIndicators(AWSEMBase): # PottsEvaluatorFromIndicators or PottsEnergyEvaluatorFromIndicators?

    def __init__(self, 
                 burial_indicator: np.ndarray,
                 direct_indicator: np.ndarray,
                 protein_indicator: np.ndarray,
                 water_indicator: np.ndarray,
                 electrostatics_indicator: Union[np.ndarray, None],
                 sequence: str,          # sequence is optional if we initialize from a Structure but not here
                 expose_indicator_functions: bool=False,
                 absolute_value_gamma: bool=False,
                 **parameters)->object:
        """
        A stripped-down version of the AWSEM class that can be initialized from a set of indicator functions

        Parameters
        ----------
        burial_indicator : np.ndarray
            Burial indicator array, most likely accessed using the burial_indicator attribute of an AWSEM
        direct_indicator : np.ndarray
            Direct indicator array, most likely accessed using the direct_indicator attribute of an AWSEM
        protein_indicator : np.ndarray
            Protein indicator array, most likely accessed using the protein_indicator attribute of an AWSEM
        water_indicator : np.ndarray
            Water indicator array, most likely accessed using the water_indicator attribute of an AWSEM
        electrostatics_indicator : Union[np.ndarray, None]
            Electrostatics indicator array, most likely accessed using the electrostatics_indicator attribute of an AWSEM.
            May be None is electrostatics were turned off (k_electrostatics=0).
        sequence :  str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. 
        expose_indicator_functions: bool
            If set to True, indicator functions of the contact and burial energy terms can be accessed by user.
        absolute_value_gamma: bool
            If True, replace gammas with their absolute values. This is helpful for the standard deviation approximation

        Returns
        -------
        AWSEMIndicators object

        """
        # if we already have our indicator functions, 
        # our goal is probably to compute the potts model,
        # so we'll just hard code a value of True for that argument  VVVV
        super().__init__(sequence, expose_indicator_functions, potts=True, **parameters)
        self.burial_indicator = burial_indicator
        self.direct_indicator = direct_indicator
        self.protein_indicator = protein_indicator
        self.water_indicator = water_indicator
        self.electrostatics_indicator = electrostatics_indicator
        # we don't have a distance matrix to a apply a minimum sequence separation to--
        #     we have to assume that this consideration was already made when computing the indicators.
        #     So we just set the "distance" matrix to zeros and set no maximum cutoff, so that nothing changes
        # however, we can apply a minimum sequence separation-based mask to the matrix
        self.sequence_mask_contact = frustration.compute_mask(np.zeros((self.N,self.N)), 
                                    maximum_contact_distance=None, 
                                    minimum_sequence_separation = self.p.min_sequence_separation_contact)
        self.electrostatics_mask = frustration.compute_mask(np.zeros((self.N,self.N)),
                                    maximum_contact_distance=None, 
                                    minimum_sequence_separation=self.p.min_sequence_separation_electrostatics)
        self.mask = frustration.compute_mask(np.zeros((self.N,self.N)), 
                                             maximum_contact_distance=self.distance_cutoff, 
                                             minimum_sequence_separation = self.sequence_cutoff)
        if absolute_value_gamma:
            self.burial_gamma = np.abs(self.burial_gamma)
            self.direct_gamma = np.abs(self.direct_gamma)
            self.protein_gamma = np.abs(self.protein_gamma)
            self.water_gamma = np.abs(self.water_gamma)
            self.electrostatics_gamma = np.abs(self.electrostatics_gamma)
        self.absolute_value_gamma = absolute_value_gamma 
        #np.save('absolute_value_gamma_1.npy',absolute_value_gamma)
        #np.save('burial_indicator_1.npy',burial_indicator)
        #np.save('direct_indicator_1.npy', direct_indicator)
        #np.save('protein_indicator_1.npy', protein_indicator)
        #np.save('water_indicator_1.npy', water_indicator)
        #np.save('electrostatics_indicator_1.npy', electrostatics_indicator)
        self.subclass_setup_helper()

    def calculate_indicators(self):
        pass # the function was initialized with indicators, so there's nothing to do

class AWSEMVariancePotts(AWSEMBase):
    def __init__(self, 
                 covariance_matrix: np.ndarray,
                 sequence: str,          # sequence is optional if we initialize from a Structure but not here
                 expose_indicator_functions: bool=False,
                 absolute_value_gamma: bool=False,
                 **parameters)->object:
        """

        Parameters
        ----------
        covariance_matrix: np.ndarray
            Covariance matrix of all __indicator functions___ (not residues) over a decoy set
        sequence :  str
            The amino acid sequence of the protein. The sequence is assumed to be in one-letter code. 
        expose_indicator_functions: bool
            If set to True, indicator functions of the contact and burial energy terms can be accessed by user.
        absolute_value_gamma: bool
            If True, replace gammas with their absolute values. This is helpful for the standard deviation approximation

        Returns
        -------
        AWSEMVariancePotts object

        """
        # if we already have our indicator functions, 
        # our goal is probably to compute the potts model,
        # so we'll just hard code a value of True for that argument  VVVV
        super().__init__(sequence, expose_indicator_functions, potts=True, **parameters)
        self.covariance_matrix = covariance_matrix
        self.num_indicators = 3*self.N + 4*(self.N**2-self.N)/2 # low, med, high burial for each N, 4 classes of pair interactions
        self.subclass_setup_helper()

    @staticmethod # trying to avoid loading down memory with too many permanent attributes
    def pairwise_mask(l): # l for length
        # Helps us figure out where a 1D array's elements were in an upper triangular matrix,
        #     assuming the matrix was flattened row-major style and the main diagonal was excluded.
        #     Each index i of this list gives us the indices of the 1D array that were in row i of the matrix
        #NbyN_matrix_rows = [[range(n*l-int(((n**2)+n)/2),(n+1)*l-int((((n+1)**2)+(n+1))/2)),
        #                     ] for n in range(l)] 
        mask = np.zeros((l,l))
        for i in range(l):
            temp = np.zeros((l,l))
            # set elements involving i equal to 1
            temp[:,i] = 1
            temp[i,:] = 1
            temp = temp[np.triu_indices(l,k=1)] # this flattens the array, keeping only the upper triangle
            found = np.where(temp==1)[0]
            try:
                mask[i, :] = [1 if index in found else 0 for index in range(l)]#[1 if index in NbyN_matrix_rows[i] else 0 for index in range(l)]
            except: 
                import pdb; pdb.set_trace()
        return mask

    def calculate_indicators(self):
        print("start calculate indicators")
        assert len(self.covariance_matrix.shape)==2, self.covariance_matrix.shape
        assert self.covariance_matrix.shape[0] == self.covariance_matrix.shape[1]
        print("assertions complete")
        # each "indicator function" is actually a covariance of two indicator functions.
        # There are a few different kinds of pairs of indicator functions.
        #     The first kind is self-covariances, AKA variances, which we further break down
        #     into burial (dependent on AA identity at a single position) 
        #     and pairwise (dependent on two positions)
        self.burial_variances = np.diag(self.covariance_matrix[:3*self.N,:3*self.N]) # shape (3N,)
        self.pairwise_variances = np.diag(self.covariance_matrix[3*self.N:,3*self.N:]) # shape (4N,)
        np.save("burial_variances.npy", self.burial_variances)
        np.save("pairwise_variances.npy", self.pairwise_variances)
        print("variances calculated")
        #     The other kind of covariance is a covarience between indicator functions.
        #     We break these down into burial-burial covariances (dependent on two identities),
        #     burial-pairwise indicator covariances (some dependent on 2, others dependent on 3 identities),
        #     and pairwise indicator-pairwise indicator covariances (dependent on 3 or 4 identities)
        self.burial_burial_covariances = self.covariance_matrix[np.triu_indices(3*self.N,k=1)] # shape (((3N)**2-3N)/2,)
        print("first covariances calculated")
        num_upper = int((self.N**2-self.N)/2)
        burial_pairwise_covariances_2 = np.zeros((3*self.N, 4*num_upper)) # shape (3N, 4((N**2-N)/2))
        #self.burial_pairwise_covariances_2 = np.concatenate([ # can be represented by 2-body term in Potts model
            # pairwise_mask tells us which pairwise indicator functions in a given row
            # involve the residue whose burial covariances are evaluated in that row
            # (the covariance matrix has more elements than there are energy terms--
            #  some elements represent relationships between residues that don't interact directly,
            #  so there are many indicators in each row i involving residues j and k but not i)
            #     
            # We repeat pairwise_mask 3 times because each residue is repeated 3 times (low, med, high density)
        #    self.covariance_matrix[:3*self.N, 3*self.N+i*num_upper:3*self.N+(i+1)*num_upper]\
        #            *self.pairwise_mask(num_upper)[:self.N,:].repeat(3,axis=0)\
        #                for i in range(4)], axis=1) # shape (3N,4((N**2-N)/2))
        for counter in range(4):
            burial_pairwise_covariances_2[:,counter*num_upper:(counter+1)*num_upper] =\
                (self.covariance_matrix[:3*self.N, 3*self.N+counter*num_upper:3*self.N+(counter+1)*num_upper]+1E-10)\
                    *self.pairwise_mask(num_upper)[:self.N,:].repeat(3,axis=0) # tiny shift of 1E-10 ensures that the only contacts at exactly 0 are those that fail the mask
        self.burial_pairwise_covariances_2 = burial_pairwise_covariances_2
        print("second covariances calculated")
        # these last three components contain many more covariances than the others
        # and are likely to be sparse
        self.burial_pairwise_covariances_3 = None # set to every burial-pairwise covariance not in the previous one
        self.pairwise_pairwise_covariances_3 = None # set to every pairwise where one residue is common between the two
        self.pairwise_pairwise_covariances_4 = None # everything not in pairwise_pairwise_covariances_3

    def calculate_energy_and_potts(self):

        J_index = np.meshgrid(range(self.N), range(self.N), range(self.q), range(self.q), indexing='ij', sparse=False)
        h_index = np.meshgrid(range(self.N), range(self.q), indexing='ij', sparse=False)
        
        # compute burial and contact energies
        #     the "energy" of our potts model representing the covariance, not a physical energy
        #     this "burial energy" is the sum of variances of the burial indicators (the one-body part of the model)
        self.burial_energy = (0.5*self.p.k_contact*self.burial_gamma[h_index[1]])**2 * self.burial_variances.reshape((self.N,1,3))
        #     the "contact energy" is ordinarily the sum of all two-body components of the model
        #     (direct, protein, water, electrostatics), so we do the analogous thing here
        template = np.zeros((self.N,self.N))
        num_upper = int((self.N**2-self.N)/2)
        triu_indices = np.triu_indices(self.N,k=1)
        template[triu_indices] = self.pairwise_variances[:num_upper]
        direct = (template+template.T)[:,:,np.newaxis,np.newaxis] * self.direct_gamma[J_index[2], J_index[3]]**2
        template[triu_indices] = self.pairwise_variances[num_upper:2*num_upper]
        protein_mediated = (template+template.T)[:,:,np.newaxis,np.newaxis] * self.protein_gamma[J_index[2], J_index[3]]**2
        template[triu_indices] = self.pairwise_variances[2*num_upper:3*num_upper]
        water_mediated = (template+template.T)[:,:,np.newaxis,np.newaxis] * self.water_gamma[J_index[2], J_index[3]]**2
        contact_energy = self.p.k_contact * np.array([direct, protein_mediated, water_mediated])
        if self.p.k_electrostatics!=0:
            template[triu_indices] = self.pairwise_variances[3*num_upper:]
            electrostatics_energy = self.electrostatics_gamma * (template+template.T)[:,:,np.newaxis,np.newaxis]**2
            contact_energy = np.append(contact_energy, electrostatics_energy[np.newaxis,:,:,:,:], axis=0)
        #    for the variance potts model, there is one more kind of two-body interaction:
        #    burial-pairwise covariance when the pairwise energy term involves the residue in the burial term
        #        self.burial_pairwise_covariances_2 has shape (3N, 4(N^2-N)/2)
        #        we first multiply each row by the appropriate burial energy
        temp = self.burial_pairwise_covariances_2
        low = temp[::3,:,np.newaxis]*0.5*self.p.k_contact*self.burial_gamma[h_index[1],0]
        med = temp[1::3,:,:]*0.5*self.p.k_contact*self.burial_gamma[h_index[1],1]
        high = temp[2::3,:,:]*0.5*self.p.k_contact*self.burial_gamma[h_index[1],2]
        #        we can now collapse our 3 burial indicator types
        temp = np.sum(np.concatenate((low[None,...], med[None,...], high[None,...]), axis=0), axis=0)
        assert temp.shape == (self.N, 4*((self.N**2-self.N)/2)), temp.shape
        #        now we split into our 4 pairwise contact types, keeping only the elements of each row
        #        that represent a pairwise interaction involving the residue whose burial covariances are found in that row
        direct, prot, wat, elec = np.split(temp[temp!=0], 4, axis=1)
        #        now we need to go from shape (N, (N^2-N)/2) to (N,N)
        #        (each residue burial indicator covaries with (N^2-N)/2 pairwise indicators,
        #         but only N of them include the same residue from the burial indicator;
        #         others have a value of 0, which we can easily eliminate)
        #        we also need to multiply by our pairwise gammas
        direct = direct[direct != 0].reshape((self.N,self.N,self.q))[...,np.newaxis]*self.direct_gamma[J_index[3]]*self.p.k_contact
        prot = prot[prot != 0].reshape((self.N,self.N,self.q))[...,np.newaxis]*self.protein_gamma[J_index[3]]*self.p.k_contact
        wat = wat[wat != 0].reshape((self.N,self.N,self.q))[...,np.newaxis]*self.water_gamma[J_index[3]]*self.p.k_contact
        elec = elec[elec != 0].reshape((self.N,self.N,self.q))[...,np.newaxis]*self.electrostatics_gamma[J_index[3]]*self.p.k_contact

        contact_energy = np.append(contact_energy, direct[np.newaxis,...], axis=0)
        contact_energy = np.append(contact_energy, prot[np.newaxis,...], axis=0)
        contact_energy = np.append(contact_energy, wat[np.newaxis,...], axis=0)
        contact_energy = np.append(contact_energy, elec[np.newaxis,...], axis=0)

        """
        direct = np.zeros((self.N,self.N))
        direct[triu_indices] = 
        direct

        num_upper = int((self.N**2-self.N)/2)
        triu_indices = np.triu_indices(self.N,k=1)        
        template = np.zeros((self.N,self.N))
        
        assert direct.shape==(self.N,self.N), direct.shape
        assert prot.shape==(self.N,self.N), prot.shape
        assert wat.shape==(self.N,self.N), wat.shape
        assert elec.shape==(self.N,self.N), elec.shape


        for counter,row in enumerate(self.burial_pairwise_covariances_2):
            direct_indicators = row[:len(row)//4]
            direct_energy = direct_indicators[direct_indicators>0].reshape((-1,1))\
                *0.5*self.p.k_contact*self.burial_gamma[:,counter%3]\
                *self.p.k_contact*self.direct_gamma[]
            direct = row[row!=0] * 0.5*self.p.k_contact*self.burial_gamma[]
            template[counter] = row[row==1] # where the residue corresponding to the burial row is involved in the pairwise indicator
        burial_pairwise_2 = self.burial_pairwise_covariances_2[]
        """
        ####################################################################
        # the potts model that we're using (AWSEMEnergy) multiplies each of the J terms by 1/2,
        # so we should multiply them by 2 to cancel that out
        contact_energy[np.diag_indices(contact_energy.shape[0])] *= 1/2
        contact_energy *= 2
        ####################################################################

        self.contact_energy = contact_energy

        # Compute potts model
        self.potts_model = {}
        self.potts_model['h'] = self.burial_energy.sum(axis=-1)[:, :]#self.aa_map_awsem_list]
        self.potts_model['J'] = self.contact_energy.sum(axis=0)[:, :, :, :]#self.aa_map_awsem_x, self.aa_map_awsem_y]
        # Set the gap energy to zero
        #self.potts_model['h'][:, 0] = 0
        #self.potts_model['J'][:, :, 0, :] = 0
        #self.potts_model['J'][:, :, :, 0] = 0
        self._native_energy=None # don't know what this does

class DecoyEnsemble():

    def __init__(self, 
                 pdb_structures: Generator[object,None,None],
                 **parameters)->object:
        """
        Generate DecoyEnsemble object

        Parameters
        ----------
        pdb_structures : Generator[object,None,None]
            yields Structure objects representing decoy structures
        other parameters:
            masks and cutoffs affecting the AWSEM class's indicator function calculations;
            they are applied to all structures in the ensemble; burial_in_context also available, but use at your own risk
        
        Returns
        -------
        DecoyEnsemble object, which holds indicator arrays (and gammas???) computed by the AWSEM class.
        """
        # the AWSEM class takes care of the indicator calculation (including masking) for us
        #     AWSEM normally accepts an amino acid sequence argument, but we don't need that here
        #     However, we do need to pass through parameters used to generate the indicator functions
        awsem_obj = AWSEM(next(pdb_structures), expose_indicator_functions=True, repair_pdb=True, **parameters)
        self.N = awsem_obj.N # number of residues
        with open('burial_indicators/burial_indicator_0.npy','ab') as f:
            np.save(f,awsem_obj.burial_indicator)
        with open('direct_indicators/direct_indicator_0.npy','ab') as f:
            np.save(f,awsem_obj.direct_indicator)
        with open('protein_indicators/protein_indicator_0.npy','ab') as f:
            np.save(f,awsem_obj.protein_indicator)
        with open('water_indicators/water_indicator_0.npy','ab') as f:
            np.save(f,awsem_obj.water_indicator)
        with open('electrostatics_indicators/electrostatics_indicator_0.npy','ab') as f:
            if hasattr(awsem_obj, 'electrostatics_indicator'):
                np.save(f,awsem_obj.electrostatics_indicator)
            else:
                np.save(f,None)
        for counter, pdb_structure in enumerate(pdb_structures): # iterate over the rest of the structures without re-initializing the entire AWSEM class
            awsem_obj.pdb_structure = pdb_structure # we can use the pdb_structure setter to update structural 
                                                    # stuff without fully re-initializing the object
            with open(f'burial_indicators/burial_indicator_{counter+1}.npy','ab') as f:
                np.save(f,awsem_obj.burial_indicator)
            with open(f'direct_indicators/direct_indicator_{counter+1}.npy','ab') as f:
                np.save(f,awsem_obj.direct_indicator)
            with open(f'protein_indicators/protein_indicator_{counter+1}.npy','ab') as f:
                np.save(f,awsem_obj.protein_indicator)
            with open(f'water_indicators/water_indicator_{counter+1}.npy','ab') as f:
                np.save(f,awsem_obj.water_indicator)
            with open(f'electrostatics_indicators/electrostatics_indicator_{counter+1}.npy','ab') as f:
                if hasattr(awsem_obj, 'electrostatics_indicator'):
                    np.save(f,awsem_obj.electrostatics_indicator)
                else:
                    np.save(f,None)
        # averages are needed to compute standard deviation
        #     having these be attributes allows them to be easily passed
        #     from the avg method to the std method
        self.avg_burial = None
        self.avg_direct = None
        self.avg_prot = None
        self.avg_wat = None
        self.avg_elec = None
        # and standard deviations can help us check our work 
        # on the covariance matrix calculation
        self.std_burial = None
        self.std_direct = None
        self.std_prot = None
        self.std_wat = None
        self.std_elec = None
        ################################################
        ## Attach gamma parameters from the AWSEM object
        ## Kind of off-topic from my current use of this class
        #self.burial_gamma = awsem_obj.burial_gamma
        #self.direct_gamma = awsem_obj.direct_gamma
        #self.protein_gamma = awsem_obj.protein_gamma
        #self.water_gamma = awsem_obj.water_gamma
        #self.electrostatics_gamma = getattr(awsem_obj, 'electrostatics_gamma', None)
        ################################################
        # this might help with memory
        del awsem_obj
 
    # To manage memory, we need the indicator attributes to be generators (see self.get_indicators).
    # But to ensure that we can iterate over them more than once, we need to 
    #     be able to reinitialize the generators. We accomplish this with properties
    @property
    def burial_indicators(self):
        return self.get_indicators("burial_indicators")
    @property
    def direct_indicators(self):
        return self.get_indicators("direct_indicators")
    @property
    def protein_indicators(self):
        return self.get_indicators("protein_indicators")
    @property
    def water_indicators(self):
        return self.get_indicators("water_indicators")
    @property
    def electrostatics_indicators(self):
        return self.get_indicators("electrostatics_indicators")

    # allows us to process indicators without holding them all in memory
    #    this requires that every method that acts on the indicators iterates over them
    def get_indicators(self, directory):
        # expecting a directory containing numpy files
        for filename in sorted(os.listdir(directory)):
            yield np.load(f"{directory}/{filename}")
        #with open(filename, 'rb') as f:
        #    while True:
        #        try:
        #            yield np.load(f, allow_pickle=True) # allow_pickle=True needed to load None if not electrostatics
        #        except EOFError:
        #            break

    # average indicator functions over all decoys
    # these averages can then be averaged to get the average of all indicator functions over all decoys
    def avg(self):
        # average burial computation from generator
        avg_burial = 0
        counter = 0
        for array in self.burial_indicators:
            counter += 1
            avg_burial += array
        avg_burial /= counter
        self.avg_burial = avg_burial
        # average direct computation from generator
        avg_direct = 0
        counter = 0
        for array in self.direct_indicators:
            counter += 1
            avg_direct += array
        avg_direct /= counter
        self.avg_direct = avg_direct
        # average prot computation from generator
        avg_prot = 0
        counter = 0
        for array in self.protein_indicators:
            counter += 1
            avg_prot += array
        avg_prot /= counter
        self.avg_prot = avg_prot
        # average wat computation from generator
        avg_wat = 0
        counter = 0
        for array in self.water_indicators:
            counter += 1
            avg_wat += array
        avg_wat /= counter
        self.avg_wat = avg_wat
        # average elec computation from generator
        #     if not defined, set to zero, which will have no impact
        if self.electrostatics_indicators == None:
            avg_elec = 0
        else:
            avg_elec = 0
            counter = 0
            for array in self.electrostatics_indicators:
                counter += 1
                avg_elec += array
            avg_elec /= counter
        self.avg_elec = avg_elec
        return self.avg_burial, self.avg_direct, self.avg_prot, self.avg_wat, self.avg_elec

    # standard deviation of each indicator function over all decoys
    # ** averaging these averages
    #    DOES NOT equal 
    #    the variance over all structures of the sum of the indicator functions of each structure **
    def std(self):
        if self.avg_burial is None or self.avg_direct is None or \
           self.avg_prot is None or self.avg_wat is None or \
           self.avg_elec is None:
            self.avg()  # compute averages if not already done
        # std burial computation from generator and previously computed average
        std_burial = 0
        counter = 0
        for array in self.burial_indicators:
            counter += 1
            std_burial += (array - self.avg_burial) ** 2
        std_burial = np.sqrt(std_burial / counter)
        self.std_burial = std_burial
        # std direct computation from generator and previously computed average
        std_direct = 0
        counter = 0
        for array in self.direct_indicators:
            counter += 1
            std_direct += (array - self.avg_direct) ** 2
        std_direct = np.sqrt(std_direct / counter)
        self.std_direct = std_direct
        # std prot computation from generator and previously computed average
        std_prot = 0
        counter = 0
        for array in self.protein_indicators:
            counter += 1
            std_prot += (array - self.avg_prot) ** 2
        std_prot = np.sqrt(std_prot / counter)
        self.std_prot = std_prot
        # std wat computation from generator and previously computed average
        std_wat = 0
        counter = 0
        for array in self.water_indicators:
            counter += 1
            std_wat += (array - self.avg_wat) ** 2
        std_wat = np.sqrt(std_wat / counter)
        self.std_wat = std_wat
        # std elec computation from generator and previously computed average
        std_elec = 0
        counter = 0
        for array in self.electrostatics_indicators:
            counter += 1
            std_elec += (array - self.avg_elec) ** 2
        std_elec = np.sqrt(std_elec / counter)
        self.std_elec = std_elec
        return std_burial, std_direct, std_prot, std_wat, std_elec

    def covariance_matrix(self):
        #
        # compute averages
        if self.avg_burial is None or self.avg_direct is None or \
           self.avg_prot is None or self.avg_wat is None or \
           self.avg_elec is None:
            self.avg()  
        triu_indices = np.triu_indices(self.N, k=1)
        all_avg = np.concatenate([self.avg_burial.flatten(), 
            self.avg_direct[triu_indices].squeeze(), self.avg_prot[triu_indices].squeeze(),
            self.avg_wat[triu_indices].squeeze(), self.avg_elec[triu_indices].squeeze()])
        ex_ey = np.outer(all_avg, all_avg)
        # compute covariances
        number_indicators = 3*self.N + 4*int((self.N**2 - self.N)/2)
        assert ex_ey.shape == (number_indicators, number_indicators), f"ex_ey.shape: {ex_ey.shape}, number_indicators: {number_indicators}"
        exy = np.zeros((number_indicators, number_indicators))
        num_decoys = 0
        #     we want all the burial indicators, 
        #     but only the unique pairwise indicators (no need to double count)
        for b, d, p, w, e in zip(self.burial_indicators, self.direct_indicators,
                                 self.protein_indicators, self.water_indicators,
                                 self.electrostatics_indicators):
            all_decoy = np.concatenate([b.flatten(), d[triu_indices].squeeze(),
                   p[triu_indices].squeeze(), w[triu_indices].squeeze(), e[triu_indices].squeeze()])
            exy += np.outer(all_decoy, all_decoy)
            num_decoys += 1
        exy /= num_decoys
        covariance_matrix = exy - ex_ey
        # check our work
        #variances = np.concatenate([np.triu(self.std_burial).flatten(), np.triu(self.std_direct).flatten(),
        #                            np.triu(self.std_prot).flatten(), np.triu(self.std_wat).flatten(),
        #                            np.triu(self.std_elec).flatten()])**2
        #variances = variances[variances!=0]
        #assert np.allclose(variances, np.diag(covariance_matrix))
        assert np.all(covariance_matrix==covariance_matrix.T)
        assert covariance_matrix.shape == exy.shape == ex_ey.shape
        self.covariance_matrix = covariance_matrix
        return covariance_matrix

    def all_decoy_indicators(self):
        # returns lists of indicator functions for each decoy
        # memory scales with the size of the structure and decoy set
        all_burial = []
        all_direct = []
        all_prot = []
        all_wat = []
        all_elec = []
        for burial, direct, prot, wat, elec in zip(self.burial_indicators,self.direct_indicators,
            self.protein_indicators, self.water_indicators, self.electrostatics_indicators):
            all_burial.append(burial)
            all_direct.append(direct)
            all_prot.append(prot)
            all_wat.append(wat)
            all_elec.append(elec)
        return all_burial, all_direct, all_prot, all_wat, all_elec
