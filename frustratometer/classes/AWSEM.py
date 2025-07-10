import numpy as np
from ..utils import _path
from .. import frustration
from .Frustratometer import Frustratometer
from .Gamma import Gamma
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import Path
from typing import List,Optional,Union,Generator

__all__ = ['AWSEM','AWSEMIndicators','DecoyEnsemble']

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
    membrane_gamma: Union[Path,Gamma] = Field(_path/'data'/'AWSEM_membrane_2015.json', description="File or Gamma object containing the membrane Gamma values (for membrane proteins)")
    eta_switching: int = Field(10, description="Switching distance for the membrane switching function")

    #Electrostatics
    min_sequence_separation_electrostatics: Optional[int] = Field(1, description="Minimum sequence separation for electrostatics calculation.")
    k_electrostatics: float = Field(17.3636, description="Coefficient for electrostatic interactions. (kJ/mol)")
    electrostatics_screening_length: float = Field(10, description="Screening length for electrostatic interactions. (Angstrom)")
    charges: np.array = Field(np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), description="Charge on each residue type")
    # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

class AWSEMBase(Frustratometer):

    #Mapping to DCA
    q = 20
    aa_map_awsem_list = [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18] #A gap has no energy
    aa_map_awsem_x, aa_map_awsem_y = np.meshgrid(aa_map_awsem_list, aa_map_awsem_list, indexing='ij')
    
    def __init__(self, 
                 sequence: str,
                 expose_indicator_functions: bool=False,
                 **parameters)->object:
        """
        Generate AWSEM object

        Parameters
        ----------
        sequence: str
            The amino acid sequence
        expose_indicator_functions: bool
            If set to True, indicator functions of the contact and burial energy terms can be accessed by user.
        
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

        # parse other arguments
        p = AWSEMParameters(**parameters)
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
        else:
            raise ValueError("Gamma parameter must be a path or a Gamma object.")
        self.gamma=gamma
        self.burial_gamma = gamma['Burial'].T
        self.direct_gamma = gamma['Direct'][0]
        self.protein_gamma = gamma['Protein'][0]
        self.water_gamma = gamma['Water'][0]

        # set other attributes
        self.burial_in_context = self.p.burial_in_context
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
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
        # ??????
        self._decoy_fluctuation = {} # don't know what this does
        self.minimally_frustrated_threshold=.78 # this should be a class variable or an argument to __init__

    def setup_model(self):
        self.calculate_indicators()
        self.calculate_energy_and_potts()

    def calculate_indicators(self):
        raise NotImplementedError("Subclasses must this method")

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
        self.potts_model['h'] = self.burial_energy.sum(axis=-1)[:, self.aa_map_awsem_list]
        self.potts_model['J'] = self.contact_energy.sum(axis=0)[:, :, self.aa_map_awsem_x, self.aa_map_awsem_y]
        # Set the gap energy to zero
        self.potts_model['h'][:, 0] = 0
        self.potts_model['J'][:, :, 0, :] = 0
        self.potts_model['J'][:, :, :, 0] = 0
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
                 pdb_structure: object,
                 sequence: str =None,
                 expose_indicator_functions: bool=False,
                 **parameters)->object:
        # assume the user wanted the sequence from the pdb structure if not given
        if not sequence:
            sequence = pdb_structure.sequence
        # load structure-independent parameters and methods
        super().__init__(sequence, expose_indicator_functions, **parameters) 
        # set up strucure
        self.setup_structure(pdb_structure)
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
        with open('my_data.txt','w') as f:
            f.write(f"self.distance_cutoff: {self.distance_cutoff}\n")
            f.write(f"self.sequence_cutoff: {self.sequence_cutoff}\n")
        np.save('my_distance_matrix.npy',self.distance_matrix)
        self.mask = frustration.compute_mask(self.distance_matrix, 
                                             maximum_contact_distance=self.distance_cutoff, 
                                             minimum_sequence_separation = self.sequence_cutoff)
        np.save('my_mask_new.npy',self.mask)
        self.selected_matrix = selected_matrix # we'll need this in the calculate_indicators function 
        self.setup_model()

    def setup_structure(self, pdb_structure):
        # check structure
        selection_CB = pdb_structure.structure.select('name CB or (resname GLY IGL and name CA)')
        resid = selection_CB.getResindices()
        N=len(resid)
        self.resid = resid
        self.N = N
        # set structure-dependent proterties
        self._pdb_structure  = pdb_structure
        self.structure=pdb_structure.structure
        self.chain=pdb_structure.chain
        self.pdb_file=pdb_structure.pdb_file
        self.init_index_shift=pdb_structure.init_index_shift
        self.full_to_aligned_index_dict=pdb_structure.full_to_aligned_index_dict
        self.distance_matrix=pdb_structure.distance_matrix
        self.full_pdb_distance_matrix=pdb_structure.full_pdb_distance_matrix

    @property
    def pdb_structure(self):
        return self._pdb_structure
    @pdb_structure.setter
    def pdb_structure(self,pdb_structure):
        # reset structural attributes
        self.setup_structure(pdb_structure)
        # check that our new structure is compatible with our old one
        if self.N != len(self.sequence):
            raise ValueError("The pdb is incomplete. Try setting 'repair_pdb=True' when constructing the Structure object.")
        self.calculate_indicators()
    def change_conformation(alternative_pdb_structure):
        # this function is an alias for the pdb_structure setter
        self.pdb_structure = alternative_pdb_structure

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
        self.gamma_array=[]
        temp_burial_gamma=self.burial_gamma[self.aa_map_awsem_list]
        temp_burial_gamma[0]=0
        temp_burial_gamma *= -0.5 * self.p.k_contact
        self.gamma_array.append(temp_burial_gamma[:,0])
        self.gamma_array.append(temp_burial_gamma[:,1])
        self.gamma_array.append(temp_burial_gamma[:,2])
        for contact_gamma in [self.direct_gamma, self.protein_gamma, self.water_gamma]:
            temp_gamma = contact_gamma[self.aa_map_awsem_x, self.aa_map_awsem_y].copy()
            temp_gamma[0, :] = 0
            temp_gamma[:, 0] = 0
            temp_gamma *= -0.5 * self.k_contact
            self.gamma_array.append(temp_gamma)
        self.burial_indicator = burial_indicator # probably could get rid of either this or indicators list
        self.direct_indicator = direct_indicator # probably could get rid of either this or indicators list
        self.water_indicator = water_indicator   # probably could get rid of either this or indicators list
        self.protein_indicator = protein_indicator # probably could get rid of either this or indicators list
        if self.p.k_electrostatics != 0:
            electrostatics_indicator = 1 / (self.distance_matrix + 1E-6) * np.exp(-self.distance_matrix / self.p.electrostatics_screening_length) * self.electrostatics_mask
            self.indicators.append(electrostatics_indicator)
            self.electrostatics_indicator = electrostatics_indicator # probably could get rid of either this or indicators list
            temp_gamma = 0.5 * self.p.k_electrostatics * self.charges2[self.aa_map_awsem_x, self.aa_map_awsem_y]
            temp_gamma[0,:]=0
            temp_gamma[:,0]=0
            self.gamma_array.append(temp_gamma)

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
        _AA='ARNDCQEGHILKMFPSTWYV'
        if aa_freq is None:
            seq_index = np.array([_AA.find(aa) for aa in self.sequence])
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
        _AA='ARNDCQEGHILKMFPSTWYV'
        seq_index = np.array([_AA.find(aa) for aa in self.sequence])
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
    

class AWSEMIndicators(AWSEMBase):

    def __init__(self, 
                 burial_indicator: np.ndarray,
                 direct_indicator: np.ndarray,
                 protein_indicator: np.ndarray,
                 water_indicator: np.ndarray,
                 electrostatics_indicator: Union[np.ndarray, None],
                 sequence: str,          # sequence is optional if we initialize from a Structure but not here
                 expose_indicator_functions: bool=False,
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
        
        Returns
        -------
        AWSEMIndicators object

        """
        super().__init__(sequence, expose_indicator_functions, **parameters)
        self.burial_indicator = burial_indicator
        self.direct_indicator = direct_indicator
        self.protein_indicator = protein_indicator
        self.water_indicator = water_indicator
        self.electrostatics_indicator = electrostatics_indicator
        self.sequence_mask_contact = np.full((self.N,self.N), True) 
        self.mask = np.full((self.N,self.N), True) 
        # mask should have been applied when calculating the indicator functions,
        #     so we set it such that no further masking is performed
        self.setup_model()

    def calculate_indicators(self):
        pass # the function was initialized with indicators, so there's nothing to do


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
            they must be the same for all structures; burial_in_context also available, but use at your own risk
        
        Returns
        -------
        DecoyEnsemble object, which holds indicator arrays and gammas computed by the AWSEM class.
        """
        burial_indicators = []
        direct_indicators = []
        protein_indicators = []
        water_indicators = []
        electrostatics_indicators = []
        # the AWSEM class takes care of the indicator calculation (including masking) for us
        #     AWSEM normally accepts an amino acid sequence argument, but we don't need that here
        #     However, we do need to pass through parameters used to generate the indicator functions
        awsem_obj = AWSEM(next(pdb_structures), expose_indicator_functions=True, **parameters)
        for pdb_structure in pdb_structures: 
            awsem_obj.pdb_structure = pdb_structure # we can use the pdb_structure setter to update structural 
                                                    # stuff without fully re-initializing the object
            burial_indicators.append(awsem_obj.burial_indicator)
            direct_indicators.append(awsem_obj.direct_indicator)
            protein_indicators.append(awsem_obj.protein_indicator)
            water_indicators.append(awsem_obj.water_indicator)
            if hasattr(awsem_obj, 'electrostatics_indicator'):
                electrostatics_indicators.append(awsem_obj.electrostatics_indicator)
        # Stack and average indicators, ensuring correct shape for calculate_energy_and_potts
        self.burial_indicator = np.mean(np.stack(burial_indicators, axis=0), axis=0)  # (N, 3)
        self.direct_indicator = np.mean(np.stack(direct_indicators, axis=0), axis=0)  # (N, N, 1, 1)
        self.protein_indicator = np.mean(np.stack(protein_indicators, axis=0), axis=0)  # (N, N, 1, 1)
        self.water_indicator = np.mean(np.stack(water_indicators, axis=0), axis=0)  # (N, N, 1, 1)
        if electrostatics_indicators:
            self.electrostatics_indicator = np.mean(np.stack(electrostatics_indicators, axis=0), axis=0)  # (N, N)
        else:
            self.electrostatics_indicator = None
        # Attach gamma parameters from the AWSEM object
        self.burial_gamma = awsem_obj.burial_gamma
        self.direct_gamma = awsem_obj.direct_gamma
        self.protein_gamma = awsem_obj.protein_gamma
        self.water_gamma = awsem_obj.water_gamma
        self.electrostatics_gamma = getattr(awsem_obj, 'electrostatics_gamma', None)

    def average(self):
        return self.burial_indicator, self.direct_indicator, self.protein_indicator, self.water_indicator, self.electrostatics_indicator