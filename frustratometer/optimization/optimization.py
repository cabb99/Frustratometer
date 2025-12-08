import numpy as np
import numba
import math
import csv
from functools import wraps
from datetime import datetime
import copy

from frustratometer import frustration
from frustratometer.classes import Frustratometer
from frustratometer.classes import Structure
from frustratometer.classes import AWSEM, AWSEMIndicators, DecoyEnsemble, AWSEMVariancePotts
from frustratometer.optimization.EnergyTerm import EnergyTerm
from frustratometer.optimization.inner_product import compute_all_region_means
from frustratometer.optimization.inner_product import build_mean_inner_product_matrix
from frustratometer.optimization.inner_product import diff_mean_inner_product_matrix
from frustratometer.utils.format_time import format_time


_AA = '-ACDEFGHIKLMNPQRSTVWY'

def index_to_sequence(seq_index, alphabet):
    """Converts sequence index array back to a sequence string."""
    seq = ''.join([alphabet[index] for index in seq_index])
    return seq

def sequence_to_index(sequence, alphabet):
    """Converts sequence string to sequence index array."""
    index = np.array([alphabet.find(aa) for aa in sequence])
    if np.any(index == -1):
        raise ValueError("Invalid amino acid in sequence")
    return index

def csv_writer(func):
    """Decorator to write the results to a CSV file."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """Wrapper function to take the csv_filename argument and add the csv_write function."""
        filename = kwargs.pop('csv_filename', None)
        if filename is None:
            filename = f"{func.__name__}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[])
            header_written = False
            columns = []
            def csv_write(step_data):
                nonlocal header_written
                nonlocal columns
                if not header_written:
                    columns = list(step_data.keys())
                    writer.fieldnames = columns
                    writer.writeheader()
                    print(*columns, sep='\t')
                    header_written = True
                
                writer.writerow(step_data)
                print(*[step_data[c] for c in columns], sep='\t')
                if step_data['Step'] % 100 == 0:
                    csvfile.flush()

            return func(self, *args, csv_write=csv_write, **kwargs)
    return wrapper

@numba.njit
def random_seed(seed):
    """Sets the random seed for the numpy random number generator.
    This function is needed to set the seed in a numba compatible way."""
    np.random.seed(seed)

class Zero(EnergyTerm):
    """ Zero energy term for testing purposes. """
    @staticmethod
    def compute_energy(seq_index:np.ndarray):
        return 0.
    
    @staticmethod
    def compute_denergy_mutation(seq_index:np.ndarray, pos:int, aa):
        return 0.
    
    @staticmethod
    def compute_denergy_swap(seq_index:np.ndarray, pos1:int, pos2:int):
        return 0.

class Heterogeneity(EnergyTerm):
    """ Heterogeneity energy term.
        This term calculates the heterogeneity of a sequence using the Shannon entropy.
        The heterogeneity is calculated as the ratio of the number of possible sequences with the same amino acid composition to the total number of sequences.
        The energy is the negative logarithm of the heterogeneity.
     """
    def __init__(self, exact=False, alphabet=_AA, use_numba=True):
        self._use_numba=use_numba
        self.exact=exact
        self.alphabet=alphabet
        self.alphabet_size=len(alphabet)
        self.initialize_functions()

    def initialize_functions(self):
        log_factorial_table=np.log(np.array([math.factorial(i) for i in range(20)],dtype=np.float64))
        alphabet_size=self.alphabet_size
        def stirling_log(n):
            if n < 20:
                return log_factorial_table[n]
            else:
                return n * np.log(n / np.e) + 0.5 * np.log(2 * np.pi * n) + 1.0 / (12 * n)

        stirling_log=self.numbify(stirling_log)

        def heterogeneity_exact(seq_index):
            n = len(seq_index)
            counts = np.bincount(seq_index, minlength=alphabet_size)
            denominator = np.prod(np.array([math.gamma(count+1) for count in counts]))
            het = math.gamma(n+1) / denominator
            return np.log(het)

        def heterogeneity_approximation(seq_index):
            """
            Uses Stirling's approximation to calculate the sequence entropy
            """
            N = len(seq_index)
            counts = np.zeros(alphabet_size, dtype=np.int64)
            
            for val in seq_index:
                counts[val] += 1
                
            log_n_factorial = stirling_log(N)
            log_denominator = sum([stirling_log(count) for count in counts])   
            het = log_n_factorial - log_denominator
            return het
        
        def dheterogeneity(seq_index, pos, aa):
            aa_old_count = np.sum(seq_index == seq_index[pos])
            aa_new_count = np.sum(seq_index == aa)
            return np.log(aa_old_count / (aa_new_count+(seq_index[pos]!=aa)))
        
        if self.exact:
            self.compute_energy=heterogeneity_exact
        else:
            self.compute_energy=heterogeneity_approximation

        self.compute_denergy_mutation=dheterogeneity

class AwsemEnergySelected(EnergyTerm):
    """ AWSEM energy term.
        This term calculates the energy of a sequence using the AWSEM model.
        The energy is calculated from the fields and couplings of the Potts model.
        """
    def __init__(self, model:Frustratometer, alphabet=_AA, use_numba=True, selection=None):
        self._use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        #Create a new alphabet with the unused aminoacids at the end
        self.complete_alphabet=alphabet + ''.join([aa for aa in _AA if aa not in alphabet])
        self.model_h = model.potts_model['h']
        self.model_J = model.potts_model['J']
        self.mask = model.mask
        if selection is None:
            selection = np.arange(len(model.sequence))
        self.selection = selection
        
        # Get the sequence index used the complete alphabet
        self.seq_index = sequence_to_index(model.sequence, self.complete_alphabet)
        if alphabet!=_AA:
            #Reindex using the complete alphabet
            self.reindex_dca=[_AA.index(aa) for aa in self.complete_alphabet]
            self.model_h=self.model_h[:,self.reindex_dca]
            self.model_J=self.model_J[:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.initialize_functions()
    
    def initialize_functions(self):
        mask=self.mask.copy()
        model_h=self.model_h.copy()
        model_J=self.model_J.copy()
        selection=self.selection
        seq_len=len(self.seq_index)
        seq_index_native=self.seq_index

        #The sequence index will need to be modified to create the complete sequence        
        def compute_energy(seq_index: np.array) -> float:
            seq_index_complete=seq_index_native.copy()
            seq_index_complete[selection]=seq_index

            energy_h = 0.0
            energy_J = 0.0

            for i in range(seq_len):
                energy_h -= model_h[i, seq_index_complete[i]]
            
            for i in range(seq_len):
                for j in range(seq_len):
                    aa_i = seq_index_complete[i]
                    aa_j = seq_index_complete[j]
                    energy_J -= model_J[i, j, aa_i, aa_j] * mask[i, j]
            
            total_energy = energy_h + energy_J / 2
            return total_energy

        def compute_denergy_mutation(seq_index: np.ndarray, pos: int, aa_new: int) -> float:
            seq_index_complete=seq_index_native.copy()
            seq_index_complete[selection]=seq_index

            pos=selection[pos]
            aa_old=seq_index_complete[pos]
            

            energy_difference = -model_h[pos, aa_new] + model_h[pos, aa_old]
            energy_difference = -model_h[pos, aa_new] + model_h[pos, aa_old]

            # Initialize j_correction to 0
            j_correction = 0.0

            # Manually iterate over the sequence indices
            for idx in range(seq_len):
                aa_idx = seq_index_complete[idx]  # The amino acid at the current position
                # Accumulate corrections for positions other than the mutated one
                j_correction += model_J[idx, pos, aa_idx, aa_old] * mask[idx, pos]
                j_correction -= model_J[idx, pos, aa_idx, aa_new] * mask[idx, pos]

            # For self-interaction, subtract the old interaction and add the new one
            j_correction -= model_J[pos, pos, aa_old, aa_old] * mask[pos, pos]
            j_correction += model_J[pos, pos, aa_new, aa_new] * mask[pos, pos]

            energy_difference += j_correction

            return energy_difference

        def compute_denergy_swap(seq_index, pos1, pos2):
            seq_index_complete=seq_index_native.copy()
            seq_index_complete[selection]=seq_index

            pos1=selection[pos1]
            pos2=selection[pos2]
            
            aa2 , aa1 = seq_index_complete[pos1],seq_index_complete[pos2]
            
            #Compute fields
            energy_difference = 0
            energy_difference -= (model_h[pos1, aa1] - model_h[pos1, seq_index_complete[pos1]])  # h correction aa1
            energy_difference -= (model_h[pos2, aa2] - model_h[pos2, seq_index_complete[pos2]])  # h correction aa2
            
            #Compute couplings
            j_correction = 0.0
            for pos in range(seq_len):
                aa = seq_index_complete[pos]
                # Corrections for interactions with pos1 and pos2
                j_correction += model_J[pos, pos1, aa, seq_index_complete[pos1]] * mask[pos, pos1]
                j_correction -= model_J[pos, pos1, aa, aa1] * mask[pos, pos1]
                j_correction += model_J[pos, pos2, aa, seq_index_complete[pos2]] * mask[pos, pos2]
                j_correction -= model_J[pos, pos2, aa, aa2] * mask[pos, pos2]

            # J correction, interaction with self aminoacids
            j_correction -= model_J[pos1, pos2, seq_index_complete[pos1], seq_index_complete[pos2]] * mask[pos1, pos2]  # Taken two times
            j_correction += model_J[pos1, pos2, aa1, seq_index_complete[pos2]] * mask[pos1, pos2]  # Correction for incorrect addition in the for loop
            j_correction += model_J[pos1, pos2, seq_index_complete[pos1], aa2] * mask[pos1, pos2]  # Correction for incorrect addition in the for loop
            j_correction -= model_J[pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
            energy_difference += j_correction
            return energy_difference
        
        self.compute_energy=compute_energy
        self.compute_denergy_mutation=compute_denergy_mutation
        self.compute_denergy_swap=compute_denergy_swap

    def regression_test(self):
        expected_energy=self.model.native_energy()
        native_seq_index = np.array([self.seq_index[a] for a in self.selection])
        computed_energy=self.energy(native_seq_index)
        assert np.isclose(expected_energy,computed_energy), f"Expected energy {expected_energy} but got {computed_energy}"

class AwsemEnergy(EnergyTerm):
    """ AWSEM energy term.
        This term calculates the energy of a sequence using the AWSEM model.
        The energy is calculated from the fields and couplings of the Potts model.
        """
    def __init__(self, model:Frustratometer, alphabet=_AA, use_numba=True):
        self._use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.model_h = model.potts_model['h']
        self.model_J = model.potts_model['J']
        self.mask = model.mask

        if alphabet!=_AA:
            self.reindex_dca=[_AA.index(aa) for aa in alphabet]
            self.model_h=self.model_h[:,self.reindex_dca]
            self.model_J=self.model_J[:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.initialize_functions()
    
    def initialize_functions(self):
        mask=self.mask.copy()
        model_h=self.model_h.copy()
        model_J=self.model_J.copy()
        
        def compute_energy(seq_index: np.array) -> float:
            seq_len = len(seq_index)
            energy_h = 0.0
            energy_J = 0.0

            for i in range(seq_len):
                energy_h -= model_h[i, seq_index[i]]
            
            for i in range(seq_len):
                for j in range(seq_len):
                    aa_i = seq_index[i]
                    aa_j = seq_index[j]
                    energy_J -= model_J[i, j, aa_i, aa_j] * mask[i, j]
            
            total_energy = energy_h + energy_J / 2
            #with open('energies.txt','a') as f:
            #    f.write(f"{total_energy}\n")
            return total_energy

        def compute_denergy_mutation(seq_index: np.ndarray, pos: int, aa_new: int) -> float:
            aa_old=seq_index[pos]
            energy_difference = -model_h[pos,aa_new] + model_h[pos,aa_old]

            energy_difference = -model_h[pos, aa_new] + model_h[pos, aa_old]

            # Initialize j_correction to 0
            j_correction = 0.0

            # Manually iterate over the sequence indices
            for idx in range(len(seq_index)):
                aa_idx = seq_index[idx]  # The amino acid at the current position
                # Accumulate corrections for positions other than the mutated one
                j_correction += model_J[idx, pos, aa_idx, aa_old] * mask[idx, pos]
                j_correction -= model_J[idx, pos, aa_idx, aa_new] * mask[idx, pos]

            # For self-interaction, subtract the old interaction and add the new one
            j_correction -= model_J[pos, pos, aa_old, aa_old] * mask[pos, pos]
            j_correction += model_J[pos, pos, aa_new, aa_new] * mask[pos, pos]

            energy_difference += j_correction

            return energy_difference

        def compute_denergy_swap(seq_index, pos1, pos2):
            aa2 , aa1 = seq_index[pos1],seq_index[pos2]
            
            #Compute fields
            energy_difference = 0
            energy_difference -= (model_h[pos1, aa1] - model_h[pos1, seq_index[pos1]])  # h correction aa1
            energy_difference -= (model_h[pos2, aa2] - model_h[pos2, seq_index[pos2]])  # h correction aa2
            
            #Compute couplings
            j_correction = 0.0
            for pos in range(len(seq_index)):
                aa = seq_index[pos]
                # Corrections for interactions with pos1 and pos2
                j_correction += model_J[pos, pos1, aa, seq_index[pos1]] * mask[pos, pos1]
                j_correction -= model_J[pos, pos1, aa, aa1] * mask[pos, pos1]
                j_correction += model_J[pos, pos2, aa, seq_index[pos2]] * mask[pos, pos2]
                j_correction -= model_J[pos, pos2, aa, aa2] * mask[pos, pos2]

            # J correction, interaction with self aminoacids
            j_correction -= model_J[pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
            j_correction += model_J[pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Correction for incorrect addition in the for loop
            j_correction += model_J[pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Correction for incorrect addition in the for loop
            j_correction -= model_J[pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
            energy_difference += j_correction
            return energy_difference
        
        self.compute_energy=compute_energy
        self.compute_denergy_mutation=compute_denergy_mutation
        self.compute_denergy_swap=compute_denergy_swap

    def regression_test(self):
        expected_energy=self.model.native_energy()
        seq_index=sequence_to_index(self.model.sequence,alphabet=self.alphabet)
        energy=self.compute_energy(seq_index)
        assert np.isclose(energy,expected_energy), f"Expected energy {expected_energy} but got {energy}"

#@numba.experimental.jitclass([('_use_numba',numba.float32),('std',numba.float32),('total_energies',numba.float32[:])])
class AwsemStdSlow(EnergyTerm):
    """ Computes the standard deviation of the AWSEM energies of a set of decoy structures
        by computing the energy of each decoy structure and then computing the std of the energies
        """
    def __init__(self, all_burial, all_direct, all_prot, all_wat, all_elec, sequence, 
                       alphabet=_AA, use_numba=True, **parameters):

        self._use_numba=use_numba
        self.alphabet=alphabet

        self.models_h = []
        self.models_J = []
        for burial, direct, prot, wat, elec in zip(all_burial, all_direct, all_prot, all_wat, all_elec):
            model = AWSEMIndicators(burial, direct, prot, wat, elec, sequence, **parameters)
            self.models_h.append(model.potts_model['h'])
            self.models_J.append(model.potts_model['J'])
        self.mask = model.mask # should be the same for all, so we put this outside the loop

        if alphabet!=_AA:
            raise NotImplementedError("Reindex your potts models according to your alphabet")
            self.reindex_dca=[_AA.index(aa) for aa in alphabet]
            self.model_h=self.model_h[:,self.reindex_dca]
            self.model_J=self.model_J[:,:,self.reindex_dca][:,:,:,self.reindex_dca]
       
        self.stds = []
        self.total_energies = []
        self.consider_stds = []
        self.consider_total_energies = []

        self.initialize_functions()
    
    def initialize_functions(self):
        mask=self.mask.copy()
        models_h=self.models_h.copy()
        models_J=self.models_J.copy()
        
        def compute_energy(seq_index: np.array) -> float:
            seq_len = len(seq_index)
            to_append = np.zeros(len(models_h)) # a new array for each seq index, with a length equal to the number of decoys
            for counter, models in enumerate(zip(models_h, models_J)):
                model_h = models[0]
                model_J = models[1]
                energy_h = 0.0
                energy_J = 0.0
                for i in range(seq_len):
                    energy_h -= model_h[i, seq_index[i]]
                for i in range(seq_len):
                    for j in range(seq_len):
                        aa_i = seq_index[i]
                        aa_j = seq_index[j]
                        energy_J -= model_J[i, j, aa_i, aa_j] * mask[i, j]
                decoy_energy = energy_h + energy_J / 2
                to_append[counter] = decoy_energy
            
            if len(self.total_energies) == 0:
                self.total_energies.append(to_append) # this may result in the total_energies list being repeated a few times because compute_energy is called a few times
                self.consider_total_energies.append(copy.deepcopy(to_append))
            else:
                self.total_energies[0] = to_append
                self.consider_total_energies[0] = copy.deepcopy(to_append)
            #breakpoint()
        
            std = to_append.std()
            if len(self.stds) == 0:
                self.stds.append(std) # this may result in the list being too long because compute_energy is called a few times
                self.consider_stds.append(std)
            else:
                self.stds[0] = std
                self.consider_stds[0] = std
            #std = np.array([1,2]).var()#total_energies.var() # doing variance for now because variances are additive
            #self.stds.append(std)
            return std

        def compute_denergy_mutation(seq_index: np.ndarray, pos: int, aa_new: int) -> float:
            aa_old=seq_index[pos]

            for counter, models in enumerate(zip(models_h, models_J)):
                model_h = models[0]
                model_J = models[1]
                #import pdb; pdb.set_trace()
                energy_difference = -model_h[pos,aa_new] + model_h[pos,aa_old]
                # Initialize j_correction to 0
                j_correction = 0.0
                # Manually iterate over the sequence indices
                for idx in range(len(seq_index)):
                    aa_idx = seq_index[idx]  # The amino acid at the current position
                    # Accumulate corrections for positions other than the mutated one
                    j_correction += model_J[idx, pos, aa_idx, aa_old] * mask[idx, pos]
                    j_correction -= model_J[idx, pos, aa_idx, aa_new] * mask[idx, pos]
                # For self-interaction, subtract the old interaction and add the new one
                j_correction -= model_J[pos, pos, aa_old, aa_old] * mask[pos, pos]
                j_correction += model_J[pos, pos, aa_new, aa_new] * mask[pos, pos]
                energy_difference += j_correction
                # our mutation might be rejected, so we don't want to overwrite self.total_energies
                self.consider_total_energies[0][counter] = self.total_energies[0][counter] + energy_difference
                #print(f"energy  difference: {energy_difference}")
            #assert not np.all(np.array(self.total_energies[0])==np.array(self.consider_total_energies[0]))
            #import pdb; pdb.set_trace()
            new_std = self.consider_total_energies[0].std()
            self.consider_stds[0] = new_std  # our mutation might be rejected, so we don't want to overwrite self.stds
            delta_std = new_std - self.stds[0]
            #print(f"mutation: {delta_std}")
            return delta_std

        def compute_denergy_swap(seq_index, pos1, pos2):
            aa2 , aa1 = seq_index[pos1],seq_index[pos2]

            for counter, models in enumerate(zip(models_h, models_J)):
                model_h = models[0]
                model_J = models[1]
                #Compute fields
                energy_difference = 0
                energy_difference -= (model_h[pos1, aa1] - model_h[pos1, seq_index[pos1]])  # h correction aa1
                energy_difference -= (model_h[pos2, aa2] - model_h[pos2, seq_index[pos2]])  # h correction aa2
                
                #Compute couplings
                j_correction = 0.0
                for pos in range(len(seq_index)):
                    aa = seq_index[pos]
                    # Corrections for interactions with pos1 and pos2
                    j_correction += model_J[pos, pos1, aa, seq_index[pos1]] * mask[pos, pos1]
                    j_correction -= model_J[pos, pos1, aa, aa1] * mask[pos, pos1]
                    j_correction += model_J[pos, pos2, aa, seq_index[pos2]] * mask[pos, pos2]
                    j_correction -= model_J[pos, pos2, aa, aa2] * mask[pos, pos2]
    
                # J correction, interaction with self aminoacids
                j_correction -= model_J[pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
                j_correction += model_J[pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Correction for incorrect addition in the for loop
                j_correction += model_J[pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Correction for incorrect addition in the for loop
                j_correction -= model_J[pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
                energy_difference += j_correction
                #import pdb; pdb.set_trace()
                #self.total_energies[counter] += energy_difference
                self.consider_total_energies[0][counter] = self.total_energies[0][counter] + energy_difference
                #print(f"energy  difference: {energy_difference}")
            #breakpoint()
            #import pdb; pdb.set_trace()
            #assert not np.all(np.array(self.total_energies[0])==np.array(self.consider_total_energies[0]))
            new_std = self.consider_total_energies[0].std()
            self.consider_stds[0] = new_std
            delta_std = new_std - self.stds[0]
            #print(f"swap: {delta_std}")
            return delta_std
        
        self.compute_energy=compute_energy
        self.compute_denergy_mutation=compute_denergy_mutation
        self.compute_denergy_swap=compute_denergy_swap

class FourBodyPottsModel(EnergyTerm):
    """ Potts model with 3-body and 4-body terms. 
        This is mainly the same as the AWSEMEnergy class, but 2 important differences:
        1. We don't recognize any mask that may be associated with the "model" input
               that's because this class is intended to evaluate changes in a covariance matrix,
               which should not be masked
        2. By definition, all coefficients are 1: "Energy" = h + J + K + L
               unlike real interactions, the sum over the entire covariance matrix
               isn't double counting, so we don't need to multiply by 1/2
        """
    def __init__(self, model:Frustratometer, alphabet=_AA, use_numba=True):
        self._use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.model_h = model.potts_model['h']
        self.model_J = model.potts_model['J']

class AwsemEnergyAverage(EnergyTerm):   
    def __init__(self, model:Frustratometer, use_numba=True, alphabet=_AA):
        self._use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.reindex_dca=[_AA.index(aa) for aa in alphabet]
        
        assert "indicators" in model.__dict__.keys(), "Indicator functions were not exposed. Initialize AWSEM function with `expose_indicator_functions=True` first."
        self.indicators = model.masked_indicators
        self.alphabet_size=len(alphabet)
        self.model=model
        self.model_h = model.potts_model['h'][:,self.reindex_dca]
        self.model_J = model.potts_model['J'][:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.mask = model.mask
        self.indicators1D=np.array([ind for ind in self.indicators if len(ind.shape)==1])
        self.indicators2D=np.array([ind for ind in self.indicators if len(ind.shape)==2])
        #TODO: Fix the gamma matrix to account for elecrostatics
        self.gamma = np.concatenate([(a[self.reindex_dca].ravel() if len(a.shape)==1 else a[self.reindex_dca][:,self.reindex_dca].ravel()) for a in model.coefficient_lambda_gamma_array])
        
        self.initialize_functions()
    
    def initialize_functions(self):
        indicators1D=self.indicators1D
        indicators2D=self.indicators2D
        len_alphabet=self.alphabet_size
        phi_len= indicators1D.shape[0]*len_alphabet + indicators2D.shape[0]*len_alphabet**2
        gamma=self.gamma
        
        # Precompute the mean of the indicators
        indicator_means=np.zeros(len(indicators1D)+len(indicators2D))
        c=0
        for indicator in indicators1D:
            indicator_means[c]=np.mean(indicator)
            c+=1
        for indicator in indicators2D:
            indicator_means[c]=(np.sum(indicator)-np.sum(np.diag(indicator)))/(len(indicator)**2-len(indicator))
            c+=1
        
        len_indicators1D=len(indicators1D)
        len_indicators2D=len(indicators2D)
        
        def compute_energy(seq_index):
            counts = np.zeros(len_alphabet, dtype=np.int64)
            for val in seq_index:
                counts[val] += 1
            
            # Calculate phi_mean
            phi_mean = np.zeros(len_alphabet*len_indicators1D + len_alphabet**2*len_indicators2D)
            
            # 1D indicators
            c=0
            for i in range(len_indicators1D):
                for j in range(len_alphabet):
                    phi_mean[c] = indicator_means[i] * counts[j]
                    c += 1

            # 2D indicators
            for i in range(len_indicators2D):
                for j in range(len_alphabet):
                    for k in range(len_alphabet):
                        t=1 if j==k else 0
                        phi_mean[c] = indicator_means[i+ len_indicators1D] * counts[j] * (counts[k] - t)
                        c += 1

            # Calculate energy
            energy = 0
            for i in range(phi_len):
                energy += gamma[i] * phi_mean[i]
            
            return energy
        
        def denergy_mutation(seq_index, pos, aa):
            counts = np.zeros(len_alphabet, dtype=np.int64)
            for val in seq_index:
                counts[val] += 1
            aa_old=seq_index[pos]
            if aa_old==aa:
                return 0.
            
            # Calculate phi_mean

            dphi_mean = np.zeros(len_alphabet*len_indicators1D + len_alphabet**2*len_indicators2D)
            
            # 1D indicators
            for i in range(len_indicators1D):
                dphi_mean[i*len_alphabet + aa_old] -= indicator_means[i]
                dphi_mean[i*len_alphabet + aa] += indicator_means[i]

            offset = len_alphabet*len_indicators1D
            for i in range(len_indicators2D):
                for j in range(len_alphabet):
                    k=aa_old
                    if j==k:
                        dphi_mean[offset + i*len_alphabet**2 + j*len_alphabet + k] -= 2 * indicator_means[i + len_indicators1D] *  (counts[j]-1)
                    elif j==aa:
                        dphi_mean[offset + i*len_alphabet**2 + j*len_alphabet + k] += indicator_means[i+ len_indicators1D] * (counts[k] - counts[j] -1)
                    else:
                        dphi_mean[offset + i*len_alphabet**2 + j*len_alphabet + k] -= indicator_means[i + len_indicators1D] * counts[j]
                        dphi_mean[offset + i*len_alphabet**2 + k*len_alphabet + j] -= indicator_means[i + len_indicators1D] * counts[j]                    
                    k=aa
                    if j==k:
                        dphi_mean[offset + i*len_alphabet**2 + j*len_alphabet + k] += 2 * indicator_means[i + len_indicators1D] *  counts[j]
                    elif j==aa_old:
                        dphi_mean[offset + i*len_alphabet**2 + j*len_alphabet + k] += indicator_means[i+ len_indicators1D] * (counts[j] - counts[k] -1)
                    else:
                        dphi_mean[offset + i*len_alphabet**2 + j*len_alphabet + k] += indicator_means[i + len_indicators1D] * counts[j]
                        dphi_mean[offset + i*len_alphabet**2 + k*len_alphabet + j] += indicator_means[i + len_indicators1D] * counts[j]
            
            # Calculate energy
            denergy = 0
            for i in range(phi_len):
                denergy += gamma[i] * dphi_mean[i]
            
            return denergy
        
        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation

        awsem_energy = AwsemEnergy(use_numba=self.use_numba, model=self.model, alphabet=self.alphabet).energy_function

        def compute_energy_sample(seq_index,n_decoys=100000):
            """ Function to compute the variance of the energy of permutations of a sequence using random shuffling.
                This function is much faster than compute_energy_permutation but is an approximation"""
            energies=np.zeros(n_decoys)
            shuffled_index=seq_index.copy()
            for i in numba.prange(n_decoys):
                energies[i]=awsem_energy(shuffled_index[np.random.permutation(len(shuffled_index))])
            return np.mean(energies)

        def compute_energy_permutation(seq_index):
            """ Function to compute the variance of the energy of all permutations of a sequence 
                Caution: This function is very slow for normal sequences """
            from itertools import permutations
            decoy_sequences = np.array(list(permutations(seq_index)))
            energies=np.zeros(len(decoy_sequences))
            for i in numba.prange(len(decoy_sequences)):
                energies[i]=awsem_energy(decoy_sequences[i])
            return np.mean(energies)
        
        self.compute_energy_sample=self.numbify(compute_energy_sample,parallel=True)
        self.compute_energy_permutation=compute_energy_permutation

    def regression_test(self, seq_index):
        expected_energy=self.compute_energy_permutation(seq_index)
        energy=self.compute_energy(seq_index)
        assert np.isclose(energy,expected_energy), f"Expected energy {expected_energy} but got {energy}"

class PairEnergyAverage(EnergyTerm):   
    """
    Computes the average pairwise energy for a given sequence.
    This class is designed to compute the average pairwise energy of a sequence
    using the AWSEM model. It calculates the energy contributions from pairwise interactions
    between amino acids in the sequence, averaged over all possible pairs.
    """
    def __init__(self, model:Frustratometer, use_numba=True, alphabet=_AA):
        self._use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.reindex_dca=[_AA.index(aa) for aa in alphabet]
        assert "indicators" in model.__dict__.keys(), "Indicator functions were not exposed. Initialize AWSEM function with `expose_indicator_functions=True` first."
        self.indicators = model.masked_indicators
        self.alphabet_size=len(alphabet)
        self.model_h = model.potts_model['h'][:,self.reindex_dca]
        self.model_J = model.potts_model['J'][:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.mask = model.mask
        self.gamma = np.concatenate([(a[self.reindex_dca].ravel() if len(a.shape)==1 else a[self.reindex_dca][:,self.reindex_dca].ravel()) for a in model.coefficient_lambda_gamma_array])
        self.initialize_functions()
    
    def initialize_functions(self):
        len_alphabet=self.alphabet_size

        distances = np.triu(self.model.distance_matrix)
        ###########################################################################################
        distances = distances[(distances<self.model.distance_cutoff_contact) & (distances>0)] # USE THIS NORMALLY
        #distances = distances[distances>0]  # USE THIS FOR TESTING THING WHERE WE NEED ALL PAIRS
        ###########################################################################################
        len_distances = len(distances)

        rho_b = np.expand_dims(self.model.rho_r, 1) #(n,1)
        rho1 = np.expand_dims(self.model.rho_r, 0) #(1,n)
        rho2 = np.expand_dims(self.model.rho_r, 1) #(n,1)

        sigma_water = 0.25 * (1 - np.tanh(self.model.eta_sigma * (rho1 - self.model.rho_0))) * (1 - np.tanh(self.model.eta_sigma * (rho2 - self.model.rho_0))) #(n,n)
        sigma_protein = 1 - sigma_water #(n,n)
            
        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.model.eta * (distances - self.model.r_min))) * (1 + np.tanh(self.model.eta * (self.model.r_max - distances))) # (c,)
        thetaII = 0.25 * (1 + np.tanh(self.model.eta * (distances - self.model.r_minII))) * (1 + np.tanh(self.model.eta * (self.model.r_maxII - distances))) #(c,)
        burial_indicator = np.tanh(self.model.burial_kappa * (rho_b - self.model.burial_ro_min)) + np.tanh(self.model.burial_kappa * (self.model.burial_ro_max - rho_b)) #(n,3)
                        # gap has 0 charge
                    #     gap, A,C,D, E, F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
        charges = np.array([0, 0,0,-1,-1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0])
        charges = charges[self.reindex_dca]  # remove unused gap, C, and P
        electrostatics_indicator = np.exp(-distances / self.model.electrostatics_screening_length) / distances

        N = self.model.N
        k_contact = self.model.k_contact
        burial_gamma = self.model.burial_gamma[self.model.aa_map_awsem_list][self.reindex_dca] 
        direct_gamma = self.model.direct_gamma[self.model.aa_map_awsem_x, self.model.aa_map_awsem_y][self.reindex_dca][:,self.reindex_dca]
        water_gamma = self.model.water_gamma[self.model.aa_map_awsem_x, self.model.aa_map_awsem_y][self.reindex_dca][:,self.reindex_dca]
        protein_gamma = self.model.protein_gamma[self.model.aa_map_awsem_x, self.model.aa_map_awsem_y][self.reindex_dca][:,self.reindex_dca]
        k_contact = self.model.k_contact
        k_electrostatics = self.model.k_electrostatics
        mask = self.mask
        sequence_mask_contact = self.model.sequence_mask_contact
        assert np.all(mask==mask.T), "Mask should be symmetric"
        assert mask.shape == (N, N), f"Mask shape {mask.shape} does not match expected shape {(N, N)}"

        n_decoys=9000
        
        # these lines used for the random sampling and analytic calculation
        foo = np.triu(self.model.distance_matrix)
        sigma_water = sigma_water[(foo<self.model.distance_cutoff_contact) & (foo>0)]
        sigma_protein = sigma_protein[(foo<self.model.distance_cutoff_contact) & (foo>0)]

        def compute_energy(seq_index):
            # adapted from AWSEM.compute_configurational_decoy_statistics,
            #     modified to be numba-friendly
            
            # analytic calculation
            aa_freq = np.array([(seq_index == i).sum() for i in range(len_alphabet)])  # frequency of each amino acid in the sequence
           
            temp = burial_gamma * aa_freq[:, np.newaxis] # (20,3) * (20,1) -> (20,3)
            scaled_burial_gamma = np.zeros((3,))  # burial gamma is a 3D vector, one for each burial indicator (low, med, high)
            for counter in range(temp.shape[0]):
                scaled_burial_gamma += temp[counter]
            scaled_burial_gamma /= temp.shape[0] # average burial gammas, weighted by amino acid frequencies
            #burial_energy = np.average(-1 * k_contact * scaled_burial_gamma * burial_indicator)
            avg_burial_indicator = np.zeros((3,))
            for counter in range(burial_indicator.shape[0]):
                avg_burial_indicator += burial_indicator[counter]
            avg_burial_indicator /= burial_indicator.shape[0] # average burial indicator
            burial_energy = -1* k_contact * avg_burial_indicator * scaled_burial_gamma
            burial_energy = (burial_energy[0] + burial_energy[1] + burial_energy[2])/(N-1)#*N  #/(N-1)  # sum over the three burial indicators
            #breakpoint()
            #assert type(burial_energy) == float
            # direct, water-mediated, and protein-mediated contact energies
            direct = np.average(theta) * np.average(direct_gamma * np.outer(aa_freq, aa_freq))
            #assert type(direct) == float
            water_mediated = np.average(thetaII*sigma_water) * np.average(water_gamma * np.outer(aa_freq, aa_freq))
            #assert type(water_mediated) == float
            protein_mediated = np.average(thetaII*sigma_protein) * np.average(protein_gamma * np.outer(aa_freq, aa_freq))
            #assert type(protein_mediated) == float
            #contact_energy = -k_contact * (direct*len(theta) + (water_mediated+protein_mediated)*len(thetaII)) # multiply by number of contacts
            contact_energy = -k_contact * (direct + (water_mediated+protein_mediated)) # multiply by number of contacts
            #electrostatics_energy = k_electrostatics * np.average(electrostatics_indicator) * np.average(np.outer(aa_freq, aa_freq)*charges[:, np.newaxis]*charges[np.newaxis, :]) * len(electrostatics_indicator) # multiply by number of contacts
            electrostatics_energy = k_electrostatics * np.average(electrostatics_indicator) * np.average(np.outer(aa_freq, aa_freq)*charges[:, np.newaxis]*charges[np.newaxis, :])
            #assert type(electrostatics_energy) == float
            mean_decoy_energy = burial_energy + contact_energy + electrostatics_energy
            #import pdb; pdb.set_trace()
            
            """# constructing the distribution by sampling, then computing the average
            decoy_energies=np.zeros(n_decoys)
            for i in range(n_decoys):
                c=np.random.randint(0,len_distances)
                n1=np.random.randint(0,N)
                n2=np.random.randint(0,N)
                qi1=np.random.randint(0,N)
                qi2=np.random.randint(0,N)
                q1=seq_index[qi1]
                q2=seq_index[qi2]
    
                burial_energy1 = (-0.5 * k_contact * burial_gamma[q1] * burial_indicator[n1]).sum(axis=0)
                burial_energy2 = (-0.5 * k_contact * burial_gamma[q2] * burial_indicator[n2]).sum(axis=0)
                burial_energy = (burial_energy1+burial_energy2)/(N-1) # normalize because double counting carlos thing

                direct = theta[c] * direct_gamma[q1, q2]
                water_mediated = sigma_water[n1,n2] * thetaII[c] * water_gamma[q1,q2]
                protein_mediated = sigma_protein[n1,n2] * thetaII[c] * protein_gamma[q1,q2]
                contact_energy = -k_contact * (direct+water_mediated+protein_mediated)
                electrostatics_energy = k_electrostatics * electrostatics_indicator[c]*charges[q1]*charges[q2]
    
                decoy_energies[i]=(burial_energy+contact_energy+electrostatics_energy)
            mean_decoy_energy = np.mean(decoy_energies)
            #std_decoy_energy = np.std(decoy_energies)
            """
            """# checking that these energy functions are able to compute the total energy of the sequence
            assert len(distances) == (N**2-N)/2, f"len(distances): {len(distances)} != (N**2-N)/2: {(N**2-N)/2}"
            decoy_energies = np.zeros((len(seq_index)**2 - len(seq_index))//2)
            index = 0
            #breakpoint()
            for i in range(len(seq_index)):
                aa1 = seq_index[i]
                for j in range(i+1, len(seq_index)):
                    aa2 = seq_index[j]

                    burial_energy1 = (-0.5 * k_contact * burial_gamma[aa1] * burial_indicator[i]).sum(axis=0)
                    burial_energy2 = (-0.5 * k_contact * burial_gamma[aa2] * burial_indicator[j]).sum(axis=0)
                    burial_energy = (burial_energy1 + burial_energy2) / ((N - 1)) #/ 2)

                    direct = theta[index] * direct_gamma[aa1, aa2]
                    water_mediated = sigma_water[i, j] * thetaII[index] * water_gamma[aa1, aa2]
                    protein_mediated = sigma_protein[i, j] * thetaII[index] * protein_gamma[aa1, aa2]
                    contact_energy = -k_contact * (direct+water_mediated+protein_mediated)*mask[i, j]*sequence_mask_contact[i, j]
                    electrostatics_energy = k_electrostatics * electrostatics_indicator[index]*charges[aa1]*charges[aa2]*mask[i,j]
                    decoy_energies[index] = contact_energy+burial_energy+electrostatics_energy#(contact_energy+electrostatics_energy)#(burial_energy + contact_energy + electrostatics_energy)
                    index += 1
            mean_decoy_energy = np.sum(decoy_energies) # for testing, we return the total energy, not the average
            """

            #import pdb; pdb.set_trace()
            return mean_decoy_energy#, std_decoy_energy
        
        compute_energy_numba = self.numbify(compute_energy)

        def denergy_mutation(seq_index, pos, aa):
            seq_index_new = seq_index.copy()
            seq_index_new[pos] = aa
            return compute_energy_numba(seq_index_new) - compute_energy_numba(seq_index)
        
        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation

    def regression_test(self, seq_index):
        raise NotImplementedError("sorry") 

class PairEnergyStd(EnergyTerm):   
    def __init__(self, model:Frustratometer, use_numba=True, alphabet=_AA):
        self._use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.reindex_dca=[_AA.index(aa) for aa in alphabet]
        assert "indicators" in model.__dict__.keys(), "Indicator functions were not exposed. Initialize AWSEM function with `expose_indicator_functions=True` first."
        self.indicators = model.masked_indicators
        self.alphabet_size=len(alphabet)
        self.model_h = model.potts_model['h'][:,self.reindex_dca]
        self.model_J = model.potts_model['J'][:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.mask = model.mask
        self.gamma = np.concatenate([(a[self.reindex_dca].ravel() if len(a.shape)==1 else a[self.reindex_dca][:,self.reindex_dca].ravel()) for a in model.coefficient_lambda_gamma_array])
        self.initialize_functions()
    
    def initialize_functions(self):
        len_alphabet=self.alphabet_size

        distances = np.triu(self.model.distance_matrix)
        ###########################################################################################
        distances = distances[(distances<self.model.distance_cutoff_contact) & (distances>0)] # USE THIS NORMALLY
        #distances = distances[distances>0]  # USE THIS FOR TESTING THING WHERE WE NEED ALL PAIRS
        ###########################################################################################
        len_distances = len(distances)

        rho_b = np.expand_dims(self.model.rho_r, 1) #(n,1)
        rho1 = np.expand_dims(self.model.rho_r, 0) #(1,n)
        rho2 = np.expand_dims(self.model.rho_r, 1) #(n,1)

        sigma_water = 0.25 * (1 - np.tanh(self.model.eta_sigma * (rho1 - self.model.rho_0))) * (1 - np.tanh(self.model.eta_sigma * (rho2 - self.model.rho_0))) #(n,n)
        sigma_protein = 1 - sigma_water #(n,n)
            
        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.model.eta * (distances - self.model.r_min))) * (1 + np.tanh(self.model.eta * (self.model.r_max - distances))) # (c,)
        thetaII = 0.25 * (1 + np.tanh(self.model.eta * (distances - self.model.r_minII))) * (1 + np.tanh(self.model.eta * (self.model.r_maxII - distances))) #(c,)
        burial_indicator = np.tanh(self.model.burial_kappa * (rho_b - self.model.burial_ro_min)) + np.tanh(self.model.burial_kappa * (self.model.burial_ro_max - rho_b)) #(n,3)
                        # gap has 0 charge
                    #     gap, A,C,D, E, F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y
        charges = np.array([0, 0,0,-1,-1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0])
        charges = charges[self.reindex_dca]  # remove unused gap, C, and P
        electrostatics_indicator = np.exp(-distances / self.model.electrostatics_screening_length) / distances

        N = self.model.N
        k_contact = self.model.k_contact
        burial_gamma = self.model.burial_gamma[self.model.aa_map_awsem_list][self.reindex_dca] 
        direct_gamma = self.model.direct_gamma[self.model.aa_map_awsem_x, self.model.aa_map_awsem_y][self.reindex_dca][:,self.reindex_dca]
        water_gamma = self.model.water_gamma[self.model.aa_map_awsem_x, self.model.aa_map_awsem_y][self.reindex_dca][:,self.reindex_dca]
        protein_gamma = self.model.protein_gamma[self.model.aa_map_awsem_x, self.model.aa_map_awsem_y][self.reindex_dca][:,self.reindex_dca]
        k_contact = self.model.k_contact
        k_electrostatics = self.model.k_electrostatics
        mask = self.mask
        sequence_mask_contact = self.model.sequence_mask_contact
        assert np.all(mask==mask.T), "Mask should be symmetric"
        assert mask.shape == (N, N), f"Mask shape {mask.shape} does not match expected shape {(N, N)}"

        n_decoys=9000
        
        #foo = np.triu(self.model.distance_matrix)
        #sigma_water = sigma_water[(foo<self.model.distance_cutoff_contact) & (foo>0)]
        #sigma_protein = sigma_protein[(foo<self.model.distance_cutoff_contact) & (foo>0)]

        def compute_energy(seq_index):
            # constructing the distribution by sampling, then computing the average
            decoy_energies=np.zeros(n_decoys)
            for i in range(n_decoys):
                c=np.random.randint(0,len_distances)
                n1=np.random.randint(0,N)
                n2=np.random.randint(0,N)
                qi1=np.random.randint(0,N)
                qi2=np.random.randint(0,N)
                q1=seq_index[qi1]
                q2=seq_index[qi2]
    
                burial_energy1 = (-0.5 * k_contact * burial_gamma[q1] * burial_indicator[n1]).sum(axis=0)
                burial_energy2 = (-0.5 * k_contact * burial_gamma[q2] * burial_indicator[n2]).sum(axis=0)
                burial_energy = (burial_energy1+burial_energy2)/(N-1) # normalize because double counting carlos thing

                direct = theta[c] * direct_gamma[q1, q2]
                water_mediated = sigma_water[n1,n2] * thetaII[c] * water_gamma[q1,q2]
                protein_mediated = sigma_protein[n1,n2] * thetaII[c] * protein_gamma[q1,q2]
                contact_energy = -k_contact * (direct+water_mediated+protein_mediated)
                electrostatics_energy = k_electrostatics * electrostatics_indicator[c]*charges[q1]*charges[q2]
    
                decoy_energies[i]=(burial_energy+contact_energy+electrostatics_energy)
                
            #mean_decoy_energy = np.mean(decoy_energies)
            std_decoy_energy = np.std(decoy_energies)
            return 1#std_decoy_energy
        
        compute_energy_numba = self.numbify(compute_energy)

        def denergy_mutation(seq_index, pos, aa):
            seq_index_new = seq_index.copy()
            seq_index_new[pos] = aa
            return compute_energy_numba(seq_index_new) - compute_energy_numba(seq_index)
        
        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation

    def regression_test(self, seq_index):
        raise NotImplementedError("sorry") 


class AwsemEnergyVariance(EnergyTerm):   
    def __init__(self, model:Frustratometer, use_numba=True, alphabet=_AA):
        self._use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.reindex_dca=[_AA.index(aa) for aa in alphabet]

        assert "indicators" in model.__dict__.keys(), "Indicator functions were not exposed. Initialize AWSEM function with `expose_indicator_functions=True` first."
        self.indicators = model.masked_indicators
        self.alphabet_size=len(alphabet)
        self.model=model
        self.model_h = model.potts_model['h'][:,self.reindex_dca]
        self.model_J = model.potts_model['J'][:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.mask = model.mask
        self.indicators1D=np.array([ind for ind in self.indicators if len(ind.shape)==1])
        self.indicators2D=np.array([ind for ind in self.indicators if len(ind.shape)==2])
        #TODO: Fix the gamma matrix to account for elecrostatics
        self.gamma = np.concatenate([(a[self.reindex_dca].ravel() if len(a.shape)==1 else a[self.reindex_dca][:,self.reindex_dca].ravel()) for a in model.coefficient_lambda_gamma_array])
        
        self.initialize_functions()
    
    def initialize_functions(self):
        indicators1D=self.indicators1D
        indicators2D=self.indicators2D
        len_alphabet=self.alphabet_size
        phi_len= indicators1D.shape[0]*len_alphabet + indicators2D.shape[0]*len_alphabet**2
        gamma=self.gamma
        
        # Precompute the mean of the indicators
        indicator_means=np.zeros(len(indicators1D)+len(indicators2D))
        c=0
        for indicator in indicators1D:
            indicator_means[c]=np.mean(indicator)
            c+=1
        for indicator in indicators2D:
            indicator_means[c]=(np.sum(indicator)-np.sum(np.diag(indicator)))/(len(indicator)**2-len(indicator))
            c+=1
        
        len_indicators1D=len(indicators1D)
        len_indicators2D=len(indicators2D)
        
        region_means=compute_all_region_means(indicators1D,indicators2D)
        # indicator_means*=0 # Set indicator means to zero to check if the problem is with the mean phi or the inner product matrix
        
        def compute_energy(seq_index):
            counts = np.zeros(len_alphabet, dtype=np.int64)
            for val in seq_index:
                counts[val] += 1
            
            # Calculate phi_mean
            phi_mean = np.zeros(len_alphabet*len_indicators1D + len_alphabet**2*len_indicators2D)
            
            # 1D indicators
            c=0
            for i in range(len_indicators1D):
                for j in range(len_alphabet):
                    phi_mean[c] = indicator_means[i] * counts[j]
                    c += 1

            # 2D indicators
            for i in range(len_indicators2D):
                for j in range(len_alphabet):
                    for k in range(len_alphabet):
                        t=1 if j==k else 0
                        phi_mean[c] = indicator_means[i+ len_indicators1D] * counts[j] * (counts[k] - t)
                        c += 1

            B = build_mean_inner_product_matrix(counts,indicators1D,indicators2D,region_means)
            energy=0
            for i in range(phi_len):
                for j in range(phi_len):
                    energy += gamma[i] * (B[i,j] - phi_mean[i]*phi_mean[j]) * gamma[j]
            return energy
        
        def denergy_mutation(seq_index, pos, aa):
            counts = np.zeros(len_alphabet, dtype=np.int64)
            for val in seq_index:
                counts[val] += 1
            aa_old=seq_index[pos]
            if aa_old==aa:
                return 0.
            
            counts_new=counts.copy()
            counts_new[aa]+=1
            counts_new[aa_old]-=1
            
            #Calculate phi_mean_old
            phi_mean_old = np.zeros(len_alphabet*len_indicators1D + len_alphabet**2*len_indicators2D)
            # 1D indicators
            c=0
            for i in range(len_indicators1D):
                for j in range(len_alphabet):
                    phi_mean_old[c] = indicator_means[i] * counts[j]
                    c += 1
            # 2D indicators
            for i in range(len_indicators2D):
                for j in range(len_alphabet):
                    for k in range(len_alphabet):
                        t=1 if j==k else 0
                        phi_mean_old[c] = indicator_means[i+ len_indicators1D] * counts[j] * (counts[k] - t)
                        c += 1
            
            #Calculate phi_mean_new
            phi_mean_new = np.zeros(len_alphabet*len_indicators1D + len_alphabet**2*len_indicators2D)
            # 1D indicators
            c=0
            for i in range(len_indicators1D):
                for j in range(len_alphabet):
                    phi_mean_new[c] = indicator_means[i] * counts_new[j]
                    c += 1
            # 2D indicators
            for i in range(len_indicators2D):
                for j in range(len_alphabet):
                    for k in range(len_alphabet):
                        t=1 if j==k else 0
                        phi_mean_new[c] = indicator_means[i+ len_indicators1D] * counts_new[j] * (counts_new[k] - t)
                        c += 1


            # # Calculate phi_mean
            # dphi_mean2 = np.zeros((phi_len,phi_len))
            # for i in range(phi_len):
            #     for j in range(phi_len):
            #         if i==aa_old:
            #             di=-1
            #         elif i==aa:
            #             di=1
            #         if j==aa_old:
            #             dj=-1
            #         elif j==aa:
            #             dj=1
            #         if di!=0 and dj!=0:
            #             indicator_i = i // len_alphabet if i < len_alphabet*len_indicators1D else len_indicators1D + (i - len_alphabet*len_indicators1D) // len_alphabet**2
            #             indicator_j = j // len_alphabet if j < len_alphabet*len_indicators1D else len_indicators1D + (j - len_alphabet*len_indicators1D) // len_alphabet**2
            #             ii=i % len_alphabet if i < len_alphabet*len_indicators1D else (i - len_alphabet*len_indicators1D) % len_alphabet**2
            #             jj=j % len_alphabet if j < len_alphabet*len_indicators1D else (j - len_alphabet*len_indicators1D) % len_alphabet**2
            #             countsi = counts[ii] if i < len_alphabet*len_indicators1D else counts[ii // len_alphabet]*counts[ii % len_alphabet]
            #             countsj = counts[jj] if j < len_alphabet*len_indicators1D else counts[jj // len_alphabet]*counts[jj % len_alphabet]
            #             indicator_i_mean = indicator_means[indicator_i]
            #             indicator_j_mean = indicator_means[indicator_j]
            #             dphi_mean2[i,j] = indicator_i_mean * (countsi + di) + indicator_j_mean * (countsj + dj) - indicator_i_mean * indicator_j_mean * countsi * countsj

            dB = diff_mean_inner_product_matrix(aa_old, aa, counts,indicators1D,indicators2D,region_means)
            # Calculate energy
            denergy=0
            for i in range(phi_len):
                for j in range(phi_len):
                    denergy += gamma[i] * (dB[i,j] - (phi_mean_new[i]*phi_mean_new[j]-phi_mean_old[i]*phi_mean_old[j])) * gamma[j]
                    
            return denergy
            

        
        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation

        awsem_energy = AwsemEnergy(use_numba=self.use_numba, model=self.model, alphabet=self.alphabet).energy_function

        def compute_energy_sample(seq_index,n_decoys=100000):
            """ Function to compute the variance of the energy of permutations of a sequence using random shuffling.
                This function is much faster than compute_energy_permutation but is an approximation"""
            energies=np.zeros(n_decoys)
            shuffled_index=seq_index.copy()
            for i in numba.prange(n_decoys):
                energies[i]=awsem_energy(shuffled_index[np.random.permutation(len(shuffled_index))])
            return np.var(energies)

        def compute_energy_permutation(seq_index):
            """ Function to compute the variance of the energy of all permutations of a sequence 
                Caution: This function is very slow for normal sequences """
            from itertools import permutations
            decoy_sequences = np.array(list(permutations(seq_index)))
            energies=np.zeros(len(decoy_sequences))
            for i in numba.prange(len(decoy_sequences)):
                energies[i]=awsem_energy(decoy_sequences[i])
            return np.var(energies)
        
        self.compute_energy_sample=self.numbify(compute_energy_sample,parallel=True)
        self.compute_energy_permutation=compute_energy_permutation

    def regression_test(self, seq_index):
        expected_energy=self.compute_energy_permutation(seq_index)
        energy=self.compute_energy(seq_index)
        assert np.isclose(energy,expected_energy), f"Expected energy {expected_energy} but got {energy}"


class AwsemEnergyStd(EnergyTerm):   
    def __init__(self, model:Frustratometer, use_numba=True, alphabet=_AA, n_decoys=None):
        self._use_numba=use_numba
        self.model=model
        self.alphabet=alphabet
        self.reindex_dca=[_AA.index(aa) for aa in alphabet]

        self.n_decoys=n_decoys
        
        assert "indicators" in model.__dict__.keys(), "Indicator functions were not exposed. Initialize AWSEM function with `expose_indicator_functions=True` first."
        self.indicators = model.masked_indicators
        self.alphabet_size=len(alphabet)
        self.model=model
        self.model_h = model.potts_model['h'][:,self.reindex_dca]
        self.model_J = model.potts_model['J'][:,:,self.reindex_dca][:,:,:,self.reindex_dca]
        self.mask = model.mask
        self.indicators1D=np.array([ind for ind in self.indicators if len(ind.shape)==1])
        self.indicators2D=np.array([ind for ind in self.indicators if len(ind.shape)==2])
        #TODO: Fix the gamma matrix to account for elecrostatics
        self.gamma = np.concatenate([(a[self.reindex_dca].ravel() if len(a.shape)==1 else a[self.reindex_dca][:,self.reindex_dca].ravel()) for a in model.coefficient_lambda_gamma_array])
                
        self.initialize_functions()
    
    def initialize_functions(self):
        indicators1D=self.indicators1D
        indicators2D=self.indicators2D
        len_alphabet=self.alphabet_size
        phi_len= indicators1D.shape[0]*len_alphabet + indicators2D.shape[0]*len_alphabet**2
        gamma=self.gamma
        rng = np.random.default_rng()

        # Precompute the mean of the indicators
        indicator_means=np.zeros(len(indicators1D)+len(indicators2D))
        c=0
        for indicator in indicators1D:
            indicator_means[c]=np.mean(indicator)
            c+=1
        for indicator in indicators2D:
            indicator_means[c]=(np.sum(indicator)-np.sum(np.diag(indicator)))/(len(indicator)**2-len(indicator))
            c+=1
        
        len_indicators1D=len(indicators1D)
        len_indicators2D=len(indicators2D)
        
        region_means=compute_all_region_means(indicators1D,indicators2D)
        # indicator_means*=0 # Set indicator means to zero to check if the problem is with the mean phi or the inner product matrix
        awsem_energy = AwsemEnergy(use_numba=self.use_numba, model=self.model, alphabet=self.alphabet).energy_function

        if self.n_decoys:
            n_decoys=self.n_decoys
            def compute_energy(seq_index):
                """ Function to compute the variance of the energy of permutations of a sequence using random shuffling.
                    This function is much faster than compute_energy_permutation but is an approximation"""
                energies=np.zeros(n_decoys)
                shuffled_index=seq_index.copy()               
                for _  in numba.prange(20):
                    to_replace = rng.integers(low=0,high=len(seq_index))
                    shuffled_index[to_replace] = rng.integers(low=0,high=len_alphabet)
                for i in numba.prange(n_decoys):
                    energies[i]=awsem_energy(shuffled_index[np.random.permutation(len(shuffled_index))])
                return np.std(energies)
        else:
            def compute_energy(seq_index):
                counts = np.zeros(len_alphabet, dtype=np.int64)
                for val in seq_index:
                    counts[val] += 1
                
                # Calculate phi_mean
                phi_mean = np.zeros(len_alphabet*len_indicators1D + len_alphabet**2*len_indicators2D)
                
                # 1D indicators
                c=0
                for i in range(len_indicators1D):
                    for j in range(len_alphabet):
                        phi_mean[c] = indicator_means[i] * counts[j]
                        c += 1

                # 2D indicators
                for i in range(len_indicators2D):
                    for j in range(len_alphabet):
                        for k in range(len_alphabet):
                            t=1 if j==k else 0 # I don't know why we do this
                            phi_mean[c] = indicator_means[i+ len_indicators1D] * counts[j] * (counts[k] - t) 
                            c += 1

                B = build_mean_inner_product_matrix(counts,indicators1D,indicators2D,region_means)
                # B[i,j] - phi_mean[i]*phi_mean[j] is the covariance of some avg-indicator/gamma
                # product i with some other avg-indicator/gamma product j
                #
                # we can think of computing the total variance (summing this matrix)
                # as evaluating a potts model (covariances playing the role of "energies") that has
                # 3 fields (the burial indicator function for each density bin) and
                # 4 couplings (direct, prot, wat, and elec pairwise indicators)
                #
                # because we averaged over all indicators and they're playing the role of 
                # couplings and fields in our model, this is a "mean field" approach
                #
                                
                energy=0
                for i in range(phi_len):
                    for j in range(phi_len):
                        energy += gamma[i] * (B[i,j] - phi_mean[i]*phi_mean[j]) * gamma[j]
                return energy**.5
            
        compute_energy_numba=self.numbify(compute_energy)
            
        def denergy_mutation(seq_index, pos, aa):
            seq_index_new = seq_index.copy()
            seq_index_new[pos] = aa
            return compute_energy_numba(seq_index_new) - compute_energy_numba(seq_index)

        def denergy_swap(seq_index, pos1, pos2):
            return 0
        
        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation

        def compute_energy_permutation(seq_index):
            """ Function to compute the variance of the energy of all permutations of a sequence 
                Caution: This function is very slow for normal sequences """
            from itertools import permutations
            decoy_sequences = np.array(list(permutations(seq_index)))
            energies=np.zeros(len(decoy_sequences))
            for i in numba.prange(len(decoy_sequences)):
                energies[i]=awsem_energy(decoy_sequences[i])
            return np.var(energies)
        
        self.compute_energy_permutation=compute_energy_permutation

    def regression_test(self, seq_index):
        expected_energy=self.compute_energy_permutation(seq_index)
        energy=self.compute_energy(seq_index)
        assert np.isclose(energy,expected_energy), f"Expected energy {expected_energy} but got {energy}"


class AwsemEnergyStdFromCovMatrix(EnergyTerm):
    def __init__(self, covariance_matrix: np.ndarray, 
                       burial_gamma: np.ndarray,
                       direct_gamma: np.ndarray,
                       protein_gamma: np.ndarray,
                       water_gamma: np.ndarray,
                       electrostatics_gamma: np.ndarray,
                       use_numba = True, alphabet = _AA):  
        """
        covariance_matrix: np.ndarray
        Covariance matrix of all __indicator functions___ (not residues) over a decoy set.
        Should have the following structure:
        ___________________________________________________________________________________________
                                                             burial                 pairwise
                                                      _____________________________________________
                                     position 1 low  |                      .                      |
                                     ...             |                      .                      |
                                     position N low  |                      .                      |
                                     position 1 med  |    burial-burial     .     burial-pairwise  |
        burial                       ...             |                      .                      |
                                     position N med  |                      .                      |
                                     position 1 high |     covariances      .       covariances    |
                                     ...             |                      .                      |
                                     position N high |                      .                      |
        -------------------------------------------------------------------------------------------
                       direct interaction 1          |                      .                      |
                       ...                           |                      .                      |
                       direct interaction (N**2-N)/2 |                      .                      |                
                       prot interaction 1            |                      .                      |
                       ...                           |                      .                      |
        pairwise       prot interaction (N**2-N)/2   |   burial-pairwise    .    pairwise-pairwise |
                       wat interaction 1             |     covariances      .       covariances    |
                       ...                           |                      .                      |
                       wat interaction (N**2-N)/2    |                      .                      |
                       elec interaction 1            |                      .                      |
                       ...                           |                      .                      |
                       elec interaction (N**2-N)/2   |                      .                      |
        ___________________________________________________________________________________________|
    
        This matrix should have the same form as the covariance matrix that would be passed into
        AwsemVariancePotts, if we were doing things that way.
    
        gamma arrays: INPUTS MUST BE REINDEXED according to the alphabet used!
    
        """ 
        # check input
        if not len(covariance_matrix.shape) == 2:
            raise ValueError(f"covariance_matrix must have dimension 2 but was {len(covariance_matrix.shape)}")
        if not covariance_matrix.shape[0] == covariance_matrix.shape[1]:
            raise ValueError(f"covariance_matrix dimensions were not equal. covariance_matrix.shape: {covariance_matrix.shape}")
        if not burial_gamma.shape[1] == 3:
            raise ValueError(f"burial_gamma.shape[1] should be 3 but was {burial_gamma.shape[1]}")
        if not direct_gamma.shape[0]==direct_gamma.shape[1]\
             or not protein_gamma.shape[0]==protein_gamma.shape[1]\
               or not water_gamma.shape[0]==water_gamma.shape[1]\
                 or not electrostatics_gamma.shape[0]==electrostatics_gamma.shape[1]:
            raise ValueError("check gamma shapes")
        if not burial_gamma.shape[0] == len(alphabet):
            raise ValueError(f"alphabet {alphabet} and burial_gamma shape {burial_gamma.shape} are inconsistent")
        if not direct_gamma.shape[0] == len(alphabet):
            raise ValueError(f"alphabet {alphabet} and direct_gamma shape {direct_gamma.shape} are inconsistent")
        if not protein_gamma.shape[0] == len(alphabet):
            raise ValueError(f"alphabet {alphabet} and protein_gamma shape {protein_gamma.shape} are inconsistent")
        if not water_gamma.shape[0] == len(alphabet):
            raise ValueError(f"alphabet {alphabet} and water_gamma shape {water_gamma.shape} are inconsistent")
        if not electrostatics_gamma.shape[0] == len(alphabet):
            raise ValueError(f"alphabet {alphabet} and electrostatics_gamma shape {electrostatics_gamma.shape} are inconsistent")
        # set attributes
        self.covariance_matrix = covariance_matrix           
        self._use_numba = use_numba
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        N = 0
        while 3*N + 4*((N**2-N)/2) < self.covariance_matrix.shape[0]:
            N += 1
        if not 3*N + 4*((N**2-N)/2) == self.covariance_matrix.shape[0]:
            raise ValueError(f"the covariance matrix seems to have been constructed incorrectly. covariance_matrix.shape: {covariance_matrix.shape}")
        self.N = N # number of amino acids
        # compute products of gamma parameters for each indicator class 
        #     (burial low density, burial med, burial high, direct, prot, wat, elec)
        #     for each combination of amino acids (in general, 4 total because we have
        #     to evaluate the covariance of two pairwise indicators each depending on 
        #     the amino acid identity at 2 different positions in the sequence)
        gamma = np.zeros((7,7,len(alphabet),len(alphabet),len(alphabet),len(alphabet)))
        # we don't need the third and fourth axes for burial-burial covariances, so we copy the
        #    2D outer product along both new axes so that the gamma array is not ragged
        #    (the third and fourth axes are needed for other terms)
        #gamma[0,0] = np.repeat(np.outer(burial_gamma[:,0],burial_gamma[:,0])[:,None,:,None], len(alphabet), axis=1)# low burial- low burial
        gamma[0,0] = np.outer(burial_gamma[:,0],burial_gamma[:,0])[:,None,:,None]# low burial- low burial
        gamma[0,1] = np.outer(burial_gamma[:,0],burial_gamma[:,1])[:,None,:,None]# low burial- med burial
        gamma[0,2] = np.outer(burial_gamma[:,0],burial_gamma[:,2])[:,None,:,None]# low burial- high burial
        gamma[0,3] = np.einsum('i,jk->ijk', burial_gamma[:,0], direct_gamma)[:,None,:,:]  # low burial- direct
        gamma[0,4] = np.einsum('i,jk->ijk', burial_gamma[:,0], protein_gamma)[:,None,:,:]  # low burial- prot
        gamma[0,5] = np.einsum('i,jk->ijk', burial_gamma[:,0], water_gamma)[:,None,:,:]  # low burial- wat
        gamma[0,6] = np.einsum('i,jk->ijk', burial_gamma[:,0], electrostatics_gamma)[:,None,:,:]  # low burial- elec
        gamma[1,1] = np.outer(burial_gamma[:,1],burial_gamma[:,1])[:,None,:,None]# med burial- med burial
        gamma[1,2] = np.outer(burial_gamma[:,1],burial_gamma[:,2])[:,None,:,None]# med burial- high burial
        gamma[1,3] = np.einsum('i,jk->ijk', burial_gamma[:,1], direct_gamma)[:,None,:,:]  # med burial- direct
        gamma[1,4] = np.einsum('i,jk->ijk', burial_gamma[:,1], protein_gamma)[:,None,:,:]  # med burial- prot
        gamma[1,5] = np.einsum('i,jk->ijk', burial_gamma[:,1], water_gamma)[:,None,:,:]  # med burial- wat
        gamma[1,6] = np.einsum('i,jk->ijk', burial_gamma[:,1], electrostatics_gamma)[:,None,:,:]  # med burial- elec
        gamma[2,2] = np.outer(burial_gamma[:,2],burial_gamma[:,2])[:,None,:,None]# high burial- high burial
        gamma[2,3] = np.einsum('i,jk->ijk', burial_gamma[:,2], direct_gamma)[:,None,:,:]  # high burial- direct
        gamma[2,4] = np.einsum('i,jk->ijk', burial_gamma[:,2], protein_gamma)[:,None,:,:]  # high burial- prot
        gamma[2,5] = np.einsum('i,jk->ijk', burial_gamma[:,2], water_gamma)[:,None,:,:]  # high burial- wat
        gamma[2,6] = np.einsum('i,jk->ijk', burial_gamma[:,2], electrostatics_gamma)[:,None,:,:]  # high burial- elec
        gamma[3,3] = np.einsum('ij,kl->ijkl', direct_gamma, direct_gamma) # direct- direct
        gamma[3,4] = np.einsum('ij,kl->ijkl', direct_gamma, protein_gamma) # direct- prot
        gamma[3,5] = np.einsum('ij,kl->ijkl', direct_gamma, water_gamma) # direct- wat
        gamma[3,6] = np.einsum('ij,kl->ijkl', direct_gamma, electrostatics_gamma) # direct- elec
        gamma[4,4] = np.einsum('ij,kl->ijkl', protein_gamma, protein_gamma) # prot- prot
        gamma[4,5] = np.einsum('ij,kl->ijkl', protein_gamma, water_gamma) # prot- wat
        gamma[4,6] = np.einsum('ij,kl->ijkl', protein_gamma, electrostatics_gamma) # prot- elec
        gamma[5,5] = np.einsum('ij,kl->ijkl', water_gamma, water_gamma) # wat- wat
        gamma[5,6] = np.einsum('ij,kl->ijkl', water_gamma, electrostatics_gamma) # wat- elec
        gamma[6,6] = np.einsum('ij,kl->ijkl', electrostatics_gamma, electrostatics_gamma) # elec- elec
        self.gamma = gamma + gamma.transpose((0,1,5,4,3,2)) # keep the indicator class axes the same
                                                            #     but transpose the gamma values
        # define energy evaluation functions
        self.initialize_functions()

    @staticmethod
    def covariance_type(N, i, j):
        # N: total number of residues
        # i: first position in the covariance matrix
        # j: second position in the covariance matrix
        if i < 3*N:
            type_i = i//N
        else:
            type_i = (i-3*N)//((N**2-N)//2)
        if j < 3*N:
            type_j = j//N
        else:
            type_j = (j-3*N)//((N**2-N)//2)
        return (type_i, type_j)

    @staticmethod 
    def residue_identities(N, i, j, seq_index, indexing_helper_rowflatten, indexing_helper_columnflatten):
        # indexing helper rowflatten looks like [0, ..., 0, 1, ..., 1, ..., N]
        #                                            ^ repeated N times
        #                                                       ^ repeated N-1 times
        #                                                               ^ repeated once
        # indexing helper columnflatten looks like [0, ..., N, 1, ..., N, ..., N]
        if i < 3*N:
            i_pos = i%3
            i_aa = (seq_index[i_pos], seq_index[i_pos])
        else:
            i_pos1 = indexing_helper_rowflatten[(i-3*N)%((N**2-N)//2)]
            i_pos2 = indexing_helper_columnflatten[(i-3*N)%((N**2-N)//2)]
            i_aa = (seq_index[i_pos1], seq_index[i_pos2])
        if j < 3*N:
            j_pos = j%3
            j_aa = (seq_index[j_pos], seq_index[j_pos])
        else:
            j_pos1 = indexing_helper_rowflatten[(j-3*N)%((N**2-N)//2)]
            j_pos2 = indexing_helper_columnflatten[(j-3*N)%((N**2-N)//2)]
            j_aa = (seq_index[j_pos1], seq_index[j_pos2])
        return i_aa + j_aa # concatenating tuples
        
    def initialize_functions(self):
        covariance_matrix = self.covariance_matrix
        gamma = self.gamma
        N = self.N # number of amino acids
        indexing_helper_rowflatten = np.repeat(np.arange(N).reshape((N,1)),N,axis=1)[np.triu_indices(N)]
        indexing_helper_columnflatten = np.transpose(np.repeat(np.arange(N).reshape((N,1)),N,axis=1))[np.triu_indices(N)]
        covariance_type = self.covariance_type
        residue_identities = self.residue_identities
        
        def compute_energy(seq_index):
            energy = 0
            for i in range(covariance_matrix.shape[0]):
                for j in range(i,covariance_matrix.shape[1]):
                    try:
                        energy += covariance_matrix[i,j] * gamma[covariance_type(N,i,j)+residue_identities(N, i,j,seq_index, indexing_helper_rowflatten, indexing_helper_columnflatten )]
                    except:
                        breakpoint()
            return energy**0.5
        compute_energy_numba=self.numbify(compute_energy)        

        def compute_denergy_mutation(seq_index, pos, aa):
            seq_index_new = seq_index.copy()
            seq_index_new[pos] = aa
            return compute_energy_numba(seq_index_new) - compute_energy_numba(seq_index) 

        def compute_denergy_swap(seq_index, pos1, pos2):
            seq_index_new = seq_index.copy()
            aa2 , aa1 = seq_index[pos1],seq_index[pos2]
            seq_index_new[pos1] = aa1
            seq_index_new[pos2] = aa2
            return compute_energy_numba(seq_index_new) - compute_energy_numba(seq_index) 

        self.compute_energy = compute_energy
        self.compute_denergy_mutation = compute_denergy_mutation
        self.compute_denergy_swap = compute_denergy_swap

class Similarity(EnergyTerm):
    """ Computes the energy of a sequence based on the similarity to a target sequence. 
        The similarity is calculated as the number of positions that are the same in the two sequences.
        The similarity is then normalized by the length of the sequence. """
    def __init__(self, target_sequence, use_numba=True, alphabet=_AA):
        self.target_sequence = target_sequence
        self._use_numba = use_numba
        self.alphabet = alphabet
        self.target_index = sequence_to_index(target_sequence, alphabet=alphabet)
        self.initialize_functions()

    def initialize_functions(self):
        target_index=self.target_index
        len_seq=float(len(target_index))
        
        def compute_energy(seq_index):
            similarity = 0
            for i in range(len(seq_index)):
                similarity += seq_index[i] == target_index[i]
            return similarity / len_seq

        def denergy_mutation(seq_index, pos, aa):
            return ((aa == target_index[pos]) - (seq_index[pos] == target_index[pos]))/len_seq
        
        def denergy_swap(seq_index, pos1, pos2):
            return ((seq_index[pos1] == target_index[pos2]) + (seq_index[pos2] == target_index[pos1]) - (seq_index[pos1] == target_index[pos1]) - (seq_index[pos2] == target_index[pos2]))/len_seq

        self.compute_energy = compute_energy
        self.compute_denergy_mutation = denergy_mutation
        self.compute_denergy_swap = denergy_swap

class MonteCarlo:
    def __init__(self, sequence: str, energy: EnergyTerm, alphabet:str=_AA, use_numba:bool=True, evaluation_energies:dict={}):
        self.seq_len=len(sequence)
        self.energy = energy
        self.alphabet = alphabet
        self.use_numba = use_numba
        self.evaluation_energies = evaluation_energies
        self.initialize_functions()

    def generate_random_sequences(self,n):
        """ Generates n random sequences of the same length as the input sequence. """
        return np.random.randint(0, len(self.alphabet), size=(n,self.seq_len))
        
    @property
    def numbify(self):
        """ Returns the numba decorator if use_numba is True, otherwise returns a dummy decorator. """
        if self.use_numba:
            return numba.njit
        else:
            return self.dummy_decorator

    @staticmethod        
    def dummy_decorator(func, *args, **kwargs):
        """ Dummy decorator that returns the function unchanged. """
        return func

    def initialize_functions(self):
        """ Initializes the Monte Carlo functions for numba. """
        alphabet=self.alphabet
        alphabet_size=len(alphabet)
        sequence_size = self.seq_len
        compute_energy=self.energy.energy_function
        mutation_energy=self.energy.denergy_mutation_function
        swap_energy=self.energy.denergy_swap_function
        compute_energies=self.energy.energies_function

        def sequence_swap(seq_index):
            seq_index_new = seq_index.copy()
            n=len(seq_index_new)
            res1 = np.random.randint(0,n)
            res2 = np.random.randint(0,n-1)
            res2 += (res2 >= res1)
            energy_difference = swap_energy(seq_index, res1, res2)
            seq_index_new[res1], seq_index_new[res2] = seq_index[res2], seq_index[res1]
            return seq_index_new, energy_difference
        
        sequence_swap=self.numbify(sequence_swap)

        def sequence_mutation(seq_index):
            seq_index_new = seq_index.copy()
            r = np.random.randint(0, alphabet_size*sequence_size) # Select a random index
            res = r // alphabet_size
            aa_new = r % alphabet_size
            seq_index_new[res] = aa_new
            energy_difference = mutation_energy(seq_index, res, aa_new)
            return seq_index_new, energy_difference
        
        sequence_mutation=self.numbify(sequence_mutation)

        def montecarlo_steps(temperature, seq_index, n_steps = 1000, kb = 0.008314) -> np.array:
            for _ in range(n_steps):
                new_sequence, energy_difference = sequence_swap(seq_index) if np.random.random() > 0.5 else sequence_mutation(seq_index)
                exponent= (-energy_difference) / (kb * temperature + 1E-10)
                #breakpoint()
                acceptance_probability = np.exp(min(0, exponent))
                #assert ((acceptance_probability == 0) or (acceptance_probability ==1)), acceptance_probability
                #print(acceptance_probability)
                if np.random.random() < acceptance_probability:
                    seq_index = new_sequence
                    #print(f"before reassignment: {self.energy.stds}")
                    #self.energy.stds = copy.deepcopy(self.energy.consider_stds)
                    #print(f"after reassignment: {self.energy.stds}")
                    #self.energy.total_energies = copy.deepcopy(self.energy.consider_total_energies)
                    #print(f"energy_difference: {energy_difference}")
            return seq_index
        
        montecarlo_steps=self.numbify(montecarlo_steps)

        def replica_exchange(energies, temperatures, kb=0.008314, exchange_id=0):
            """
            Determine pairs of configurations between replicas for exchange.
            Returns a list of tuples with the indices of replicas to be exchanged.
            """
            n_replicas = len(temperatures)
            start_index = exchange_id % 2
            order = np.arange(len(temperatures), dtype=np.int64)
            
            for i in np.arange(start_index, n_replicas - 1, 2):
                energy1, energy2 = energies[i], energies[i + 1]
                temp1, temp2 = temperatures[i], temperatures[i + 1]
                delta = (1/temp2 - 1/temp1) * (energy2 - energy1)

                exponent = delta / kb # Sign is correct, as we want to swap when the system with higher temperature has lower energy
                prob = np.exp(min(0., exponent)) 

                if np.random.random() <= prob:
                    order[i]=i+1
                    order[i+1]=i
            return order
        
        replica_exchange=self.numbify(replica_exchange)

        def parallel_montecarlo_step(seq_indices, temperatures, n_steps_per_cycle):
            n_replicas = len(temperatures)

            for i in numba.prange(n_replicas):
                temp_seq_index = seq_indices[i]
                seq_indices[i] = montecarlo_steps(temperatures[i], seq_index=temp_seq_index, n_steps=n_steps_per_cycle)
                # Compute energy for the new sequence

            return seq_indices
        parallel_montecarlo_step=self.numbify(parallel_montecarlo_step, parallel=True)

        def parallel_tempering_steps(seq_indices, temperatures, n_steps, n_steps_per_cycle):
            for s in range(n_steps//n_steps_per_cycle):
                seq_indices = parallel_montecarlo_step(seq_indices, temperatures, n_steps_per_cycle)
                energies = compute_energies(seq_indices)
                # Yield data every 10 exchanges
                if s % 10 == 9:
                    yield s, seq_indices, energies

                # Perform replica exchanges
                order = replica_exchange(energies, temperatures, exchange_id=s)
                seq_indices = seq_indices[order]

        parallel_tempering_steps=self.numbify(parallel_tempering_steps)

        self.sequence_swap=sequence_swap
        self.sequence_mutation=sequence_mutation
        self.montecarlo_steps=montecarlo_steps
        self.replica_exchange=replica_exchange
        self.parallel_montecarlo_step=parallel_montecarlo_step
        self.parallel_tempering_steps=parallel_tempering_steps
            
    @csv_writer
    def parallel_tempering(self, seq_indices=None, temperatures=np.logspace(0,6,25), n_steps=int(1E8), n_steps_per_cycle=int(1E4), csv_filename="parallel_tempering_results.csv", csv_write=None):
        if seq_indices is None:
            seq_indices = self.generate_random_sequences(len(temperatures))
        
        total_energies = self.energy.energies(seq_indices)
        energies={key:energy_term.energies(seq_indices) for key,energy_term in self.evaluation_energies.items()}
        for i, temp in enumerate(temperatures):
            sequence_str = index_to_sequence(seq_indices[i],alphabet=self.alphabet)  # Convert sequence index back to string
            step_data=({'Step': 0, 'Temperature': temp, 'Sequence': sequence_str, 'Total Energy': total_energies[i]})
            step_data.update({key: energies[key][i] for key in self.evaluation_energies.keys()})
            csv_write(step_data)

        # Run the simulation and append data periodically
        for s, updated_seq_indices, total_energy in self.parallel_tempering_steps(seq_indices, temperatures, n_steps, n_steps_per_cycle):
            # Prepare data for this chunk
            energies={key:energy_term.energies(updated_seq_indices) for key,energy_term in self.evaluation_energies.items()}
            for i, temp in enumerate(temperatures):
                sequence_str = index_to_sequence(updated_seq_indices[i],alphabet=self.alphabet)  # Convert sequence index back to string
                step_data=({'Step': (s+1) * n_steps_per_cycle, 'Temperature': temp, 'Sequence': sequence_str, 'Total Energy': total_energy[i]})
                step_data.update({key: energies[key][i] for key in self.evaluation_energies.keys()})
                csv_write(step_data)

    @csv_writer
    def annealing(self, seq_index=None, temperatures=np.arange(500,0,-1), n_steps=int(1E8), csv_filename="annealing.csv", csv_write=None):
        if seq_index is None:
            seq_index = self.generate_random_sequences(1)[0]
        
        done_steps=0
        total_energy = self.energy.energy(seq_index, )

        #Write data to file
        step_data={'Step': done_steps, 'Temperature': temperatures[0], 'Sequence': index_to_sequence(seq_index,alphabet=self.alphabet), 'TotalEnergy': total_energy}
        step_data.update({key: energy_term.energy_function(seq_index) for key, energy_term in self.evaluation_energies.items()})
        csv_write(step_data)

        for t,temp in enumerate(temperatures):
            steps=(n_steps-done_steps)//(len(temperatures)-t)
            assert steps >= 1, f"steps: {steps}"
            seq_index= self.montecarlo_steps(temp, seq_index, n_steps=steps)
            total_energy = self.energy.energy(seq_index)
            done_steps+=steps

            #Write data to file
            step_data={'Step': done_steps, 'Temperature': temp, 'Sequence': index_to_sequence(seq_index,alphabet=self.alphabet), 'TotalEnergy': total_energy}
            step_data.update({key: energy_term.energy_function(seq_index) for key, energy_term in self.evaluation_energies.items()})
            csv_write(step_data)
        return seq_index

    def benchmark_montecarlo_steps(self, n_repeats=10, n_steps=20000):
        import time
        times = []

        #Adds one step for numba compilation time
        seq_index = np.random.randint(1, len(self.alphabet), size=self.seq_len)
        self.montecarlo_steps(temperature=500, seq_index=self.generate_random_sequences(1)[0], n_steps=1)

        for _ in range(n_repeats):  # Run benchmark 10 times
            # Generate a new random sequence for each run
            seq_index = np.random.randint(1, len(self.alphabet), size=self.seq_len)
            start_time = time.time()
            
            self.montecarlo_steps(temperature=500, seq_index=seq_index, n_steps=n_steps)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        average_time_per_step = sum(times) / len(times) / n_steps
        std_time_per_step = np.std(times) / len(times) / n_steps
        
        steps_per_hour = 3600 / average_time_per_step
        time_needed = 1E10 * average_time_per_step / 8  # 8 processes in parallel

        print(f"Time needed to explore 10 billion sequences assumming 8 process in parallel: {format_time(time_needed)}")
        print(f"Number of sequences explored per hour: {steps_per_hour:.2e}")
        print(f"Average execution time per step: {format_time(average_time_per_step)} +- {format_time(3*std_time_per_step)}")

    def benchmark_parallel_montecarlo_steps(self, n_repeats=10, n_steps=20000, n_replicas=8):
        import time
        times = []

        # Add one step for numba compilation time
        seq_indices = self.generate_random_sequences(n_replicas)
        temperatures = np.logspace(0, 6, n_replicas)
        self.parallel_montecarlo_step(seq_indices, temperatures, n_steps_per_cycle=1)

        for _ in range(n_repeats):
            seq_indices = self.generate_random_sequences(n_replicas)
            start_time = time.time()
            self.parallel_montecarlo_step(seq_indices, temperatures, n_steps_per_cycle=n_steps)
            end_time = time.time()
            times.append(end_time - start_time)
        
        average_time_per_step_s = np.mean(times) / n_steps / n_replicas
        std_time_per_step_s = np.std(times) / n_steps / n_replicas
        
        steps_per_hour = 3600 / average_time_per_step_s
        time_needed = 1E10 * average_time_per_step_s

        print(f"Time needed for 10 billion steps with {n_replicas} in parallel: {format_time(time_needed)}")
        print(f"Number of sequences explored per hour with {n_replicas}: {steps_per_hour:.2e}")
        print(f"Average time per step with {n_replicas}: {format_time(average_time_per_step_s)} ± {format_time(std_time_per_step_s)}")
        

    def find_optimal_replicas(self, max_replicas=32, n_repeats=5, n_steps=10000):
        import time
        results = []

        for n_replicas in range(1, max_replicas + 1):
            times = []
            seq_indices = self.generate_random_sequences(n_replicas)
            temperatures = np.logspace(0, 6, n_replicas)

            # Warm-up run
            self.parallel_montecarlo_step(seq_indices, temperatures, n_steps_per_cycle=1)

            for _ in range(n_repeats):
                seq_indices = self.generate_random_sequences(n_replicas)
                start_time = time.time()
                self.parallel_montecarlo_step(seq_indices, temperatures, n_steps_per_cycle=n_steps)
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = np.mean(times)
            steps_per_second = n_steps * n_replicas / avg_time
            results.append((n_replicas, steps_per_second))

        optimal_replicas, max_steps_per_second = max(results, key=lambda x: x[1])
        
        print(f"Optimal number of replicas: {optimal_replicas}")
        print(f"Maximum steps per second: {max_steps_per_second:.2f}")
        print(f"Time needed for 10 billion steps: {format_time(1E10 / max_steps_per_second)}")

        return results



if __name__ == '__main__':

    reduced_alphabet = 'ADEFGHIKLMNQRSTVWY'
    pdb = "tests/data/1r69.pdb"    
    s = Structure(pdb, chain=None)
    model = AWSEM(s, expose_indicator_functions=True,
                    distance_cutoff_contact=10, min_sequence_separation_contact=2,)
    variance = AwsemEnergyVariance(model, alphabet=reduced_alphabet)
    monte_carlo = MonteCarlo(sequence="SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT",
                             energy=variance, alphabet=reduced_alphabet)
    monte_carlo.annealing(n_steps=10)
    exit()

    pdb_list = ["tests/data/1r69.pdb","tests/data/1r69.pdb","tests/data/1r69.pdb"]
    pdb_structures = (Structure(pdb, chain=None) for pdb in pdb_list)
    ensemble = DecoyEnsemble(pdb_structures,  distance_cutoff_contact=10, min_sequence_separation_contact=10)
    burial_indicators, direct_indicators, protein_indicators, water_indicators, electrostatics_indicators = ensemble.average()
    average = AWSEMIndicators(burial_indicators, direct_indicators, protein_indicators, water_indicators, electrostatics_indicators,
                                  "SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT")
    
    reduced_alphabet = 'ADEFGHIKLMNQRSTVWY'

    awsem_energy = AwsemEnergy(average, alphabet=reduced_alphabet)
    heterogeneity = Heterogeneity(exact=False, use_numba=True)

    energy_mix = awsem_energy - 10*heterogeneity
    
    monte_carlo = MonteCarlo(sequence = "SISSRVKSKRIQLGLNQAELAQKVGTTQQSIEQLENGKTKRPRFLPELASALGVSVDWLLNGT", energy=energy_mix, alphabet=reduced_alphabet)
    monte_carlo.annealing(n_steps=1000)
    
    
    
if False:    
    
    #native_pdb = "tests/data/1bfz.pdb"
    #native_pdb = "tests/data/1r69.pdb"
    
    #pdbs = {"Swap": "frustratometer/optimization/swap_EGFR.pdb", "LinkerBack": "frustratometer/optimization/10.3_model_LinkerBack_partialEGFR.pdb", }
    #pdbs = {"Swap": "frustratometer/optimization/swap_EGFR.pdb"}
    pdbs={"1bfz": "tests/data/1bfz.pdb"}
    #gammas = {"original":"frustratometer/data/AWSEM_2015.json", "ddG":"frustratometer/data/AWSEM_ddG_trained.json"}
    gammas = {"original":"frustratometer/data/AWSEM_2015.json"}

    for pdb_name in pdbs:
        native_pdb = pdbs[pdb_name]
        
        structure_bound = Structure.full_pdb(native_pdb, chain=None)
        structure_free = Structure.full_pdb(native_pdb, "A")



        for gamma_name in gammas:
            
            gamma = gammas[gamma_name]
            

            model_bound = AWSEM(structure_bound, distance_cutoff_contact=10, min_sequence_separation_contact=2, expose_indicator_functions=True, gamma=gamma)
            model_free = AWSEM(structure_free, distance_cutoff_contact=10, min_sequence_separation_contact=2, expose_indicator_functions=True, gamma=gamma)
            reduced_alphabet = 'ADEFHIKLMNQRSTVWY'

            print(model_bound.sequence)
            print(model_free.sequence)

            binding_region=np.array([1, 2, 3, 4, 26, 27, 28, 29, 30, 31, 32, 33, 49, 50, 51, 52, 53, 54, 55, 56, 57, 68, 69, 70, 90, 91, 92, 93, 94, 95, 96, 97, 109, 110, 111, 112, 113, 114, 115, 116, 117, 127, 128, 129, 130, 131, 132, 133, 134, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 190, 191, 192, 193, 194, 195, 196, 197])-1
            energy_bound = AwsemEnergySelected(model_bound, alphabet=reduced_alphabet, selection = binding_region) #selection=np.array(range(len(model_free.sequence)))
            energy_free = AwsemEnergySelected(model_free, alphabet=reduced_alphabet, selection = binding_region)
            energy_average = AwsemEnergyAverage(model_free, alphabet=reduced_alphabet)
            energy_std = AwsemEnergyStd(model_free, alphabet=reduced_alphabet)
            energy_variance = AwsemEnergyVariance(model_free, alphabet=reduced_alphabet)
            heterogeneity = Heterogeneity(exact=False, use_numba=True)
            similarity = Similarity(model_free.sequence, use_numba=True)

            # energy_mix = energy_free - 20 * heterogeneity
            energy_mix = (energy_free - energy_average) / energy_std
            # energy_mix = energy_bound - energy_free
            
            if False:
                    # Profiling of the parallel tempering
                import cProfile
                import pstats
                import io

                monte_carlo = MonteCarlo(sequence='A'*len(binding_region),  energy=energy_mix, alphabet=reduced_alphabet, 
                                                evaluation_energies={"EnergyFree": energy_free, "EnergyBound": energy_bound, "Heterogeneity": heterogeneity, 
                                                                    "EnergyAverage": energy_average, "EnergyStd": energy_std, "Binding": (energy_bound - energy_free), 
                                                                    "Similarity": similarity, "Zscore":(energy_free - energy_average) / energy_std})
                        
                monte_carlo.benchmark_montecarlo_steps(n_repeats=3, n_steps=1000)

                # Run the profiler
                profiler = cProfile.Profile()
                profiler.enable()

                monte_carlo.parallel_tempering(temperatures=np.logspace(3,-4,8), n_steps=1E5, n_steps_per_cycle=1E3)
                profiler.disable()
                
                # Print the stats
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
                ps.print_stats()
                ps.dump_stats('parallel_temperingv2.prof')


            if False:
                energy_mixes = {"Ivan":energy_bound - 40 * heterogeneity, 
                                "Takada": (energy_bound - energy_average) / energy_std, 
                                "Ivan_binding":(energy_bound - energy_free) - 40 * heterogeneity,
                                "Takada_binding":(energy_free - energy_average) / energy_std + (energy_bound - energy_free),
                                "Ivan_Takada_binding": (energy_free - energy_average) / energy_std + (energy_bound - energy_free) - 40 * heterogeneity,
                                "Corrected_Takada": (energy_bound - energy_average) / (energy_std+5), 
                                "Corrected_Takada_binding":(energy_free - energy_average) / (energy_std+5) + (energy_bound - energy_free),
                                "Ivan_Corrected_Takada_binding": (energy_free - energy_average) / (energy_std+5) + (energy_bound - energy_free) - 40 * heterogeneity,
                                "Ivan_bindidng similarity": (energy_bound - energy_free) - 40 * heterogeneity - 100*similarity,
                                "Corrected_Takada_binding_similarity":(energy_free - energy_average) / (energy_std+5) + (energy_bound - energy_free) - 100*similarity,
                                "Ivan_bindidng_similarityv2": (energy_bound - energy_free) - 40 * heterogeneity,
                }

                for energy_mix_name in energy_mixes:
                    print(f"Native PDB: {pdb_name}")
                    print(f"Gamma file: {gamma_name}")
                    print(f"Energy mix: {energy_mix_name}")
                    energy_mix = energy_mixes[energy_mix_name]
                    monte_carlo = MonteCarlo(sequence='A'*len(binding_region),  energy=energy_mix, alphabet=reduced_alphabet, 
                                            evaluation_energies={"EnergyFree": energy_free, "EnergyBound": energy_bound, "Heterogeneity": heterogeneity, "EnergyAverage": energy_average, "EnergyStd": energy_std, "Binding": (energy_bound - energy_free), "Similarity": similarity, "Zscore":(energy_free - energy_average) / energy_std})
                    
                    monte_carlo.benchmark_montecarlo_steps(n_repeats=3, n_steps=1000)
                    csv_filename = f"Montecarlov2_{energy_mix_name}_pdb_{pdb_name}_gamma_{gamma_name}_1E5.csv"
                    seq_index = monte_carlo.annealing(temperatures=np.logspace(2,-4,36), n_steps=1E5, csv_filename=csv_filename)
                    seq = index_to_sequence(seq_index, alphabet=reduced_alphabet)
                    complete_seq = ""
                    c=0
                    for i,x in enumerate(model_free.sequence):
                        if i in binding_region:
                            complete_seq+=seq[c]
                            c+=1
                        else:
                            complete_seq+=x
                    print("Complete sequence ",complete_seq)
                    
                    


    #energy_terms={"energy_bound": energy_bound, "energy_free": energy_free, "energy_free2": energy_free2, "heterogeneity": heterogeneity, "energy_average": energy_average, "energy_variance":energy_variance, "energy_mix":energy_mix}

    # for energy_name,energy_term in energy_terms.items():
    #     print (f"Energy term: {energy_name}")
    #     energy_term.benchmark(seq_indices=np.random.randint(0, len(reduced_alphabet), size=(100,len(structure_free.sequence))))
    #     energy_term.test(seq_index=np.random.randint(0, len(reduced_alphabet), size=len(structure_free.sequence)))
        
    #     monte_carlo = MonteCarlo(sequence = structure_free.sequence, energy=energy_term, alphabet=reduced_alphabet, evaluation_energies=energy_terms)
    #     monte_carlo.benchmark_montecarlo_steps(n_repeats=3,n_steps=100)


    # monte_carlo.parallel_tempering(temperatures=np.logspace(0,6,49), n_steps=1E8, n_steps_per_cycle=1E6)
    # monte_carlo.annealing(temperatures=np.logspace(2,-4,36), n_steps=1E6)

    # self = AwsemEnergySelected(model_free, alphabet=reduced_alphabet,selection=np.array([1,4]))
    # assert np.isclose(self.energy(np.array([self.seq_index[1],self.seq_index[4]])),model_free.native_energy()), "Selected energy does not match native energy"
    # self.test(np.array([0,0]))

    # energy_bound.test(seq_index=np.random.randint(0, len(reduced_alphabet), size=len(structure_free.sequence)))
    # energy_bound.regression_test()

    # energy_mix = energy_bound - energy_free
    # monte_carlo.annealing(temperatures=np.logspace(2,-4,36), n_steps=1E6)



    if True:
        # Benchmarking of terms    
        energy_terms={"energy_bound": energy_bound, "energy_free": energy_free, "heterogeneity": heterogeneity, "energy_average": energy_average, "energy_variance":energy_variance, "energy_mix":energy_mix}

        for energy_name,energy_term in energy_terms.items():
            print (f"Energy term: {energy_name}")
            energy_term.benchmark(seq_indices=np.random.randint(0, len(reduced_alphabet), size=(100,len(structure_free.sequence))))
            energy_term.test(seq_index=np.random.randint(0, len(reduced_alphabet), size=len(structure_free.sequence)))
            
            monte_carlo = MonteCarlo(sequence = structure_free.sequence, energy=energy_term, alphabet=reduced_alphabet, evaluation_energies=energy_terms)
            monte_carlo.benchmark_montecarlo_steps(n_repeats=3,n_steps=100)

        
        # Profiling of the parallel tempering
        import cProfile
        import pstats
        import io

        # Run the profiler
        profiler = cProfile.Profile()
        profiler.enable()
        monte_carlo.parallel_tempering(temperatures=np.logspace(2,-4,8), n_steps=1E5, n_steps_per_cycle=1E3)
        profiler.disable()
        
        # Print the stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        ps.dump_stats('parallel_tempering.prof')



