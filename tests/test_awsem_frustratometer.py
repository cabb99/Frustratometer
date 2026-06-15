import pytest
import pandas as pd
import numpy as np
import frustratometer
from pathlib import Path


test_path=Path('tests')
test_data_path=Path('tests/data')


def _dense_mask(model):
    """Dense (L, L) boolean mask from a model, accepting a SparseMatrix or ndarray mask."""
    mask = model.mask
    dense = mask.to_dense(fill=0.0) if hasattr(mask, "to_dense") else mask
    return np.asarray(dense).astype(bool)


tests_config = pd.read_csv(test_path/"test_awsem_config.csv",comment='#')

# Build parametrize list, marking 1jge rows as memory_heavy (~7.5 GB peak RSS)
_MEMORY_HEAVY_PDBS = {"1jge"}
_test_params = []
for _rec in tests_config.to_dict(orient="records"):
    if _rec["pdb"] in _MEMORY_HEAVY_PDBS:
        _test_params.append(pytest.param(_rec, marks=pytest.mark.memory_heavy))
    else:
        _test_params.append(_rec)

@pytest.fixture(scope="module")
def test_structure():
    return {test_data['pdb']: frustratometer.Structure(test_data_path/f"{test_data['pdb']}.pdb") for test_data in tests_config.to_dict(orient="records")}


@pytest.fixture(scope="module")
def awsem_6u5e():
    """6u5e + AWSEM(k_elec=0, sep=10, cutoff=None) — shared by fields/couplings energy tests."""
    structure = frustratometer.Structure(test_data_path / '6u5e.pdb', "A")
    return frustratometer.AWSEM(structure, k_electrostatics=0,
                                min_sequence_separation_contact=10,
                                distance_cutoff_contact=None)


@pytest.fixture(scope="module")
def awsem_6u5e_density():
    """6u5e + AWSEM(cutoff=9.499, sep=2, k_elec=0) — shared by single-residue energy/decoy tests."""
    structure = frustratometer.Structure(test_data_path / '6u5e.pdb', "A")
    return frustratometer.AWSEM(structure,
                                distance_cutoff_contact=9.499,
                                min_sequence_separation_contact=2,
                                k_electrostatics=0)


@pytest.fixture(scope="module")
def awsem_1mba():
    """1MBA_A sub/full models (k_elec=0, sep=10, cutoff=10) — shared by subsequence tests."""
    substructure = frustratometer.Structure(test_data_path / '1MBA_A.pdb', "A",
                                            seq_selection="resnum 39to146")
    model_sub = frustratometer.AWSEM(substructure, k_electrostatics=0.0,
                                     min_sequence_separation_contact=10,
                                     distance_cutoff_contact=10.0)
    model_sub_no_context = frustratometer.AWSEM(substructure, k_electrostatics=0.0,
                                                min_sequence_separation_contact=10,
                                                distance_cutoff_contact=10.0,
                                                burial_in_context=False)
    full_structure = frustratometer.Structure(test_data_path / '1MBA_A.pdb', "A")
    model_full = frustratometer.AWSEM(full_structure, k_electrostatics=0.0,
                                      min_sequence_separation_contact=10,
                                      distance_cutoff_contact=10.0)
    return {"sub": model_sub, "full": model_full, "sub_no_context": model_sub_no_context}

def test_prody_expected_error():
    test_data=tests_config.iloc[0]
    try:
        structure = frustratometer.Structure(test_data_path/f"{test_data['pdb']}.pdb")
        assert True
    except TypeError as e:
        if "can't multiply sequence by non-int of type 'Forward'" in str(e):
            print("Encountered a ProDy TypeError on initial run. Error logged for future reference")
        else:
            raise


@pytest.mark.parametrize("test_data", _test_params)
def test_density_residues(test_data, test_structure):
    #structure = frustratometer.Structure(test_data_path/f"{test_data['pdb']}.pdb")
    structure = test_structure[test_data['pdb']]
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, k_electrostatics=0)
    data = pd.read_csv(test_data['singleresidue'], sep=r'\s+')
    data['Calculated_density'] = model.rho_r
    data['Expected_density'] = data['DensityRes']
    max_atol = np.max(np.abs(data['Calculated_density'] - data['Expected_density']))
    print(max_atol)
    try:
        assert np.allclose(data['Calculated_density'], data['Expected_density'], atol=1E-3)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_density'] - data['Expected_density']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of 1E-3.")

@pytest.mark.parametrize("test_data", _test_params)
def test_single_residue_frustration(test_data,test_structure):
    #structure = frustratometer.Structure(test_data_path/f"{test_data['pdb']}.pdb")
    structure = test_structure[test_data['pdb']]
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, min_sequence_separation_contact=2, k_electrostatics=test_data['k_electrostatics'] * 4.184, min_sequence_separation_electrostatics=1)
    data = pd.read_csv(test_data['singleresidue'], sep=r'\s+')
    data['Calculated_frustration'] = model.frustration(kind='singleresidue')
    data['Expected_frustration'] = data['FrstIndex']
    try:
        assert np.allclose(data['Calculated_frustration'], data['Expected_frustration'], atol=3E-1)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_frustration'] - data['Expected_frustration']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of 3E-1.")

@pytest.mark.parametrize("test_data", _test_params)
def test_mutational_frustration(test_data,test_structure):
    #structure = frustratometer.Structure(test_data_path/f"{test_data['pdb']}.pdb")
    structure = test_structure[test_data['pdb']]
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    if test_data['k_electrostatics']==1000:
        assert True
        return
    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, min_sequence_separation_rho=sequence_separation, min_sequence_separation_contact=0, k_electrostatics=test_data['k_electrostatics'] * 4.184, min_sequence_separation_electrostatics=1)
    data = pd.read_csv(test_data['mutational'], sep=r'\s+')
    
    if test_data['pdb']!="ijge":
        chains=['A','B','C']
        for chain,next_chain in zip(chains,chains[1:]):
            max_resid={'A':277,'B':277+99,'C':9}
            data.loc[(data['ChainRes1']==next_chain),'#Res1']+=max_resid[chain]
            data.loc[(data['ChainRes2']==next_chain),'Res2']+=max_resid[chain]
    
    start_pdb=1 if test_data['pdb']!="6u5e" else 2
    data['Calculated_frustration'] = model.frustration(kind='mutational')[data['#Res1']-start_pdb, data['Res2']-start_pdb]
    data['Expected_frustration'] = data['FrstIndex']
    #data.to_csv(f"/home/fc36/dump/{test_data['pdb']}_seqsep_{test_data['seqsep']}_kelec_{test_data['k_electrostatics']}_mutational.csv") 
    #np.savetxt(f"/home/fc36/dump/{test_data['pdb']}_seqsep_{test_data['seqsep']}_kelec_{test_data['k_electrostatics']}_mutational_test_full.csv", model.frustration(kind='mutational'),delimiter=',')
    #np.savetxt(f'/home/fc36/dump/{test_data["pdb"]}_seqsep_{test_data["seqsep"]}_kelec_{test_data["k_electrostatics"]}_other_save.csv',model.frustration(kind='mutational')[data['#Res1']-start_pdb, data['Res2']-start_pdb],delimiter=',')
    if test_data['pdb'] == 'sequence0':
        atol=3.5E-1
    else:
        atol=3E-1
    try:
        assert np.allclose(data['Calculated_frustration'], data['Expected_frustration'], atol=atol)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_frustration'] - data['Expected_frustration']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of {atol}.")

@pytest.mark.slow
@pytest.mark.stochastic
@pytest.mark.parametrize("test_data", _test_params)
def test_configurational_frustration(test_data,test_structure):
    #This test may fail due to the randomness of the decoy generation

    #structure = frustratometer.Structure(test_data_path/f"{test_data['pdb']}.pdb")
    structure = test_structure[test_data['pdb']]
    sequence_separation = 2 if test_data['seqsep'] == 3 else 13
    
    if test_data['k_electrostatics'] == 1000:
        assert True
        return

    model = frustratometer.AWSEM(structure, distance_cutoff_contact=9.5, 
                                 min_sequence_separation_rho=sequence_separation, 
                                 min_sequence_separation_contact=0, 
                                 k_electrostatics=test_data['k_electrostatics'] * 4.184, 
                                 min_sequence_separation_electrostatics=1)
    
    data = pd.read_csv(test_data['configurational'], sep=r'\s+')
    
    if test_data['pdb'] != "ijge":
        chains = ['A', 'B', 'C']
        for chain, next_chain in zip(chains, chains[1:]):
            max_resid = {'A': 277, 'B': 277 + 99, 'C': 9}
            data.loc[data['ChainRes1'] == next_chain, '#Res1'] += max_resid[chain]
            data.loc[data['ChainRes2'] == next_chain, 'Res2'] += max_resid[chain]

    start_pdb = 1 if (test_data['pdb'] != "6u5e" or test_data['lammps']) else 2
    data['Calculated_frustration'] = model.configurational_frustration(n_decoys=10000)[data['#Res1'] - start_pdb, data['Res2'] - start_pdb]
    #data.to_csv(f"/home/fc36/dump/{test_data['pdb']}_seqsep_{test_data['seqsep']}_kelec_{test_data['k_electrostatics']}_configurational.csv")
    data['Expected_frustration'] = data['FrstIndex']
    #np.savetxt(f"/home/fc36/dump/{test_data['pdb']}_seqsep_{test_data['seqsep']}_kelec_{test_data['k_electrostatics']}_configurational_full.csv",model.configurational_frustration(n_decoys=10000),delimiter=',')
    if test_data['pdb'] in ['sequence0','sequence1']:
        atol = 6E-1
    else:
        atol = 3E-1
    try:
        assert np.allclose(data['Calculated_frustration'], data['Expected_frustration'], atol=atol)
    except AssertionError:
        max_atol = np.max(np.abs(data['Calculated_frustration'] - data['Expected_frustration']))
        print(f"Assertion failed: Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance.")
        raise AssertionError(f"Maximum absolute tolerance found was {max_atol}, which exceeds the allowed tolerance of {atol}.")

#####
#Test AWSEM Native Energy Calculations
#####
def test_residue_density_calculation():
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(test_data_path/"6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep=r"\s+")
    lammps_single_frustration_dataframe["i"]=lammps_single_frustration_dataframe["i"]-1
    expected_rho_values=lammps_single_frustration_dataframe["rho_i"]

    structure=frustratometer.Structure(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=2)
    assert np.round(model.rho_r,2).all()==np.round(expected_rho_values,2).all()

def test_AWSEM_native_energy():
    structure=frustratometer.Structure(test_data_path/f'1l63.pdb',"A")
    model=frustratometer.AWSEM(structure,k_electrostatics=0, min_sequence_separation_contact = 10, distance_cutoff_contact = None)
    e = model.native_energy()
    print(e)
    assert np.round(e, 0) == -915

def test_AWSEM_fields_energy(awsem_6u5e):
    e = awsem_6u5e.fields_energy()
    print(e)
    assert np.round(e, 0) == -555

def test_AWSEM_couplings_energy(awsem_6u5e):
    e = awsem_6u5e.couplings_energy()
    print(e)
    assert np.round(e, 0) == -362

def test_fields_couplings_AWSEM_energy():
    structure=frustratometer.Structure(test_data_path/f'6u5e.pdb',"A")
    model = frustratometer.AWSEM(structure)
    assert model.fields_energy() + model.couplings_energy() - model.native_energy()  < 1E-6

def test_single_residue_AWSEM_energy(awsem_6u5e_density):
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep=r"\s+")
    ###
    model = awsem_6u5e_density
    #Calculate fields
    seq_index = np.array([_AA.find(aa) for aa in model.sequence])
    seq_len = len(seq_index)
    h = -model.potts_model['h'][range(seq_len), seq_index]

    #Calculate couplings
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    j = -model.potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * _dense_mask(model)

    test_residue_total_energy=(h +j_prime.sum(axis=0))/4.184

    assert (abs(np.array(lammps_single_frustration_dataframe["native_energy"])-test_residue_total_energy) < 1E-1).all()

def test_contact_pair_AWSEM_energy():
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer mutational frustration values
    lammps_mutational_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep=r"\s+")
    lammps_mutational_frustration_dataframe["i"]=lammps_mutational_frustration_dataframe["i"]-1
    lammps_mutational_frustration_dataframe["j"]=lammps_mutational_frustration_dataframe["j"]-1
    ###
    structure=frustratometer.Structure(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,distance_cutoff_contact=9.499,
                                                  min_sequence_separation_contact=0,
                                                  k_electrostatics=0)
    #Calculate fields
    seq_index = np.array([_AA.find(aa) for aa in structure.sequence])
    seq_len = len(seq_index)
    h = -model.potts_model['h'][range(seq_len), seq_index]

    #Calculate couplings
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    j = -model.potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * _dense_mask(model)
    test_contact_energy_matrix=h[pos1]+h[pos2]+j_prime.sum(axis=0)[pos1]+j_prime.sum(axis=0)[pos2]-j_prime[pos1,pos2]
    ###
    lammps_mutational_frustration_dataframe["Test_Native_Energy"]=lammps_mutational_frustration_dataframe.apply(lambda x: test_contact_energy_matrix[x.i,x.j],axis=1)
    lammps_mutational_frustration_dataframe["Test_Native_Energy"]=lammps_mutational_frustration_dataframe["Test_Native_Energy"]/4.184

    assert (abs(np.array(lammps_mutational_frustration_dataframe["native_energy"])-np.array(lammps_mutational_frustration_dataframe["Test_Native_Energy"])) < 1E-1).all()

def test_selected_subsequence_AWSEM_contact_energy_matrix():
    structure=frustratometer.Structure(test_data_path/f'4wnc.pdb',"A",seq_selection="resnum 3to26")
    model=frustratometer.AWSEM(structure)
    assert model.potts_model['h'].shape==(24,21)

def test_selected_subsequence_AWSEM_burial_energy_matrix():
    structure=frustratometer.Structure(test_data_path/f'4wnc.pdb',"A",seq_selection="resnum 150to315")
    model=frustratometer.AWSEM(structure)
    assert model.potts_model['J'].shape==(166,166,21,21)

#####
#Test Protein Segment Native AWSEM Energy Calculation
#####

def test_selected_subsequence_AWSEM_rho_calculations(awsem_1mba):
    model_1 = awsem_1mba["sub"]
    model_2 = awsem_1mba["full"]
    model_1_init_index = model_1.init_index_shift
    model_1_fin_index = model_1.fin_index_shift
    #Check if shape and entries of rho matrices are identical
    assert model_1.rho_r.shape==model_2.rho_r[model_1_init_index:model_1_fin_index].shape
    assert model_1.rho_r.all()==model_2.rho_r[model_1_init_index:model_1_fin_index].all()

def test_selected_subsequence_AWSEM_burial_energy(awsem_1mba):
    model_1 = awsem_1mba["sub"]
    model_2 = awsem_1mba["full"]
    model_1_init_index = model_1.init_index_shift
    model_1_fin_index = model_1.fin_index_shift
    #Check if burial energies are identical
    assert model_1.burial_energy.shape==model_2.burial_energy[model_1_init_index:model_1_fin_index].shape
    assert model_1.burial_energy.all()==model_2.burial_energy[model_1_init_index:model_1_fin_index].all()

def test_selected_subsequence_AWSEM_contact_energy(awsem_1mba):
    model_1 = awsem_1mba["sub"]
    model_2 = awsem_1mba["full"]
    model_1_init_index = model_1.init_index_shift
    model_1_fin_index = model_1.fin_index_shift
    #Check if couplings (J) are identical for the subsequence
    assert model_1.potts_model['J'].shape==model_2.potts_model['J'][model_1_init_index:model_1_fin_index,model_1_init_index:model_1_fin_index,:,:].shape
    assert model_1.potts_model['J'].all()==model_2.potts_model['J'][model_1_init_index:model_1_fin_index,model_1_init_index:model_1_fin_index,:,:].all()

def test_selected_subsequence_AWSEM_burial_energy_without_protein_context(awsem_1mba):
    model = awsem_1mba["sub_no_context"]
    selected_region_burial = model.fields_energy()
    # Energy units are in kJ/mol
    assert np.round(selected_region_burial, 2) == -377.95

def test_selected_subsequence_AWSEM_contact_energy_without_protein_context(awsem_1mba):
    model = awsem_1mba["sub_no_context"]
    selected_region_contact = model.couplings_energy()
    # Energy units are in kJ/mol
    assert np.round(selected_region_contact, 2) == -148.92

def test_single_residue_decoy_AWSEM_energy_statistics(awsem_6u5e_density):
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer single residue frustration values
    lammps_single_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_singleresidue_1E8decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep=r"\s+")
    ###
    model = awsem_6u5e_density
    #Calculate fields
    seq_index = np.array([_AA.find(aa) for aa in model.sequence])
    seq_len = len(seq_index)
    h = -model.potts_model['h'][range(seq_len), seq_index]

    #Calculate couplings
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    j = -model.potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * _dense_mask(model)

    residue_total_energy=(h +j_prime.sum(axis=0))/4.184
    ###
    decoy_fluctuations=(model.decoy_fluctuation(kind='singleresidue'))/4.184
    weighted_decoy_fluctations=(model.aa_freq*decoy_fluctuations).sum(axis=1)/ model.aa_freq.sum()

    expected_mean_decoy_energy=(model.aa_freq*(residue_total_energy[:, np.newaxis]+decoy_fluctuations)).sum(axis=1)/ model.aa_freq.sum()
    expected_std_decoy_energy=np.sqrt(((model.aa_freq * (decoy_fluctuations - weighted_decoy_fluctations[:, np.newaxis]) ** 2) / model.aa_freq.sum()).sum(axis=1))
    
    assert (abs(np.array(lammps_single_frustration_dataframe["<decoy_energies>"])-(expected_mean_decoy_energy)) < 1.2E-1).all()
    assert (abs(np.array(lammps_single_frustration_dataframe["std(decoy_energies)"])-(expected_std_decoy_energy)) < 1.2E-1).all()

def test_contact_pair_decoy_AWSEM_energy_statistics():
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer mutational frustration values
    lammps_mutational_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep=r"\s+")
    lammps_mutational_frustration_dataframe["i"]=lammps_mutational_frustration_dataframe["i"]-1
    lammps_mutational_frustration_dataframe["j"]=lammps_mutational_frustration_dataframe["j"]-1
    ###
    structure=frustratometer.Structure(test_data_path/f'6u5e.pdb',"A")
    model=frustratometer.AWSEM(structure,distance_cutoff_contact=9.5, min_sequence_separation_contact=None, k_electrostatics=0)
    spm = model.sparse_potts_model
    ci, cj = spm['contact_i'], spm['contact_j']
    #Calculate fields
    seq_index = np.array([_AA.find(aa) for aa in structure.sequence])
    seq_len = len(seq_index)
    h = -model.potts_model['h'][range(seq_len), seq_index]

    #Calculate couplings
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)
    j = -model.potts_model['J'][pos1, pos2, aa1, aa2]
    j_prime = j * _dense_mask(model)
    test_contact_energy_matrix=h[pos1]+h[pos2]+j_prime.sum(axis=0)[pos1]+j_prime.sum(axis=0)[pos2]-j_prime[pos1,pos2]
    ###
    # Build dataframe only at contact positions (sparse)
    calculated_mutational_frustration_dataframe=pd.DataFrame({
        "Test_Native_Energy": test_contact_energy_matrix[ci, cj] / 4.184,
        "i": ci,
        "j": cj,
    })
    ###
    # Sparse decoy fluctuation: (N_contacts, 21, 21)
    decoy_fluctuations=(model.decoy_fluctuation(kind='mutational'))/4.184
    N_contacts = decoy_fluctuations.shape[0]
    flat_fluct = decoy_fluctuations.reshape(N_contacts, 21 * 21)
    flat_freq = model.contact_freq.flatten()
    weighted_decoy_fluctations=np.average(flat_fluct, weights=flat_freq, axis=-1)
    calculated_mutational_frustration_dataframe["Weighted_Decoy_Fluctuations"]=weighted_decoy_fluctations
    calculated_mutational_frustration_dataframe["Test_Mean_Decoy_Energy"]=calculated_mutational_frustration_dataframe["Test_Native_Energy"]+calculated_mutational_frustration_dataframe["Weighted_Decoy_Fluctuations"]
    calculated_mutational_frustration_dataframe["STD_Decoy_Energy"]=np.sqrt(np.average((flat_fluct - weighted_decoy_fluctations[:, np.newaxis]) ** 2, weights=flat_freq, axis=-1))
    
    merged_dataframe=calculated_mutational_frustration_dataframe.merge(lammps_mutational_frustration_dataframe,on=["i","j"])

    assert (abs(np.array(merged_dataframe["<decoy_energies>"]-merged_dataframe["Test_Mean_Decoy_Energy"])) < 1.2E-1).all()
    assert (abs(np.array(merged_dataframe["std(decoy_energies)"]-merged_dataframe["STD_Decoy_Energy"])) < 1.2E-1).all()


@pytest.fixture(scope="module")
def structure():
    return frustratometer.Structure(test_data_path/f'1l63.pdb',"A")

@pytest.mark.parametrize("k_electrostatics", [0, 4])
@pytest.mark.parametrize("min_sequence_separation_contact", [2, 10])
@pytest.mark.parametrize("distance_cutoff_contact", [None, 10])
def test_expose_indicators(structure, k_electrostatics, min_sequence_separation_contact, distance_cutoff_contact):
    """ Check that the AWSEM indicators exposed can reproduce the native energy, where E_native = -sum_{i} h_i - sum_{i,j} J_ij = sum_{i} gamma_i * I_i """
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    model=frustratometer.AWSEM(structure,k_electrostatics=k_electrostatics, min_sequence_separation_contact = min_sequence_separation_contact, distance_cutoff_contact = distance_cutoff_contact, expose_indicator_functions=True)
    model_seq_index=np.array([_AA.find(aa) for aa in model.sequence])

    # --- Burial (1D indicators — same shape in sparse and dense) ---
    indicators1D=np.array(model.indicators[0:3])
    true_indicator1D=np.array([indicators1D[:,model_seq_index==i].sum(axis=1) for i in range(21)]).T
    burial_gamma=np.concatenate(model.gamma_array[:3])
    burial_energy_predicted = (burial_gamma * np.concatenate(true_indicator1D)).sum()
    burial_energy_expected = -model.potts_model['h'][range(len(model_seq_index)), model_seq_index].sum()
    assert np.isclose(burial_energy_predicted,burial_energy_expected), f"Expected energy {burial_energy_expected} but got {burial_energy_predicted}"

    # --- Contact (2D indicators — sparse: 1D arrays at contact positions) ---
    ci = model.indicator_contact_i
    cj = model.indicator_contact_j
    indicators2D_sparse = model.indicators[3:]  # list of arrays
    # Aggregate: for each gamma term, sum indicator*gamma over contact pairs grouped by (aa_i, aa_j)
    contact_energy_predicted = 0.0
    for k, contact_gamma in enumerate(model.gamma_array[3:]):
        indicator_vals = indicators2D_sparse[k]
        if indicator_vals.ndim == 1:
            # Sparse 1D indicator at contact positions
            aa_i = model_seq_index[ci]
            aa_j = model_seq_index[cj]
            contact_energy_predicted += (contact_gamma[aa_i, aa_j] * indicator_vals).sum()
        else:
            # Full (L, L) indicator (e.g. electrostatics)
            for i in range(21):
                for j in range(21):
                    contact_energy_predicted += contact_gamma[i, j] * indicator_vals[model_seq_index == i][:, model_seq_index == j].sum()
    contact_energy_expected = model.couplings_energy()
    assert np.isclose(contact_energy_predicted,contact_energy_expected), f"Expected energy {contact_energy_expected} but got {contact_energy_predicted}"


from frustratometer.frustration.frustration import (
    potts_model_dense_to_sparse,
    compute_native_energy,
    compute_native_energy_sparse,
    compute_couplings_energy_sparse,
    compute_singleresidue_decoy_energy_fluctuation,
    compute_singleresidue_decoy_energy_fluctuation_sparse,
    compute_mutational_decoy_energy_fluctuation,
    compute_mutational_decoy_energy_fluctuation_sparse,
    compute_pseudoconfigurational_decoy_energy_fluctuation,
    compute_pseudoconfigurational_decoy_energy_fluctuation_sparse,
    compute_contact_decoy_energy_fluctuation,
    compute_contact_decoy_energy_fluctuation_sparse,
    compute_pair_frustration,
    compute_pair_frustration_sparse,
    sparse_frustration_to_dense,
    compute_mask,
    compute_elec_indicator,
    build_elec_data,
    compute_native_energy_elec,
    apply_elec_correction_singleresidue,
    apply_elec_correction_mutational,
    apply_elec_correction_contact,
    apply_elec_correction_pseudoconfigurational,
    _CHARGES,
)

# --- Sparse cross-validation fixtures (module-scoped, computed once) ---

@pytest.fixture(scope="module")
def sparse_6u5e_density(awsem_6u5e_density):
    return awsem_6u5e_density.sparse_potts_model

# --- Sparse cross-validation tests ---

@pytest.mark.parametrize("kind", ["mutational", "contact", "pseudoconfigurational"])
def test_frustration_dense_false_returns_sparse(kind, awsem_6u5e_density):
    """frustration(dense=False) on a sparse model returns a SparseMatrix aligned to the
    Potts contacts that round-trips (.to_dense(fill=0.0)) to the dense (L, L) result."""
    from frustratometer.classes.Structure import SparseMatrix
    model = awsem_6u5e_density
    assert model._is_sparse, "fixture must be a sparse-Potts model for this contract"

    dense = model.frustration(kind=kind, dense=True)
    sparse = model.frustration(kind=kind, dense=False)

    assert isinstance(sparse, SparseMatrix)
    spm = model.sparse_potts_model
    np.testing.assert_array_equal(sparse.row, spm['contact_i'])
    np.testing.assert_array_equal(sparse.col, spm['contact_j'])
    np.testing.assert_allclose(sparse.to_dense(fill=0.0), dense, atol=1e-6)


def test_vmd_sparse_writes_tcl(tmp_path, monkeypatch):
    """vmd() works on a sparse-mask model: the SparseMatrix mask/distance are densified
    inside write_tcl_script so the pair-drawing path runs and a tcl file is written."""
    from frustratometer.classes.Structure import SparseMatrix
    structure = frustratometer.Structure((test_data_path / '6u5e.pdb').resolve(), "A")
    model = frustratometer.AWSEM(structure, k_electrostatics=0)
    assert isinstance(model.mask, SparseMatrix), "sparse Structure should give a SparseMatrix mask"

    monkeypatch.setattr(frustratometer.frustration, "call_vmd", lambda *a, **k: None)
    model.vmd(pair='mutational', debug_directory=tmp_path)

    tcl = tmp_path / 'frustration.tcl'
    assert tcl.exists() and tcl.stat().st_size > 0
    # 'mol delrep' is written after the contact-drawing loop -> the pair branch ran to completion.
    assert 'mol delrep' in tcl.read_text()


def test_total_frustration_sparse_runs(awsem_6u5e_density):
    """total_frustration() runs on a sparse model via the sparse-native path."""
    model = awsem_6u5e_density
    assert model._is_sparse
    val = model.total_frustration(n_decoys=200)
    assert np.isfinite(val)


@pytest.fixture(scope="module")
def elec_setup(awsem_6u5e_density, sparse_6u5e_density):
    """
    Build electrostatics test data from the real 6u5e model.

    Takes the k_elec=0 model, constructs J_elec using compute_elec_indicator
    (same formula as build_elec_data), adds it to J to create 'combined' dense
    Potts model.  Then builds elec_data for the sparse path.

    Returns dict with: model, spm, elec_data, combined_potts, mask, k_elec.
    """
    model = awsem_6u5e_density
    spm = sparse_6u5e_density
    distance_matrix = model.distance_matrix.to_dense(fill=np.inf)
    mask = _dense_mask(model)
    k_elec = 4 * 4.184  # same value used in AWSEM test configs
    screening_length = 10.0
    min_sep_elec = 1

    # Build indicator (includes -k factor and elec_mask)
    indicator = compute_elec_indicator(distance_matrix, k_elec, screening_length)
    elec_mask = compute_mask(distance_matrix,
                             maximum_contact_distance=None,
                             minimum_sequence_separation=min_sep_elec)
    indicator = indicator * elec_mask

    # Construct J_elec: indicator[i,j] * q_a * q_b
    charges = _CHARGES
    J_elec = indicator[:, :, np.newaxis, np.newaxis] * charges[np.newaxis, np.newaxis, :, np.newaxis] * charges[np.newaxis, np.newaxis, np.newaxis, :]

    # Combined dense Potts model (contact + electrostatics)
    combined_potts = {
        'h': model.potts_model['h'].copy(),
        'J': model.potts_model['J'] + J_elec,
    }

    # Build elec_data for sparse path
    elec_data = build_elec_data(distance_matrix, mask, model.sequence, spm,
                                k_elec, screening_length, min_sep_elec)

    return {
        'model': model,
        'spm': spm,
        'elec_data': elec_data,
        'combined_potts': combined_potts,
        'mask': mask,
    }


def test_elec_native_energy(elec_setup):
    """Sparse native energy + electrostatic energy must match dense combined model."""
    s = elec_setup
    model, spm, elec_data, mask = s['model'], s['spm'], s['elec_data'], s['mask']

    dense_energy = compute_native_energy(model.sequence, s['combined_potts'], mask)
    sparse_energy = compute_native_energy_sparse(model.sequence, spm)
    elec_energy = compute_native_energy_elec(model.sequence, elec_data, mask)

    np.testing.assert_allclose(sparse_energy + elec_energy, dense_energy, rtol=1e-10)


@pytest.mark.parametrize("kind", ["singleresidue", "mutational", "contact", "pseudoconfigurational"])
def test_elec_decoy_correction(kind, elec_setup):
    """Sparse decoy + electrostatic correction must match dense combined decoy."""
    s = elec_setup
    model, spm, elec_data, mask = s['model'], s['spm'], s['elec_data'], s['mask']
    combined_potts = s['combined_potts']
    seq = model.sequence

    # Dense decoy from combined (contact + elec) Potts model
    if kind == 'singleresidue':
        dense_decoy = compute_singleresidue_decoy_energy_fluctuation(seq, combined_potts, mask)
        sparse_decoy = compute_singleresidue_decoy_energy_fluctuation_sparse(seq, spm)
        corrected = apply_elec_correction_singleresidue(sparse_decoy, elec_data)
        np.testing.assert_allclose(corrected, dense_decoy, atol=1e-4)
    elif kind == 'mutational':
        dense_decoy = compute_mutational_decoy_energy_fluctuation(seq, combined_potts, mask)
        sparse_decoy = compute_mutational_decoy_energy_fluctuation_sparse(seq, spm)
        corrected = apply_elec_correction_mutational(sparse_decoy, spm, elec_data)
        ci, cj = spm['contact_i'], spm['contact_j']
        np.testing.assert_allclose(corrected, dense_decoy[ci, cj], atol=1e-4)
    elif kind == 'contact':
        dense_decoy = compute_contact_decoy_energy_fluctuation(seq, combined_potts, mask)
        sparse_decoy = compute_contact_decoy_energy_fluctuation_sparse(seq, spm)
        corrected = apply_elec_correction_contact(sparse_decoy, spm, elec_data)
        ci, cj = spm['contact_i'], spm['contact_j']
        np.testing.assert_allclose(corrected, dense_decoy[ci, cj], atol=1e-4)
    elif kind == 'pseudoconfigurational':
        dense_decoy = compute_pseudoconfigurational_decoy_energy_fluctuation(seq, combined_potts, mask)
        sparse_decoy = compute_pseudoconfigurational_decoy_energy_fluctuation_sparse(seq, spm, mask.mean())
        corrected = apply_elec_correction_pseudoconfigurational(sparse_decoy, spm, elec_data)
        ci, cj = spm['contact_i'], spm['contact_j']
        np.testing.assert_allclose(corrected, dense_decoy[ci, cj], atol=1e-4)


if __name__ == "__main__":
    pytest.main()
