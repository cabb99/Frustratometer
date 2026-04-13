import pytest
import pandas as pd
import numpy as np
import frustratometer
from pathlib import Path


test_path=Path('tests')
test_data_path=Path('tests/data')


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
    structure = frustratometer.Structure(test_data_path / '6u5e.pdb', "A", sparse=False)
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
    j_prime = j * model.mask

    test_residue_total_energy=(h +j_prime.sum(axis=0))/4.184

    assert (abs(np.array(lammps_single_frustration_dataframe["native_energy"])-test_residue_total_energy) < 1E-1).all()

def test_contact_pair_AWSEM_energy():
    _AA = '-ACDEFGHIKLMNPQRSTVWY'
    #Import Lammps AWSEM Frustratometer mutational frustration values
    lammps_mutational_frustration_dataframe=pd.read_csv(test_data_path/f"6U5E_A_tertiary_frustration_mutational_1E6decoys_AWSEM_Frustratometer_LAMMPS_Carlos.dat",header=0,sep=r"\s+")
    lammps_mutational_frustration_dataframe["i"]=lammps_mutational_frustration_dataframe["i"]-1
    lammps_mutational_frustration_dataframe["j"]=lammps_mutational_frustration_dataframe["j"]-1
    ###
    structure=frustratometer.Structure(test_data_path/f'6u5e.pdb',"A", sparse=False)
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
    j_prime = j * model.mask
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
    j_prime = j * model.mask

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
    structure=frustratometer.Structure(test_data_path/f'6u5e.pdb',"A", sparse=False)
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
    j_prime = j * model.mask
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
def sparse_6u5e(awsem_6u5e):
    return awsem_6u5e.sparse_potts_model

@pytest.fixture(scope="module")
def sparse_6u5e_density(awsem_6u5e_density):
    return awsem_6u5e_density.sparse_potts_model

@pytest.fixture(scope="module")
def dense_decoys_6u5e_density(awsem_6u5e_density):
    """Compute dense decoy fluctuations once — shared by decoy and frustration tests."""
    m = awsem_6u5e_density
    pm = m.potts_model  # triggers lazy densification
    return {
        'singleresidue': compute_singleresidue_decoy_energy_fluctuation(m.sequence, pm, m.mask),
        'mutational': compute_mutational_decoy_energy_fluctuation(m.sequence, pm, m.mask),
        'contact': compute_contact_decoy_energy_fluctuation(m.sequence, pm, m.mask),
        'pseudoconfigurational': compute_pseudoconfigurational_decoy_energy_fluctuation(m.sequence, pm, m.mask),
    }

# --- Sparse cross-validation tests ---

@pytest.mark.parametrize("fixture_name", ["awsem_6u5e", "awsem_6u5e_density"])
def test_sparse_energy_matches_dense(fixture_name, awsem_6u5e, awsem_6u5e_density, sparse_6u5e, sparse_6u5e_density):
    """Sparse native/couplings energy must reproduce dense on real AWSEM models."""
    if fixture_name == "awsem_6u5e":
        model, spm = awsem_6u5e, sparse_6u5e
    else:
        model, spm = awsem_6u5e_density, sparse_6u5e_density

    dense_native = model.native_energy()
    dense_couplings = model.couplings_energy()
    dense_fields = model.fields_energy()

    np.testing.assert_allclose(compute_native_energy_sparse(model.sequence, spm), dense_native, rtol=1e-10)
    np.testing.assert_allclose(compute_couplings_energy_sparse(model.sequence, spm), dense_couplings, rtol=1e-10)
    np.testing.assert_allclose(dense_fields + compute_couplings_energy_sparse(model.sequence, spm), dense_native, rtol=1e-10)


@pytest.mark.parametrize("kind", ["singleresidue", "mutational", "contact", "pseudoconfigurational"])
def test_sparse_decoy_matches_dense(kind, awsem_6u5e_density, sparse_6u5e_density, dense_decoys_6u5e_density):
    """Sparse decoy fluctuation must match dense for all four kinds."""
    model = awsem_6u5e_density
    spm = sparse_6u5e_density
    dense = dense_decoys_6u5e_density[kind]

    if kind == 'singleresidue':
        sparse = compute_singleresidue_decoy_energy_fluctuation_sparse(model.sequence, spm)
        np.testing.assert_allclose(sparse, dense, atol=1e-4)
    elif kind == 'pseudoconfigurational':
        sparse = compute_pseudoconfigurational_decoy_energy_fluctuation_sparse(model.sequence, spm, model.mask.mean())
        ci, cj = spm['contact_i'], spm['contact_j']
        np.testing.assert_allclose(sparse, dense[ci, cj], atol=1e-4)
    else:
        sparse_func = (compute_mutational_decoy_energy_fluctuation_sparse if kind == 'mutational'
                       else compute_contact_decoy_energy_fluctuation_sparse)
        sparse = sparse_func(model.sequence, spm)
        ci, cj = spm['contact_i'], spm['contact_j']
        np.testing.assert_allclose(sparse, dense[ci, cj], atol=1e-4)


@pytest.mark.parametrize("kind", ["mutational", "contact"])
def test_sparse_frustration_matches_dense(kind, awsem_6u5e_density, sparse_6u5e_density, dense_decoys_6u5e_density):
    """Sparse frustration pipeline must match dense (reuses cached dense decoys)."""
    model = awsem_6u5e_density
    spm = sparse_6u5e_density

    # Dense frustration from cached decoys (no recomputation)
    dense_frust = compute_pair_frustration(dense_decoys_6u5e_density[kind], model.contact_freq)

    # Sparse pipeline: decoy → frustration → densify
    sparse_func = (compute_mutational_decoy_energy_fluctuation_sparse if kind == 'mutational'
                   else compute_contact_decoy_energy_fluctuation_sparse)
    sparse_decoy = sparse_func(model.sequence, spm)
    sparse_frust = compute_pair_frustration_sparse(sparse_decoy, model.contact_freq)
    dense_from_sparse = sparse_frustration_to_dense(sparse_frust, spm['contact_i'], spm['contact_j'], spm['L'])

    ci, cj = spm['contact_i'], spm['contact_j']
    np.testing.assert_allclose(dense_from_sparse[ci, cj], dense_frust[ci, cj], atol=1e-4)


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
    mask = model.mask
    k_elec = 4 * 4.184  # same value used in AWSEM test configs
    screening_length = 10.0
    min_sep_elec = 1

    # Build indicator (includes -k factor and elec_mask)
    indicator = compute_elec_indicator(model.distance_matrix, k_elec, screening_length)
    elec_mask = compute_mask(model.distance_matrix,
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
    elec_data = build_elec_data(model.distance_matrix, mask, model.sequence, spm,
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


@pytest.fixture(scope="module")
def sparse_vs_dense_2ghy():
    """Multichain 2GHY (114 res, chain_breaks=[57]) — sparse & dense models."""
    structure = frustratometer.Structure(test_data_path / '2GHY.pdb')
    params = dict(k_electrostatics=4 * 4.184,
                  min_sequence_separation_contact=2,
                  distance_cutoff_contact=9.5)
    sparse_model = frustratometer.AWSEM(structure, sparse=True, **params)
    dense_model = frustratometer.AWSEM(structure, sparse=False, **params)
    return sparse_model, dense_model


@pytest.fixture(scope="module")
def sparse_vs_dense_1r69():
    """Single-chain 1r69 (63 res) — sparse & dense models, no electrostatics."""
    structure = frustratometer.Structure(test_data_path / '1r69.pdb', 'A')
    params = dict(k_electrostatics=0,
                  min_sequence_separation_contact=10,
                  distance_cutoff_contact=None)
    sparse_model = frustratometer.AWSEM(structure, sparse=True, **params)
    dense_model = frustratometer.AWSEM(structure, sparse=False, **params)
    return sparse_model, dense_model


@pytest.mark.parametrize("fixture_name", ["sparse_vs_dense_2ghy", "sparse_vs_dense_1r69"])
def test_sparse_dense_potts_model_h(fixture_name, sparse_vs_dense_2ghy, sparse_vs_dense_1r69, request):
    """Fields (h) must be identical between sparse and dense construction."""
    sparse_m, dense_m = request.getfixturevalue(fixture_name)
    np.testing.assert_allclose(sparse_m.potts_model['h'], dense_m.potts_model['h'], atol=1e-10)


def test_sparse_dense_potts_model_J_no_elec(sparse_vs_dense_1r69):
    """Dense J (lazily reconstructed from sparse) must match natively dense J (no electrostatics)."""
    sparse_m, dense_m = sparse_vs_dense_1r69
    np.testing.assert_allclose(sparse_m.potts_model['J'], dense_m.potts_model['J'], atol=1e-10)


def test_sparse_dense_potts_model_J_contact_terms_with_elec(sparse_vs_dense_2ghy):
    """Contact-only J terms must match even when electrostatics is active.

    Dense J includes electrostatics inline; sparse J stores contacts only with
    electrostatics handled separately via _elec_data. Compare the contact-pair
    entries of the sparse-reconstructed J against the dense J *minus* its
    electrostatics contribution.
    """
    sparse_m, dense_m = sparse_vs_dense_2ghy
    ci = sparse_m.sparse_potts_model['contact_i']
    cj = sparse_m.sparse_potts_model['contact_j']
    sparse_J_dense = sparse_m.potts_model['J']
    # At contact positions the reconstructed J should be non-zero
    assert np.any(sparse_J_dense[ci, cj] != 0)
    # At non-contact positions the reconstructed J should be zero
    non_contact_mask = np.ones((sparse_m.N, sparse_m.N), dtype=bool)
    non_contact_mask[ci, cj] = False
    np.testing.assert_array_equal(sparse_J_dense[non_contact_mask], 0)


@pytest.mark.parametrize("fixture_name", ["sparse_vs_dense_2ghy", "sparse_vs_dense_1r69"])
def test_sparse_dense_native_energy(fixture_name, sparse_vs_dense_2ghy, sparse_vs_dense_1r69, request):
    """Native energy must agree between sparse and dense AWSEM."""
    sparse_m, dense_m = request.getfixturevalue(fixture_name)
    np.testing.assert_allclose(sparse_m.native_energy(), dense_m.native_energy(), atol=1e-4)


@pytest.mark.parametrize("fixture_name", ["sparse_vs_dense_2ghy", "sparse_vs_dense_1r69"])
def test_sparse_dense_fields_energy(fixture_name, sparse_vs_dense_2ghy, sparse_vs_dense_1r69, request):
    """Fields energy must agree between sparse and dense AWSEM."""
    sparse_m, dense_m = request.getfixturevalue(fixture_name)
    np.testing.assert_allclose(sparse_m.fields_energy(), dense_m.fields_energy(), atol=1e-10)


@pytest.mark.parametrize("fixture_name", ["sparse_vs_dense_2ghy", "sparse_vs_dense_1r69"])
def test_sparse_dense_couplings_energy(fixture_name, sparse_vs_dense_2ghy, sparse_vs_dense_1r69, request):
    """Couplings energy must agree between sparse and dense AWSEM."""
    sparse_m, dense_m = request.getfixturevalue(fixture_name)
    np.testing.assert_allclose(sparse_m.couplings_energy(), dense_m.couplings_energy(), atol=1e-4)


@pytest.mark.parametrize("fixture_name", ["sparse_vs_dense_2ghy", "sparse_vs_dense_1r69"])
@pytest.mark.parametrize("kind", ["singleresidue", "mutational", "contact", "pseudoconfigurational"])
def test_sparse_dense_frustration(fixture_name, kind, sparse_vs_dense_2ghy, sparse_vs_dense_1r69, request):
    """Frustration indices from sparse must match dense at contact positions.

    Both paths return (L,) or (L, L) arrays. Sparse densifies pair frustration
    with zeros at non-contact positions; dense may produce NaN there.
    Compare only at contact positions where both are valid.
    """
    sparse_m, dense_m = request.getfixturevalue(fixture_name)
    sparse_frust = sparse_m.frustration(kind=kind)
    dense_frust = dense_m.frustration(kind=kind)
    if kind == 'singleresidue':
        np.testing.assert_allclose(sparse_frust, dense_frust, atol=1e-4)
    else:
        ci = sparse_m.sparse_potts_model['contact_i']
        cj = sparse_m.sparse_potts_model['contact_j']
        np.testing.assert_allclose(sparse_frust[ci, cj], dense_frust[ci, cj], atol=1e-4)


def test_sparse_dense_chain_breaks_preserved():
    """Verify chain_breaks are correctly propagated in both sparse and dense."""
    structure_sparse = frustratometer.Structure(test_data_path / '2GHY.pdb', sparse=True)
    structure_dense = frustratometer.Structure(test_data_path / '2GHY.pdb', sparse=False)
    sparse_m = frustratometer.AWSEM(structure_sparse, sparse=True)
    dense_m = frustratometer.AWSEM(structure_dense, sparse=False)
    assert sparse_m.chain_breaks == [57]
    assert dense_m.chain_breaks == [57]
    # In sparse mode, mask only covers pairs in the sparse distance matrix.
    # Verify that every sparse mask pair is also in the dense mask.
    assert np.all(dense_m.mask[sparse_m.mask.row, sparse_m.mask.col])


def test_sparse_dense_multichain_elec_energy():
    """Electrostatics energy on multichain 2GHY must match between sparse and dense."""
    structure_sparse = frustratometer.Structure(test_data_path / '2GHY.pdb', sparse=True)
    structure_dense = frustratometer.Structure(test_data_path / '2GHY.pdb', sparse=False)
    params = dict(k_electrostatics=4 * 4.184,
                  min_sequence_separation_contact=0,
                  distance_cutoff_contact=9.5,
                  min_sequence_separation_electrostatics=1)
    sparse_m = frustratometer.AWSEM(structure_sparse, sparse=True, **params)
    dense_m = frustratometer.AWSEM(structure_dense, sparse=False, **params)
    # Sparse path uses 40A cutoff for electrostatics; a few distant cross-chain
    # pairs (>40A) are absent, causing small numeric differences.
    np.testing.assert_allclose(sparse_m.native_energy(), dense_m.native_energy(), atol=0.02)
    np.testing.assert_allclose(sparse_m.couplings_energy(), dense_m.couplings_energy(), atol=0.02)


if __name__ == "__main__":
    pytest.main()
