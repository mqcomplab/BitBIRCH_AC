import pandas as pd
import numpy as np
import glob as glob
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import time
import pickle 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

###########################################################
# Generation of RDKit fingerprints from SMILES ############
###########################################################

input_path = 'data'  # Path to the folder containing the CSV files

# Reading data bases 
for file in glob.glob(os.path.join(input_path, '*CHEMBL*.csv')):
 name = file.split('.')[0]
 data = pd.read_csv(file)

 # Generating mol objects from smiles column in the csv

 data['Mol'] = data['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
 
 # Filter out any failed molecule conversions
 data = data[data['Mol'].notnull()]
 
 # Taking the properties in the csv
 prop = data['exp_mean [nM]'].values.tolist()

 # Taking the assigned split of the molecule 
 split = data['split'].values.tolist()

 # Taking the cliff / no cliff column
 cliff = data['cliff_mol'].values.tolist()

 # Generating the RDKit Fingerprints
 fp_rdkit = []
 for mol in data.loc[:,'Mol']:
  new_fp = np.array([])
  rdkit.DataStructs.cDataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(mol), new_fp)
  fp_rdkit.append(new_fp)

 # Generating the ECPF4 Fingerprints
 fp_ecpf4 = []
 for mol in data.loc[:,'Mol']:
  new_fp = np.array([])
  rdkit.DataStructs.cDataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024), new_fp)
  fp_ecpf4.append(new_fp)

 # Generating the MACCS Fingerprints
 fp_maccs = []
 for mol in data.loc[:,'Mol']:
  new_fp = np.array([])
  rdkit.DataStructs.cDataStructs.ConvertToNumpyArray(Chem.MACCSkeys.GenMACCSKeys(mol), new_fp)
  fp_maccs.append(new_fp)

 # Generating a dictionary with the three types of fingerprints, property and the test/train split
 data = {'split' : split,
         'prop' : prop,
         'cliff': cliff, 
         'RDKIT': fp_rdkit,
         'ECFP': fp_ecpf4,
         'MACCS': fp_maccs,
         }

 with open(name + '_fp.pkl', 'wb') as f:
  pickle.dump(data, f)

