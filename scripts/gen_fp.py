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
from pathlib import Path

###########################################################
# Generation of RDKit fingerprints from SMILES ############
###########################################################

parent_dir = Path(__file__).resolve().parent.parent
input_path = parent_dir / "data"

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

 output_file = os.path.join(parent_dir/'pkl', os.path.basename(name) + '_fp.pkl')
 print(f"Saving fingerprints to: {output_file}")
 with open(output_file, 'wb') as f:
     pickle.dump(data, f)
 
 print(f"Completed processing {os.path.basename(file)}\n")

