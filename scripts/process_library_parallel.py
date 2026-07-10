'''Once the .pkl files containing the fingerprints and properties are generated, 
this script processes those files to create numpy arrays for fingerprints, properties, Tanimoto similarity matrices,
and property difference matrices.'''

import numpy as np
import pandas as pd
import os
from pathlib import Path

def pair_sim(mol1, mol2):
    return np.dot(mol1, mol2) / (np.dot(mol1, mol1) + np.dot(mol2, mol2) - np.dot(mol1, mol2))

# Specify input and output directories
parent_dir = Path(__file__).resolve().parent.parent
input_dir = parent_dir / "new_data_pkl_copy"
output_dir = parent_dir / "new_data_files_copy"
os.makedirs(output_dir, exist_ok=True)

# List of fingerprint types
fp_types = ['ECFP', 'MACCS', 'RDKIT']

# Process all .pkl files in the pkl directory for each fingerprint type
for library in os.listdir(input_dir):
    if library.endswith('.pkl'):
        name = library.split('.')[0]
        obj = pd.read_pickle(os.path.join(input_dir, library))
        for fp_type in fp_types:
            fps = np.array(obj[fp_type])
            props = np.array(obj['prop'])  # already in log scale

            np.save(os.path.join(output_dir, f'fps_{name}_{fp_type}.npy'), fps)
            np.save(os.path.join(output_dir, f'props_{name}_{fp_type}.npy'), props)
            print(f"Saved fps and props for {name} ({fp_type})")

            # sim_matrix = []
            # prop_matrix = []
            # for i in range(len(props)):
            #     prop_matrix.append([])
            #     sim_matrix.append([])
            #     for j in range(len(props)):
            #         if abs(props[i] - props[j]) >= 1:
            #             prop_matrix[-1].append(1)
            #         else:
            #             prop_matrix[-1].append(0)
            #         if i == j:
            #             sim_matrix[-1].append(0)
            #         else:
            #             sim_matrix[-1].append(pair_sim(fps[i], fps[j]))
            # np.save(os.path.join(output_dir, f'tani_matrix_{name}_{fp_type}.npy'), sim_matrix)
            # np.save(os.path.join(output_dir, f'prop_diff_matrix_{name}_{fp_type}.npy'), prop_matrix)