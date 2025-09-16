'''Main script to run AC analysis with various parameters, including ordering and recursion options. Useful for benchmarking different configurations.

Key functions include:
- set_order: Sets the order in which fingerprints are passed to the BB tree. 4 options available: random, decreasing_sum, increasing_sum, identity
- process_tani_file: Processes a single Tanimoto matrix file to compute ACs using both pairwise comparison and BitBIRCH clustering. This is paralelized across multiple files.
- close_analysis: If recursion is enabled, this function takes the remaining molecules after initial AC detection and re-runs the BB clustering to find additional ACs.
- offset can be used to relax the similarity threshold for BB clustering, potentially helpful when accuracy is needed.

Instructions to run the script:
- You can specify multiple orders and recursion options via command line arguments.
- Results are saved to CSV files named based on the parameters used, allowing easy comparison.
Example command (to use increasing_sum, increasing_sum_cent orders with recursion):
python run_test.py --order increasing_sum increasing_sum_cent --recursive True --max_workers 20

Check the argparse section for all available options and defaults.'''

import numpy as np
import sys
import os
# Change this line
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Then import
from bb_utils.help_funcs import *
from bb_utils.help_funcs import *
import glob
import os
import pandas as pd
import time
from sklearn.metrics import pairwise_distances
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


# Candidate BB AC implementations
bb_options = ['bb_ac_minmax', 'bb_int_minmax', 'bb_rcent_minmax']

# bb_ac_minmax: Traditional BB, just with added support to handle properties
# bb_int_minmax: The "centroid" information contains the linear sum of the cluster
# bb_rcent_minmax: Real centroids are used: linear_sum/n_mols

# Choose a BB AC implementation:
import bb_utils.bb_rcent as bb

# Finding exact ACs
input = 'files'
output = 'results'
offsets= [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # Offset for the threshold
# Similarity threshold to calculate ACs
threshold = np.arange(0.90, 1.00, 0.01)  # [0.90, 0.91, ..., 0.99]
tani_files = glob.glob(os.path.join(input, 'tani_matrix_*.npy'))

'''I was really clumsy while naming the files based on used of offset, order and recursion 
so I have tried to automate the flow'''

def parse_arguments():

    parser = argparse.ArgumentParser(description="AC cluster analysis with tunable parameters.")

    parser.add_argument("--order", type=str, nargs='*', default=['increasing_sum'],
                        choices=['random', 'decreasing_sum', 'increasing_sum', 'increasing_sum_cent', 'identity'],
                        help="Order in which fingerprints are passed to the BB tree (default: increasing_sum).")

    parser.add_argument('--use_offsets', action='store_true', default=False,
                        help="Whether to use offsets in the analysis.")

    parser.add_argument('--offsets', type=float, nargs='*', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                        help="List of offsets to use in the analysis.")
    
    parser.add_argument('--recursive', nargs='*', choices=['True', 'False'], default=['False'],
                        help="Whether to use recursion in the analysis.")
    
    parser.add_argument('--max_workers', type=int, default=30,
                        help="Number of parallel workers to use (default: 30).")
    
    return parser.parse_args()


'''Now this can generate a filename based on the parameters provided, so that I don't have to manually change the filename everytime'''
def generate_filename(order_method, use_offsets, recursive):
     '''Generates a filename based on the provided parameters.'''

     order_map = {
          'random': 'Random',
          'decreasing_sum': 'Decreasing_Sum',
          'increasing_sum': 'Increasing_Sum',
          'increasing_sum_cent': 'Increasing_Sum_Cent',
          'identity': 'Identity'
     }

     order_str = order_map.get(order_method, order_method)
     offset_str = 'With_Offsets' if use_offsets else 'No_Offsets'
     recursive_str = 'With_Recur' if recursive else 'No_Recur'

     return f"{order_str}_{offset_str}_{recursive_str}.csv"

def set_order(fps, order): # Sets the order in which fingerprints are passed to the BB tree
         if order == 'random':
             ##random order
             order_inds = list(range(len(fps)))
             new_order = np.random.permutation(order_inds)
         elif order == 'decreasing_sum':
             ##order by nbits in fps (max to min)
             row_sums = np.sum(fps, axis=1)
             new_order = np.argsort(row_sums)[::-1]

             '''Order becomes significantly important when the fps are not random.'''

         elif order == 'increasing_sum':
             ##order by nbits in fps (min to max)
             row_sums = np.sum(fps, axis=1)
             new_order = np.argsort(row_sums)
         elif order == 'increasing_sum_cent':
             # Calculate the centroid of all fps
             centroid = np.mean(fps, axis=0)
             row_sums = np.sum(fps, axis=1)
             # Compute pair_sim to centroid for each fps using bb.pair_sim
             sims = np.array([bb.pair_sim(fp, centroid) for fp in fps])
             combination=0 * row_sums + 1 * sims * row_sums
             new_order = np.argsort(combination)
         else:
             #identity (original order)
             new_order = list(range(len(fps)))
         return new_order 

def close_analysis(fps, props, threshold, offset): # Function to take the fps and props which were not removed as ACs in the first pass and run them through BB again to find more ACs
             brc = bb.BitBirch(branching_factor = 50, threshold = threshold-offset)
             brc.fit(fps, props)
             
             inds = brc.get_cluster_mol_ids()
             
             tot = 0
             total_indices = []
             for k in inds:
                 if len(k) > 1:
                     n_pairs, new_inds = count_pairs(fps[k], props[k], threshold, k) # Count pairs of ACs in the cluster at a specified threshold
                     tot += n_pairs
                     total_indices += new_inds
             return tot, total_indices

def process_tani_file(tani_path, th, offset, order_method, recursive): 
     
    filename = os.path.basename(tani_path)
    suffix = filename[len('tani_matrix_'):-4]
    prop_diff_path = os.path.join(input, f'prop_diff_matrix_{suffix}.npy')

    '''Pairwise calculation of ACs
    If both the similarity and property difference conditions are met, then the pair is an AC.'''

    tani_matrix = np.load(tani_path)
    prop_diffs = np.load(prop_diff_path)
    tani_matrix[tani_matrix >= th] = 1
    tani_matrix[tani_matrix < th] = 0
    ac_matrix = tani_matrix * prop_diffs
    i, j = np.where(np.triu(ac_matrix, k=1) == 1)
    ac_pairs = list(zip(i, j))
    n_acs = len(ac_pairs)
    

    '''Using BitBIRCH to identify ACs, with ordering and recursion options,
       makes the BitBIRCH tree then finds clusters, counts ACs in each cluster.
       If recursion is enabled, it removes found ACs and re-runs on remaining molecules.'''

    fps = np.load(f'files/fps_{suffix}.npy')
    props = np.load(f'files/props_{suffix}.npy')
    brc = bb.BitBirch(branching_factor=50, threshold=th-offset)
    new_order = set_order(fps, order_method)
    fps = fps[new_order]
    props = props[new_order]
    brc.fit(fps, props)
    inds = brc.get_cluster_mol_ids()


    n_acs_bb = 0
    bb_ac_inds = []
    for cluster_inds in inds:
        if len(cluster_inds) > 1:
            #p_vals = props[cluster_inds]
            #diffs = np.abs(p_vals[:, None] - p_vals[None, :]).flatten()
            n, inds_found = count_pairs(fps[cluster_inds], props[cluster_inds], th, cluster_inds)
            n_acs_bb += n
            bb_ac_inds += inds_found
    
    print('One run:', n_acs_bb, n_acs, "Threshold:", th, "Suffix:", suffix)
    # (recursive part omitted for brevity, add if needed)
    
    if recursive:
         # Load fps
         #fps = np.load(f'files/fps_{suffix}.npy')

         # Load log props
         #props = np.load(f'files/props_{suffix}.npy')

         mask = np.ones(len(fps), dtype=bool)
         mask[bb_ac_inds] = False
         new_fps = fps[mask]
         new_props = props[mask]
         

         
         #n_acs_bb = 0
         #bb_ac_inds = []
         n = 1
         new_order = set_order(new_fps, order_method)
         new_fps = new_fps[new_order]
         new_props = new_props[new_order]
         r = 1
         while n:
             print(f'Recursion {r}')
             n, total_indices = close_analysis(new_fps, new_props, threshold=th, offset=offset)
             print(f'Found {n} ACs in recursion {r}')
             n_acs_bb += n
             bb_ac_inds += total_indices
             mask = np.ones(len(new_fps), dtype=bool)
             mask[total_indices] = False
             new_fps = new_fps[mask]
             new_props = new_props[mask]
             new_order = set_order(new_fps, order_method)
             new_fps = new_fps[new_order]
             new_props = new_props[new_order]

             r += 1
    
    ratio = n_acs_bb / n_acs if n_acs != 0 else -1
    print('ratio:', ratio)
    
    return {'Threshold': th, 'suffix': suffix, 'ratio': ratio, 'offset': offset, 'order': order_method, 'recursive': recursive}

if __name__ == "__main__":
    args = parse_arguments()

    recursive_options = [r.lower() == 'true' for r in args.recursive]


    # Determine if offsets should be used
    if args.use_offsets:
        offsets = args.offsets
    else:
        offsets = [0.0]

    print(f"Configuration:")
    print(f"  Order method: {args.order}")
    print(f"  Use offsets: {args.use_offsets}")
    print(f"  Offsets: {offsets}")
    print(f"  Recursive: {args.recursive}")
    print(f"  Max workers: {args.max_workers}")
    print()

    for order_method in args.order:
         for recursive in recursive_options:
              
        

           csv_filename = generate_filename(order_method, args.use_offsets, recursive)
           csv_path = os.path.join(output, csv_filename)
       
           
       
           results = []
           start_time = time.time()
       
       
           for th in threshold:
               for offset in offsets:
                   with ProcessPoolExecutor(max_workers=30) as executor:
                       futures = [executor.submit(process_tani_file, tani_path, th, offset, order_method, recursive) for tani_path in tani_files]
                       for future in as_completed(futures):
                           results.append(future.result())
       
           end_time = time.time()
           elapsed_time = end_time - start_time
           print(f"Elapsed time for {order_method} (recursive={recursive}): {elapsed_time:.2f} seconds")
       
           df = pd.DataFrame(results)
       
           # If the file exists, append without writing the header; otherwise, write header
           if os.path.exists(csv_path):
               df.to_csv(csv_path, mode='a', header=False, index=False)
           else:
               df.to_csv(csv_path, index=False)
           print(df)