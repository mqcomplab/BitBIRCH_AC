'''Main code for generating smooth clusters and visualizations for a given dataset. Use the following functions:

- FingerprintClusterAnalyzer: Main class to handle clustering and visualization for different fingerprint types
- load_fingerprint_data: Load fingerprint and property data
- perform_clustering (Smooth and Cliff clusters): Perform clustering for all fingerprint types and thresholds
- visualize_molecules: Visualize molecular structures for a specific fingerprint type, threshold, and cluster
- compare_fingerprint_types: Compare the same cluster across different fingerprint types
- save_results_to_csv: Save all clustering results to CSV (View details such as number of molecules, property mean/std, etc.)

'''



import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from bb_utils import bb_ac_new
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from IPython.display import SVG, display

class FingerprintClusterAnalyzer:
    """
    A comprehensive class to handle clustering and visualization for different fingerprint types
    """
    
    def __init__(self, fingerprint_types=['ECFP', 'MACCS', 'RDKIT'], thresholds=[0.95], top=10):
        self.fingerprint_types = fingerprint_types
        self.thresholds = thresholds
        self.top = top
        self.results = []
        self.cluster_data = {}  # Store cluster results for each fingerprint type
        
    def load_fingerprint_data(self, data_prefix='files/fps_CHEMBL234_Ki_fp_', prop_prefix='files/props_CHEMBL234_Ki_fp_'): # Needs to remove hard coding, please change the file paths accordingly. 
        """Load fingerprint and property data for all fingerprint types"""
        self.fingerprint_data = {}
        self.property_data = {}
        
        for fp_type in self.fingerprint_types:
            try:
                fps_file = f'{data_prefix}{fp_type}.npy'
                props_file = f'{prop_prefix}{fp_type}.npy'
                
                self.fingerprint_data[fp_type] = np.load(fps_file)
                self.property_data[fp_type] = np.load(props_file)
                
                print(f"Loaded {fp_type}: {self.fingerprint_data[fp_type].shape[0]} molecules")
                
            except FileNotFoundError as e:
                print(f"Warning: Could not load {fp_type} data - {e}")
                continue
                
    def perform_clustering(self):
        """Perform clustering for all fingerprint types and thresholds"""
        self.cluster_data = defaultdict(dict)
        
        for fp_type in self.fingerprint_types:
            if fp_type not in self.fingerprint_data:
                print(f"Skipping {fp_type} - data not loaded")
                continue
                
            fps = self.fingerprint_data[fp_type]
            props = self.property_data[fp_type]
            
            print(f"\nProcessing {fp_type} fingerprints...")

            for th in self.thresholds:
                print(f"  Threshold: {th}")
                
                # Activity cliffs clustering
                bb_ac_new.activity_cliffs = True
                brc_cliff = bb_ac_new.BitBirch(threshold=th)
                brc_cliff.fit(fps, props)
                
                # Smooth clustering
                bb_ac_new.activity_cliffs = False
                brc_smooth = bb_ac_new.BitBirch(threshold=th)
                brc_smooth.fit(fps, props)
                
                cliff_ids = brc_cliff.get_cluster_mol_ids()
                smooth_ids = brc_smooth.get_cluster_mol_ids()
                
                # Store cluster data
                key = f"{fp_type}_{th}"
                self.cluster_data[key] = {
                    'fingerprint_type': fp_type,
                    'threshold': th,
                    'cliff_ids': cliff_ids,
                    'smooth_ids': smooth_ids,
                    'properties': props
                }
                
                print(f'    Cliff clusters: {len(cliff_ids)} (first cluster size: {len(cliff_ids[0]) if cliff_ids else 0})')
                print(f'    Smooth clusters: {len(smooth_ids)} (first cluster size: {len(smooth_ids[0]) if smooth_ids else 0})')
                
                # Calculate statistics for top clusters
                for i, (c_id, s_id) in enumerate(zip(cliff_ids[:self.top], smooth_ids[:self.top])):
                    cliff_nmols = len(c_id)
                    smooth_nmols = len(s_id)
                    cliff_p = props[c_id]
                    smooth_p = props[s_id]
                    cliff_p_mean = np.mean(cliff_p)
                    cliff_p_std = np.std(cliff_p)
                    smooth_p_mean = np.mean(smooth_p)
                    smooth_p_std = np.std(smooth_p)
                    
                    self.results.append({
                        'fingerprint_type': fp_type,
                        'threshold': th,
                        'cluster_index': i,
                        'cliff_nmols': cliff_nmols,
                        'smooth_nmols': smooth_nmols,
                        'cliff_p_mean': cliff_p_mean,
                        'cliff_p_std': cliff_p_std,
                        'smooth_p_mean': smooth_p_mean,
                        'smooth_p_std': smooth_p_std
                    })
        self.df_results = pd.DataFrame(self.results)
        self.df_results.to_csv('clustering_results.csv', index=False)

    def get_available_fingerprint_configs(self):
        """Get list of available fingerprint type and threshold combinations"""
        return list(self.cluster_data.keys())
    
    def visualize_molecules(self, fingerprint_type, threshold, cluster_index, smiles_file, 
                          max_molecules=20, save_images=True, show_images=True):
        """
        Visualize molecular structures for a specific fingerprint type, threshold, and cluster
        
        Parameters:
        fingerprint_type (str): Type of fingerprint ('ECFP', 'MACCS', 'RDKIT')
        threshold (float): Clustering threshold used
        cluster_index (int): Index of the cluster to visualize
        smiles_file (str): Path to CSV file containing SMILES and properties
        max_molecules (int): Maximum number of molecules to display per cluster
        save_images (bool): Whether to save images to files
        show_images (bool): Whether to display images
        """
        
        key = f"{fingerprint_type}_{threshold}"
        
        if key not in self.cluster_data:
            print(f"Error: No clustering data found for {fingerprint_type} with threshold {threshold}")
            print(f"Available configurations: {self.get_available_fingerprint_configs()}")
            return
        
        cluster_info = self.cluster_data[key]
        cliff_ids = cluster_info['cliff_ids']
        smooth_ids = cluster_info['smooth_ids']
        
        try:
            # Load the SMILES data
            df_smiles = pd.read_csv(smiles_file)
            print(f"\n{'='*80}")
            print(f"VISUALIZING: {fingerprint_type} Fingerprint, Threshold: {threshold}, Cluster: {cluster_index}")
            print(f"{'='*80}")
            print(f"Loaded {len(df_smiles)} molecules from {smiles_file}")
            print(f"Columns available: {list(df_smiles.columns)}")
            
            # Check if cluster_index is valid
            if cluster_index >= len(cliff_ids) or cluster_index >= len(smooth_ids):
                print(f"Error: cluster_index {cluster_index} is out of range.")
                print(f"Available cliff clusters: {len(cliff_ids)}")
                print(f"Available smooth clusters: {len(smooth_ids)}")
                return
            
            # Get the specific cluster indices
            cliff_cluster_indices = cliff_ids[cluster_index]
            smooth_cluster_indices = smooth_ids[cluster_index]
            
            print(f"\nCluster {cluster_index} Information:")
            print(f'  Cliff cluster size: {len(cliff_cluster_indices)}')
            print(f'  Smooth cluster size: {len(smooth_cluster_indices)}')
            
            # Limit the number of molecules to display
            cliff_indices_display = cliff_cluster_indices[:max_molecules]
            smooth_indices_display = smooth_cluster_indices[:max_molecules]
            
            # Process cliff cluster
            cliff_results = self._process_cluster_molecules(
                df_smiles, cliff_indices_display, "Cliff", fingerprint_type, threshold, cluster_index
            )
            
            # Process smooth cluster
            smooth_results = self._process_cluster_molecules(
                df_smiles, smooth_indices_display, "Smooth", fingerprint_type, threshold, cluster_index
            )
            
            # Create visualizations
            if cliff_results['molecules'] and show_images:
                self._create_cluster_visualization(
                    cliff_results, "Cliff", fingerprint_type, threshold, cluster_index, save_images
                )
            
            if smooth_results['molecules'] and show_images:
                self._create_cluster_visualization(
                    smooth_results, "Smooth", fingerprint_type, threshold, cluster_index, save_images
                )
            
            # Print comparison statistics
            self._print_cluster_comparison(cliff_results, smooth_results, fingerprint_type, threshold, cluster_index)
            
            return {
                'cliff_results': cliff_results,
                'smooth_results': smooth_results,
                'fingerprint_type': fingerprint_type,
                'threshold': threshold,
                'cluster_index': cluster_index
            }
            
        except Exception as e:
            print(f"Error in visualize_molecules: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_cluster_molecules(self, df_smiles, indices, cluster_type, fp_type, threshold, cluster_idx):
        """Process molecules for a cluster and return valid molecules and properties"""
        
        results = {
            'molecules': [],
            'properties': [],
            'smiles': [],
            'indices': indices,
            'valid_count': 0,
            'invalid_count': 0
        }
        
        try:
            # Extract SMILES and properties
            cluster_smiles = df_smiles.iloc[indices]['smiles'].tolist()
            cluster_props = df_smiles.iloc[indices]['exp_mean [nM]'].tolist()
            
            # Convert SMILES to RDKit molecules
            for i, (smile, prop) in enumerate(zip(cluster_smiles, cluster_props)):
                mol = Chem.MolFromSmiles(smile)
                if mol is not None:
                    results['molecules'].append(mol)
                    results['properties'].append(prop)
                    results['smiles'].append(smile)
                    results['valid_count'] += 1
                else:
                    results['invalid_count'] += 1
                    print(f"Warning: Invalid SMILES in {cluster_type} cluster: {smile}")
            
        except KeyError as e:
            print(f"Error: Column not found - {e}")
            print("Please check column names in your CSV file")
        except IndexError as e:
            print(f"Error: Index out of range - {e}")
            print("Some molecule indices may be invalid")
        
        return results
    


    def _create_cluster_visualization(self, cluster_results, cluster_type, fp_type, threshold, cluster_idx, save_image):
        """Create and display cluster visualization (SVG-compatible)"""
        
        molecules = cluster_results['molecules']
        properties = cluster_results['properties']
        
        if not molecules:
            print(f"No valid molecules to visualize for {cluster_type} cluster")
            return
        
        print(f"\nCreating {cluster_type.lower()} cluster visualization (SVG)...")
        legends = [f'{p:.2f} nM' for p in properties]
        
        try:
            # Generate SVG grid image
            svg_result = Draw.MolsToGridImage(
                molecules,
                molsPerRow=4,
                subImgSize=(400, 400),
                legends=legends,
                useSVG=True   # <-- IMPORTANT

            )

            # FIX: Handle the SVG object properly
            if hasattr(svg_result, 'data'):
                svg_str = svg_result.data  # Extract string from SVG object
            else:
                svg_str = str(svg_result)  # Convert to string
            
            # Now you can use svg_str safely
            display(SVG(svg_str))
            
        
            
            # Show cluster info separately with matplotlib
            plt.figure(figsize=(16, 1))
            plt.text(
                0.5, 0.5,
                f'{cluster_type} Cluster - {fp_type} Fingerprint\n'
                f'Threshold: {threshold}, Cluster: {cluster_idx}\n'
                f'({len(molecules)} molecules, '
                f'Activity range: {min(properties):.2f} - {max(properties):.2f} nM)',
                fontsize=12,
                ha="center",
                va="center",
                weight="bold"
            )
            plt.axis("off")
            plt.show()
            
            # Save to file if requested
            if save_image:
                filename = f'molecules/{cluster_type.lower()}_cluster_{fp_type}_{threshold}_{cluster_idx}.svg'
                with open(filename, "w") as f:
                    f.write(svg_str)
                print(f"Saved SVG image: {filename}")
                
        except Exception as e:
            print(f"Error creating visualization: {e}")

    
    def _print_cluster_comparison(self, cliff_results, smooth_results, fp_type, threshold, cluster_idx):
        """Print comparison statistics between cliff and smooth clusters"""
        
        print(f"\n{'='*60}")
        print(f"CLUSTER COMPARISON - {fp_type} (Threshold: {threshold}, Cluster: {cluster_idx})")
        print(f"{'='*60}")
        
        if cliff_results['properties']:
            cliff_props = cliff_results['properties']
            print(f"\nCliff Cluster Statistics:")
            print(f"  Valid molecules: {cliff_results['valid_count']}")
            print(f"  Invalid SMILES: {cliff_results['invalid_count']}")
            print(f"  Activity range: {min(cliff_props):.2f} - {max(cliff_props):.2f} nM")
            print(f"  Activity mean: {np.mean(cliff_props):.2f} ± {np.std(cliff_props):.2f} nM")
        
        if smooth_results['properties']:
            smooth_props = smooth_results['properties']
            print(f"\nSmooth Cluster Statistics:")
            print(f"  Valid molecules: {smooth_results['valid_count']}")
            print(f"  Invalid SMILES: {smooth_results['invalid_count']}")
            print(f"  Activity range: {min(smooth_props):.2f} - {max(smooth_props):.2f} nM")
            print(f"  Activity mean: {np.mean(smooth_props):.2f} ± {np.std(smooth_props):.2f} nM")
        
        # Compare activity cliff characteristics
        if cliff_results['properties'] and smooth_results['properties']:
            cliff_std = np.std(cliff_results['properties'])
            smooth_std = np.std(smooth_results['properties'])
            print(f"\nActivity Cliff Analysis:")
            print(f"  Cliff cluster std: {cliff_std:.2f} nM")
            print(f"  Smooth cluster std: {smooth_std:.2f} nM")
            print(f"  Cliff/Smooth std ratio: {cliff_std/smooth_std:.2f}")
    
    def compare_fingerprint_types(self, threshold, cluster_index, smiles_file, max_molecules=16):
        """Compare the same cluster across different fingerprint types"""
        
        print(f"\n{'='*80}")
        print(f"FINGERPRINT COMPARISON - Threshold: {threshold}, Cluster: {cluster_index}")
        print(f"{'='*80}")
        
        comparison_results = {}
        
        for fp_type in self.fingerprint_types:
            key = f"{fp_type}_{threshold}"
            if key in self.cluster_data:
                print(f"\nProcessing {fp_type}...")
                result = self.visualize_molecules(
                    fp_type, threshold, cluster_index, smiles_file, 
                    max_molecules=max_molecules, show_images=True
                )
                comparison_results[fp_type] = result
        
        # Print summary comparison
        print(f"\n{'='*80}")
        print(f"FINGERPRINT COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        for fp_type, result in comparison_results.items():
            if result:
                cliff_props = result['cliff_results']['properties']
                smooth_props = result['smooth_results']['properties']
                
                print(f"\n{fp_type} Fingerprint:")
                if cliff_props:
                    print(f"  Cliff: {len(cliff_props)} molecules, std: {np.std(cliff_props):.2f} nM")
                if smooth_props:
                    print(f"  Smooth: {len(smooth_props)} molecules, std: {np.std(smooth_props):.2f} nM")
        
        return comparison_results
    
    def save_results_to_csv(self, filename='fingerprint_cluster_analysis.csv'): 
        """Save all clustering results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
            return df
        else:
            print("No results to save. Run perform_clustering() first.")
            return None

# Usage Examples
def main():
    """Main function demonstrating usage"""
    
    # Initialize the analyzer
    analyzer = FingerprintClusterAnalyzer(
        fingerprint_types=['ECFP', 'MACCS', 'RDKIT'],
        thresholds=np.linspace(0.3,0.9,7), # Might need to adjust based on available data, use only a single threshold for now
        top=10
    )
    
    # Load data and perform clustering
    print("Loading fingerprint data...")
    analyzer.load_fingerprint_data() # Loads hardcoded files, please change accordingly. Add the fps and props prefix as shown in the function definition.
    
    print("\nPerforming clustering...")
    analyzer.perform_clustering() # Perform clustering for all fingerprint types and thresholds
    
    # Save clustering results
    analyzer.save_results_to_csv() # Save results to CSV
    
    # Visualize specific clusters
    smiles_file = 'data/CHEMBL234_Ki.csv' # Example usage, ensure that SMILES and npy files are consistent
    
    # Example 1: Visualize ECFP cluster
    print("\n" + "="*80)
    print("EXAMPLE 1: ECFP Fingerprint Analysis")
    print("="*80)
    analyzer.visualize_molecules('ECFP', 0.9, 0, smiles_file, max_molecules=20) # Visualize first cluster of ECFP at 0.9 threshold, please change accordingly 
    
    # Example 2: Compare all fingerprint types for the same cluster
    print("\n" + "="*80)
    print("EXAMPLE 2: Fingerprint Type Comparison")
    print("="*80)
    analyzer.compare_fingerprint_types(0.9, 0, smiles_file, max_molecules=20) # Compare first cluster across all fingerprint types at 0.9 threshold, please change accordingly

    # Example 3: Analyze multiple clusters for one fingerprint type
    print("\n" + "="*80)
    print("EXAMPLE 3: Multiple Cluster Analysis (MACCS)")
    print("="*80)
    for cluster_idx in range(min(3, len(analyzer.cluster_data.get('MACCS_0.8', {}).get('cliff_ids', [])))):
        analyzer.visualize_molecules('MACCS', 0.8, cluster_idx, smiles_file,
                                   max_molecules=8, show_images=False)

# Run the analysis
if __name__ == "__main__":
    main()

# Alternative simple usage:
"""
# Quick usage example:
analyzer = FingerprintClusterAnalyzer()
analyzer.load_fingerprint_data()
analyzer.perform_clustering()

# Visualize ECFP results
smiles_file = 'data/CHEMBL234_Ki.csv'
analyzer.visualize_molecules('ECFP', 0.9, 0, smiles_file)

# Compare all fingerprint types
analyzer.compare_fingerprint_types(0.9, 0, smiles_file)

"""
