import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load both CSV files

def plot_overall_comparison(df1, df2, offset1, offset2): #To plot overall comparison across different/same offsets for different/same dataset. 

    # Remove the rows where ratio is -1
    df1 = df1[df1['ratio'] != -1]
    df2 = df2[df2['ratio'] != -1]
    df1 = df1[(df1['offset']==offset1)]
    df2 = df2[(df2['offset']==offset2)]

    fingerprints = ['ECFP', 'MACCS', 'RDKIT']

    means1 = []
    stds1 = []
    means2 = []
    stds2 = []

    for fp in fingerprints:

        # Dataset 1
        ratios1 = df1[df1['suffix'].str.endswith(fp)]['ratio']
        means1.append(ratios1.mean())
        stds1.append(ratios1.std())

        # Dataset 2
        ratios2 = df2[df2['suffix'].str.endswith(fp)]['ratio']
        means2.append(ratios2.mean())
        stds2.append(ratios2.std())

    plt.figure(figsize=(10, 6))

    x = np.arange(len(fingerprints))
    width = 0.35
    
    df1_recursive = df1['recursive'].unique()
    df2_recursive = df2['recursive'].unique()

    if len(df1_recursive) != 1 or len(df2_recursive) != 1:
        raise ValueError("DataFrames must have a single unique value for 'recursive' column.")
    df1_recursive = df1_recursive[0]
    df2_recursive = df2_recursive[0]
    if df1_recursive==False:
        label1='Non-Recursive'
    else:
        label1='Recursive'

    if df2_recursive==False:
        label2='Non-Recursive'
    else:
        label2='Recursive'

    bars1 = plt.bar(x - width/2, means1, width, yerr=stds1, capsize=5, 
                    color=['tab:blue', 'tab:orange', 'tab:green'], 
                alpha=0.8, label=f'{label1}, Offset={offset1}') # TODO: Remove hardcoding

    bars2 = plt.bar(x + width/2, means2, width, yerr=stds2, capsize=5, 
                    color=['lightblue', 'peachpuff', 'lightgreen'], 
                alpha=0.8, label=f'{label2}, Offset={offset2}') # TODO: Remove hardcoding

    plt.xlabel('Fingerprint', fontsize=12)
    plt.ylabel('Mean Ratio', fontsize=12)
    plt.title('Mean Ratio by Fingerprint Type', fontsize=14)
    plt.xticks(x, fingerprints)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/overall_comparison_{label1}_{offset1}_{label2}_{offset2}.png')
    plt.show()
    plt.close()

def plot_by_threshold(df, offset, output_file): # To check the efficiency of AC retrieval for a dataset across different thresholds.

    df = df[df['ratio'] != -1]
    df = df[(df['offset']==offset)]
    fingerprints = ['ECFP', 'MACCS', 'RDKIT']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    plt.figure(figsize=(8,6))

    df = df.sort_values(by='threshold')

    for fp, color in zip(fingerprints, colors):

        means=[]
        stds=[]
        thresholds=[]

        for th in sorted(df['threshold'].unique()):
            df_th = df[df['threshold'] == th]
            ratios = df_th[df_th['suffix'].str.endswith(fp)]['ratio']
            if not ratios.empty:
                means.append(ratios.mean())
                stds.append(ratios.std())
                thresholds.append(th)

        plt.errorbar(thresholds, means, yerr=stds, fmt='-o', capsize=5, label=fp, color=color, linewidth=2, marker='s')
    
    plt.title(f'Ratio vs Threshold for each Fingerprint', fontsize=14)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.legend(title='Fingerprint')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    plt.close()


def plot_comparison_by_threshold(df1, df2, offset1, offset2): # To compare the AC retrieval performance across different/same dataset at different/same offset.

    df1 = df1[df1['ratio'] != -1]
    df2 = df2[df2['ratio'] != -1]
    df1 = df1[(df1['offset']==offset1)]
    df2 = df2[(df2['offset']==offset2)]
    df1_recursive = df1['recursive'].unique()
    df2_recursive = df2['recursive'].unique()

    if len(df1_recursive) != 1 or len(df2_recursive) != 1:
        raise ValueError("DataFrames must have a single unique value for 'recursive' column.")
    df1_recursive = df1_recursive[0]
    df2_recursive = df2_recursive[0]
    if df1_recursive==False:
        label1='Non-Recursive'
    else:
        label1='Recursive'

    if df2_recursive==False:
        label2='Non-Recursive'
    else:
        label2='Recursive'

    fingerprints = ['ECFP', 'MACCS', 'RDKIT']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    plt.figure(figsize=(10, 6))


    for fp, color in zip(fingerprints, colors):

        means1=[]
        stds1=[]
        means2=[]
        stds2=[]
        thresholds=[]

        for th in sorted(df1['Threshold'].unique()):
            df1_th = df1[df1['Threshold'] == th]
            ratios1 = df1_th[df1_th['suffix'].str.endswith(fp)]['ratio']
            if not ratios1.empty:
                means1.append(ratios1.mean())
                stds1.append(ratios1.std())

            df2_th = df2[df2['Threshold'] == th]
            ratios2 = df2_th[df2_th['suffix'].str.endswith(fp)]['ratio']
            if not ratios2.empty:
                means2.append(ratios2.mean())
                stds2.append(ratios2.std())
                thresholds.append(th)

        plt.errorbar(thresholds, means1, yerr=stds1, fmt='-o', capsize=5, label=f'{fp} {label1} Offset={offset1}', color=color, linewidth=2, marker='s')
        plt.errorbar(thresholds, means2, yerr=stds2, fmt='--o', capsize=5, label=f'{fp} {label2} Offset={offset2}', color=color, linewidth=2, marker='o')

    plt.title('Ratio vs Threshold Comparison', fontsize=14)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Ratio', fontsize=12)
    plt.legend(title='Fingerprint and Offset')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('plots/comparison_by_threshold_'+label1+'_'+str(offset1)+'_'+label2+'_'+str(offset2)+'.png')
    plt.show()
    plt.close()

def plot_num_molecule(df, fps=['ECFP', 'MACCS', 'RDKIT']): #Number of molecules in the top clusters
    

    colors=plt.cm.viridis

    for fp in fps:
        plt.figure(figsize=(10,6))
        df_fp = df[df['fingerprint_type']==fp]
        thresholds = sorted(df_fp['threshold'].unique())
        color_map = {th: colors(i / (len(thresholds)-1)) for i, th in enumerate(thresholds)}
        for th in thresholds:
            df_th = df_fp[df_fp['threshold'] == th]
            x = df_th['cluster_index']
            plt.plot(x, df_th['cliff_nmols'], '-o', color=color_map[th], label=f'Cliff th={th:.2f}' , alpha=0.8)
            plt.plot(x, df_th['smooth_nmols'], '--o', color=color_map[th], label=f'Smooth th={th:.2f}' , alpha=0.8)
        plt.title(f'{fp}: Cluster Size vs Cluster Number')
        plt.xlabel('Cluster Number')
        plt.ylabel('Number of Molecules')
        # Custom legend for only one threshold (to avoid duplicate labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=10)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(f'Plots/Smooth/cluster_size_vs_number_{fp}.png', dpi=300)
        plt.show()

def plot_prop_std(df, fps=['ECFP', 'MACCS', 'RDKIT']): # Property standard deviation in top clusters
        colors=plt.cm.viridis

        for fp in fps:
          plt.figure(figsize=(10, 6))
          df_fp = df[df['fingerprint_type'] == fp]
          thresholds = sorted(df_fp['threshold'].unique())
          color_map = {th: colors(i / (len(thresholds)-1)) for i, th in enumerate(thresholds)}
          for th in thresholds:
              df_th = df_fp[df_fp['threshold'] == th]
              x = df_th['cluster_index']
              # Cliff std (solid)
              plt.plot(
                  x, df_th['cliff_p_std'],
                  '-o', color=color_map[th], label=f'Cliff th={th:.2f}' , alpha=0.8
              )
              # Smooth std (dashed)
              plt.plot(
                  x, df_th['smooth_p_std'],
                  '--o', color=color_map[th], label=f'Smooth th={th:.2f}' , alpha=0.8
              )
          plt.title(f'{fp}: Property Std vs Cluster Number')
          plt.xlabel('Cluster Number')
          plt.ylabel('Property Std')
          plt.ylim(0, 2.5)
          # Custom legend for only one threshold (to avoid duplicate labels)
          handles, labels = plt.gca().get_legend_handles_labels()
          by_label = dict(zip(labels, handles))
          plt.legend(by_label.values(), by_label.keys(), fontsize=10)
          plt.grid(False)
          plt.tight_layout()
          plt.savefig(f'Plots/Smooth/cluster_property_std_vs_number_{fp}.png', dpi=300)
          plt.show()

