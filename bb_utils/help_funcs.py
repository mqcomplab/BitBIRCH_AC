import numpy as np

def pair_sim(mol1, mol2):
    return np.dot(mol1,mol2)/(np.dot(mol1,mol1)+np.dot(mol2,mol2)-np.dot(mol1,mol2))

def count_pairs(fps, props, threshold, k):
    mol_inds = []
    comps = []
    for i1, m1 in enumerate(fps):
        for i2, m2 in enumerate(fps):
            if i1 == i2:
                pass
            else:
                if pair_sim(m1, m2) >= threshold and abs(props[i1] - props[i2]) >= 1:
                    #print(k[i1], k[i2])
                    comps.append(1)
                    if k[i1] not in mol_inds:
                        mol_inds.append(k[i1])
                    if k[i2] not in mol_inds:
                        mol_inds.append(k[i2])
                    
    comps = np.array(comps)
    return int(np.sum(comps)/2), mol_inds