# BitBIRCH is an open-source clustering module based on iSIM
#
# Please, cite the BitBIRCH paper: https://www.biorxiv.org/content/10.1101/2024.08.10.607459v1
#
# BitBIRCH is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# BitBIRCH is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# BitBIRCH authors (PYTHON): Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
#                            Vicky (Vic) Jung <jungvicky@ufl.edu>
#                            Kenneth Lopez Perez <klopezperez@chem.ufl.edu>
#
# BitBIRCH License: LGPL-3.0 https://www.gnu.org/licenses/lgpl-3.0.en.html#license-text
#
### Part of the tree-management code was derived from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
### Authors: Manoj Kumar <manojkumarsivaraj334@gmail.com>
###          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
###          Joel Nothman <joel.nothman@gmail.com>
### License: BSD 3 clause

import numpy as np
from scipy import sparse

def pair_sim(mol1, mol2):
    return np.dot(mol1,mol2)/(np.dot(mol1,mol1)+np.dot(mol2,mol2)-np.dot(mol1,mol2))

def jt_isim(c_total, n_objects):
    """iSIM Tanimoto calculation
    
    https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00041b
    
    Parameters
    ----------
    c_total : np.ndarray
              Sum of the elements column-wise
              
    n_objects : int
                Number of elements
                
    Returns
    ----------
    isim : float
           iSIM Jaccard-Tanimoto value
    """
    sum_kq = np.sum(c_total)
    sum_kqsq = np.dot(c_total, c_total)
    a = (sum_kqsq - sum_kq)/2

    return a/(a + n_objects * sum_kq - sum_kqsq)

def max_separation(X):
    """Finds two objects in X that are very separated
    This is an approximation (not guaranteed to find
    the two absolutely most separated objects), but it is
    a very robust O(N) implementation. Quality of clustering
    does not diminish in the end.
    
    Algorithm:
    a) Find centroid of X
    b) mol1 is the molecule most distant from the centroid
    c) mol2 is the molecule most distant from mol1
    
    Returns
    -------
    (mol1, mol2) : (int, int)
                   indices of mol1 and mol2
    1 - sims_mol1 : np.ndarray
                   Distances to mol1
    1 - sims_mol2: np.ndarray
                   Distances to mol2
    These are needed for node1_dist and node2_dist in _split_node
    """
    # Get the centroid of the set
    cent_matrix = []
    for cent1 in X:
        cent_matrix.append([])
        for cent2 in X:
            cent_matrix[-1].append(pair_sim(cent1, cent2))
            #isim_matrix[-1].append(jt_cent(cent1, cent2))
    cent_matrix = np.array(cent_matrix)
    #np.fill_diagonal(isim_matrix, 0)
    (mol1, mol2) = np.unravel_index(np.argmin(cent_matrix), cent_matrix.shape)
    sims_mol1 = cent_matrix[mol1]
    #sims_mol1[mol1] = 3.08
    sims_mol2 = cent_matrix[mol2]
    #n_samples = len(X)
    #linear_sum = np.sum(X, axis = 0)
    #centroid = calc_centroid(linear_sum, n_samples)
    #
    ## Get the similarity of each molecule to the centroid
    #pop_counts = np.sum(X, axis = 1)
    #a_centroid = np.dot(X, centroid)
    #sims_med = a_centroid / (pop_counts + np.sum(centroid) - a_centroid)
    #
    ## Get the least similar molecule to the centroid
    #mol1 = np.argmin(sims_med)
    #
    ## Get the similarity of each molecule to mol1
    #a_mol1 = np.dot(X, X[mol1])
    #sims_mol1 = a_mol1 / (pop_counts + pop_counts[mol1] - a_mol1)
    #
    ## Get the least similar molecule to mol1
    #mol2 = np.argmin(sims_mol1)
    #
    ## Get the similarity of each molecule to mol2
    #a_mol2 = np.dot(X, X[mol2])
    #sims_mol2 = a_mol2 / (pop_counts + pop_counts[mol2] - a_mol2)
    
    return (mol1, mol2), sims_mol1, sims_mol2

def calc_centroid(linear_sum, n_samples):
    """Calculates centroid
    
    Parameters
    ----------
    
    linear_sum : np.ndarray
                 Sum of the elements column-wise
    n_samples : int
                Number of samples
                
    Returns
    -------
    centroid : np.ndarray
               Centroid fingerprints of the given set
    """
    return linear_sum/n_samples

def _iterate_sparse_X(X):
    """This little hack returns a densified row when iterating over a sparse
    matrix, instead of constructing a sparse matrix for every row that is
    expensive.
    """
    n_samples, n_features = X.shape
    X_indices = X.indices
    X_data = X.data
    X_indptr = X.indptr

    for i in range(n_samples):
        row = np.zeros(n_features)
        startptr, endptr = X_indptr[i], X_indptr[i + 1]
        nonzero_indices = X_indices[startptr:endptr]
        row[nonzero_indices] = X_data[startptr:endptr]
        yield row

def _split_node(node, threshold, branching_factor):
    """The node has to be split if there is no place for a new subcluster
    in the node.
    1. Two empty nodes and two empty subclusters are initialized.
    2. The pair of distant subclusters are found.
    3. The properties of the empty subclusters and nodes are updated
       according to the nearest distance between the subclusters to the
       pair of distant subclusters.
    4. The two nodes are set as children to the two subclusters.
    """
    new_subcluster1 = _BFSubcluster()
    new_subcluster2 = _BFSubcluster()
    new_node1 = _BFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_node2 = _BFNode(
        threshold=threshold,
        branching_factor=branching_factor,
        is_leaf=node.is_leaf,
        n_features=node.n_features,
        dtype=node.init_centroids_.dtype,
    )
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2  
    
    # O(N) implementation of max separation
    farthest_idx, node1_dist, node2_dist = max_separation(node.centroids_)    
    # Notice that max_separation is returning similarities and not distances
    node1_closer = node1_dist > node2_dist
    # Make sure node1 is closest to itself even if all distances are equal.
    # This can only happen when all node.centroids_ are duplicates leading to all
    # distances between centroids being zero.
    node1_closer[farthest_idx[0]] = True

    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
    return new_subcluster1, new_subcluster2


class _BFNode:
    """Each node in a BFTree is called a BFNode.

    The BFNode can have a maximum of branching_factor
    number of BFSubclusters.

    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a BFSubcluster.

    branching_factor : int
        Maximum number of BF subclusters in each node.

    is_leaf : bool
        We need to know if the BFNode is a leaf or not, in order to
        retrieve the final subclusters.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : list
        List of subclusters for a particular BFNode.

    prev_leaf_ : _BFNode
        Useful only if is_leaf is True.

    next_leaf_ : _BFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.

    init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
        Manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    centroids_ : ndarray of shape (branching_factor + 1, n_features)
        View of ``init_centroids_``.

    """

    def __init__(self, *, threshold, branching_factor, is_leaf, n_features, dtype):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features

        # The list of subclusters, centroids and squared norms
        # to manipulate throughout.
        self.subclusters_ = []
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features), dtype=dtype)
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        
        # Keep centroids as views. In this way
        # if we change init_centroids, it is sufficient
        self.centroids_ = self.init_centroids_[: n_samples + 1, :]
        
    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        """Remove a subcluster from a node and update it with the
        split subclusters.
        """
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.centroids_[ind] = new_subcluster1.centroid_
        self.append_subcluster(new_subcluster2)

    def insert_bf_subcluster(self, subcluster, set_bits):
        """Insert a new subcluster into the node."""
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        threshold = self.threshold
        branching_factor = self.branching_factor
        # We need to find the closest subcluster among all the
        # subclusters so that we can insert our new subcluster.
        a = np.dot(self.centroids_, subcluster.centroid_)
        sim_matrix = a / (np.sum(self.centroids_, axis = 1) + set_bits - a)
        closest_index = np.argmax(sim_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # If the subcluster has a child, we need a recursive strategy.
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_bf_subcluster(subcluster, set_bits)

            if not split_child:
                # If it is determined that the child need not be split, we
                # can just update the closest_subcluster
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                self.centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                return False

            # things not too good. we need to redistribute the subclusters in
            # our child node, and add a new subcluster in the parent
            # subcluster to accommodate the new child.
            else:
                new_subcluster1, new_subcluster2 = _split_node(
                    closest_subcluster.child_,
                    threshold,
                    branching_factor
                )
                self.update_split_subclusters(
                    closest_subcluster, new_subcluster1, new_subcluster2
                )

                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # good to go!
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            if merged:
                self.centroids_[closest_index] = closest_subcluster.centroid_
                self.init_centroids_[closest_index] = closest_subcluster.centroid_
                return False

            # not close to any other subclusters, and we still
            # have space, so add.
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False

            # We do not have enough space nor is it closer to an
            # other subcluster. We need to split.
            else:
                self.append_subcluster(subcluster)
                return True


class _BFSubcluster:
    """Each subcluster in a BFNode is called a BFSubcluster.

    A BFSubcluster can have a BFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``BFNode.centroids_`` is called.
    
    mol_indices : list, default=[]
        List of indices of molecules included in the given cluster.

    child_ : _BFNode
        Child Node of the subcluster. Once a given _BFNode is set as the child
        of the _BFNode, it is set to ``self.child_``.
    """

    def __init__(self, *, linear_sum = None, mol_indices = [], p_min = 0, p_max = 0):
        if linear_sum is None:
            self.n_samples_ = 0
            self.centroid_ = self.linear_sum_ = 0
            self.mol_indices = []
            self.p_min = 0
            self.p_max = 0
        else:
            self.n_samples_ = 1
            self.centroid_ = self.linear_sum_ = linear_sum
            self.mol_indices = mol_indices
            self.p_min = p_min
            self.p_max = p_max
        
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.mol_indices += subcluster.mol_indices
        self.centroid_ = calc_centroid(self.linear_sum_, self.n_samples_)
        self.p_min = min(self.p_min, subcluster.p_min)
        self.p_max = max(self.p_max, subcluster.p_max)

    def merge_subcluster(self, nominee_cluster, threshold):
        """Check if a cluster is worthy enough to be merged. If
        yes then merge.
        """
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        new_centroid = calc_centroid(new_ls, new_n)
        new_max = max(self.p_max, nominee_cluster.p_max)
        new_min = min(self.p_min, nominee_cluster.p_min)
        
        #corr = np.where(self.linear_sum_ >= self.n_samples_ * 0.5 , 1, 0)
        #jt_sim = (jt_isim(new_ls + corr, new_n + 1) * (new_n + 1) - jt_isim(new_ls, new_n) * (new_n - 1))/2
        jt_sim = jt_isim(new_ls, new_n)
        if jt_sim >= threshold:
            (
                self.n_samples_,
                self.linear_sum_,
                self.centroid_,
                self.mol_indices
            ) = (new_n, new_ls, new_centroid, self.mol_indices + nominee_cluster.mol_indices)
            self.p_max = new_max
            self.p_min = new_min
            #print(jt_sim)
            return True
        return False


class BitBirch():
    """Implements the BitBIRCH clustering algorithm.
    
    BitBIRCH paper: 

    Memory- and time-efficient, online-learning algorithm.
    It constructs a tree data structure with the cluster centroids being read off the leaf.
    
    Parameters
    ----------
    threshold : float, default=0.5
        The similarity radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be greater than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default=50
        Maximum number of BF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    Attributes
    ----------
    root_ : _BFNode
        Root of the BFTree.

    dummy_leaf_ : _BFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, mol_indices and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, mol_indices and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.
    """


    def __init__(
        self,
        *,
        threshold=0.5,
        branching_factor=50,
    ):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.index_tracker = 0
        self.first_call = True

    def fit(self, X, props):
        """
        Build a BF Tree for the input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self
            Fitted estimator.
        """

        # TODO: Add input verification

        return self._fit(X, props)

    def _fit(self, X, props):
        threshold = self.threshold
        branching_factor = self.branching_factor

        n_features = X.shape[1]
        d_type = X.dtype

        # If partial_fit is called for the first time or fit is called, we
        # start a new tree.
        if self.first_call:
            # The first root is the leaf. Manipulate this object throughout.
            self.root_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
    
            # To enable getting back subclusters.
            self.dummy_leaf_ = _BFNode(
                threshold=threshold,
                branching_factor=branching_factor,
                is_leaf=True,
                n_features=n_features,
                dtype=d_type,
            )
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_

        # Cannot vectorize. Enough to convince to use cython.
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X

        for sample, prop in zip(X, props):
            set_bits = np.sum(sample)
            subcluster = _BFSubcluster(linear_sum=sample, mol_indices = [self.index_tracker], p_min = prop, p_max = prop)
            split = self.root_.insert_bf_subcluster(subcluster, set_bits)

            if split:
                new_subcluster1, new_subcluster2 = _split_node(
                    self.root_, threshold, branching_factor
                )
                del self.root_
                self.root_ = _BFNode(
                    threshold=threshold,
                    branching_factor=branching_factor,
                    is_leaf=False,
                    n_features=n_features,
                    dtype=d_type,
                )
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)
            self.index_tracker += 1

        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids
        self._n_features_out = self.subcluster_centers_.shape[0]
        
        self.first_call = False
        return self

    def _get_leaves(self):
        """
        Retrieve the leaves of the BF Node.

        Returns
        -------
        leaves : list of shape (n_leaves,)
            List of the leaf nodes.
        """
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves
    
    def get_centroids(self):
        """Method to return a list of Numpy arrays containing the centroids' fingerprints"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        centroids = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                centroids.append(subcluster.centroid_)

        return centroids
    
    def get_cluster_mol_ids(self):
        """Method to return the indices of molecules in each cluster"""
        if self.first_call:
            raise ValueError('The model has not been fitted yet.')
        
        clusters_mol_id = []
        for leaf in self._get_leaves():
            for subcluster in leaf.subclusters_:
                clusters_mol_id.append(subcluster.mol_indices)

        return clusters_mol_id

#import time
#fps = np.load('fps.npy')
#props = np.load('props.npy')
#
#def pair_sim(mol1, mol2):
#    return np.dot(mol1,mol2)/(np.dot(mol1,mol1)+np.dot(mol2,mol2)-np.dot(mol1,mol2))
#
#def count_pairs(fps, props, ac_thre, k):
#    #print(k)
#    mol_inds = []
#    comps = []
#    for i1, m1 in enumerate(fps):
#        for i2, m2 in enumerate(fps):
#            if i1 == i2:
#                pass
#            else:
#                if pair_sim(m1, m2) >= ac_thre and abs(props[i1] - props[i2]) >= 1:
#                    #print(k[i1], k[i2])
#                    comps.append(1)
#                    if k[i1] not in mol_inds:
#                        mol_inds.append(k[i1])
#                    if k[i2] not in mol_inds:
#                        mol_inds.append(k[i2])
#                    
#    comps = np.array(comps)
#    return np.sum(comps)/2, mol_inds
#
#fps = np.load('fps.npy')
#
#def close_analysis(fps, props, b_thre=0.95, ac_thre=0.95):
#    #thre = 0.95
#    brc = BitBirch(branching_factor = 50, threshold = b_thre)
#    brc.fit(fps, props)
#    
#    inds = brc.get_cluster_mol_ids()
#    
#    tot = 0
#    total_indices = []
#    for k in inds:
#        if len(k) > 1:
#            #tot += len(k) * (len(k) - 1)/2
#            #print(k)
#            n_pairs, new_inds = count_pairs(fps[k], props[k], ac_thre, k)
#            tot += n_pairs
#            total_indices += new_inds
#    return tot, total_indices
#
#b_thre = 0.9
#ac_thre = 0.9
#total_pairs = 0
#molecule_indices = []
#
##order_inds = list(range(len(fps)))
##new_order = np.random.permutation(order_inds)
#
#row_sums = np.sum(fps, axis=1)
#new_order = np.argsort(row_sums)#[::-1]
#
#tot = 1
#new_fps = fps[new_order]
#new_props = props[new_order]
#r = 1
#while tot:
#    print(r)
#    tot, total_indices = close_analysis(new_fps, new_props, b_thre=b_thre, ac_thre=ac_thre)
#    print(tot)
#    total_pairs += tot
#    molecule_indices += total_indices
#    
#    new_fps = np.delete(new_fps, total_indices, axis=0)
#    new_props = np.delete(new_props, total_indices, axis=0)
#    
#    r += 1
#print(total_pairs)
##print(sorted(molecule_indices))
#
#
## PAIR COMPARISON
#
#tani_matrix = np.load('tani_matrix.npy')
#prop_matrix = np.load('prop_matrix.npy')
#
#thre = ac_thre
#tani_matrix[tani_matrix >= thre] = 1
#tani_matrix[tani_matrix < thre] = 0
#
##print(np.sum(tani_matrix)/2)
##print(np.sum(prop_matrix)/2)
#
#cliff_matrix = prop_matrix * tani_matrix
#
#indices = np.nonzero(cliff_matrix > 0)
##indices = np.nonzero(tani_matrix > 0)
#coordinates = list(zip(indices[0], indices[1]))
##unique_coordinates = []
##for p in coordinates:
##    if p[0] < p[1]:
##        unique_coordinates.append(p)
#print(len(coordinates)/2)
##print(coordinates)
#unique_coordinates = []
#for pair in coordinates:
#    if pair[0] not in unique_coordinates:
#        unique_coordinates.append(pair[0])
#    if pair[1] not in unique_coordinates:
#        unique_coordinates.append(pair[1])
#
##print(sorted(unique_coordinates))
#
#common = set(unique_coordinates) & set(molecule_indices)
##print(len(common)/len(unique_coordinates))
#
## OLD
##def count_exact_acs(tani_matrix, prop_matrix, thre):
##    tani_matrix[tani_matrix >= thre] = 1
##    tani_matrix[tani_matrix < thre] = 0
##    
##    #print(np.sum(tani_matrix)/2)
##    #print(np.sum(prop_matrix)/2)
##    
##    cliff_matrix = prop_matrix * tani_matrix
##    
##    indices = np.nonzero(cliff_matrix > 0)
##    #indices = np.nonzero(tani_matrix > 0)
##    coordinates = list(zip(indices[0], indices[1]))
##    unique_coordinates = []
##    for p in coordinates:
##        if p[0] < p[1]:
##            unique_coordinates.append(p)
##    return unique_coordinates
##
##def count_acs(fps, props, thre, k):
##    #print(k)
##    sim_matrix=[]
##    prop_matrix = []
##    for i in range(len(props)):
##        #print(i)
##        prop_matrix.append([])
##        sim_matrix.append([])
##        for j in range(len(props)):
##            if abs(props[i] - props[j]) >= 1:
##                prop_matrix[-1].append(1)
##            else:
##                prop_matrix[-1].append(0)
##            #prop_matrix[-1].append((props[i] - props[j])**2)
##            if i == j:
##                sim_matrix[-1].append(0)
##            else:
##                sim_matrix[-1].append(pair_sim(fps[i], fps[j]))
##    
##    sim_matrix = np.array(sim_matrix)
##    
##    sim_matrix[sim_matrix >= thre] = 1
##    sim_matrix[sim_matrix < thre] = 0
##    
##    cliff_matrix = prop_matrix * sim_matrix
##    
##    indices = np.nonzero(cliff_matrix > 0)
##    coordinates = list(zip(indices[0], indices[1]))
##    unique_coordinates = []
##    for p in coordinates:
##        if k[p[0]] < k[p[1]]:
##            unique_coordinates.append((k[p[0]], k[p[1]]))
##    return unique_coordinates
##
###r_fps = np.repeat(fps, reps, axis=0)
###r_props = np.repeat(props, reps, axis=0)
##summary = []
##reps = 1
##for _ in range(reps):
##    
##    ##identity (original order)
##    new_order = list(range(len(fps)))
##    
##    ##random order
##    #order_inds = list(range(len(fps)))
##    #new_order = np.random.permutation(order_inds)
##    
##    ##order by nbits in fps
##    #row_sums = np.sum(fps, axis=1)
##    #new_order = np.argsort(row_sums)#[::-1]
##    
##    ##fancy bit ordering linear increase
##    #template = np.arange(1, len(fps[0]) + 1)
##    #dots = np.dot(fps, template)
##    #new_order = np.argsort(dots)[::-1]
##    
##    ##fancy bit ordering symmetric array
##    #def symmetric_array(n):
##    #    mid = (n + 1) // 2  # Find the midpoint
##    #    first_half = np.arange(1, mid + 1)  # Increasing part
##    #    second_half = first_half[::-1] if n % 2 == 0 else first_half[-2::-1]  # Decreasing part
##    #    return np.concatenate([first_half, second_half])
##    #template = symmetric_array(len(fps[0]))
##    #dots = np.dot(fps, template)
##    #new_order = np.argsort(dots)[::-1]
##    
##    ##fancy bit ordering random
##    #template = np.random.permutation(np.arange(0, len(fps[0] + 1)))
##    #dots = np.dot(fps, template)
##    #new_order = np.argsort(dots)[::-1]
##    
##    fps = fps[new_order]
##    props = props[new_order]
##    
##    c_thre = 0.9
##    brc = BitBirch(branching_factor = 50, threshold = c_thre)
##    v = time.time()
##    brc.fit(fps, props)
##    #print(time.time() - v)
##    maxi = 0
##    inds = brc.get_cluster_mol_ids()
##    c = 0
##    total = 0
##    unique = 0
##    ac_inds = []
##    for k in inds:
##        if len(k) > 1:
##            #print(len(k))
##            old_order = []
##            for new_ind in k:
##                old_order.append(new_order[new_ind])
##            #print(old_order)
##            #print('AAAAAAAAAAAAAAAAAAAAAAA')
##            p_vals = props[k]
##            diffs = []
##            for i in range(len(p_vals)):
##                pi = p_vals[i]
##                for j in range(len(p_vals)):
##                    pj = p_vals[j]
##                    diffs.append(abs(pi - pj))
##            diffs = np.array(diffs)
##            #print(diffs)
##            if len(k) > maxi:
##                maxi = len(k)
##            # counting total number of acs
##            ac_thre = 0.95
##            birch_acs = count_acs(fps[k], props[k], ac_thre, old_order)
##            ac_inds += birch_acs
##            total += len(birch_acs)
##            # 'unique' acs
##            unique += 1
##    summary.append(total)
##    #print(total)
##exact_acs = count_exact_acs(tani_matrix, prop_matrix, ac_thre)
##common = set(exact_acs) & set(ac_inds)
###print(exact_acs)
###print(ac_inds)
##print(len(common), len(common)/len(exact_acs))
##missing = []
##for p in exact_acs:
##    if p not in ac_inds:
##        missing.append(p)
##print(missing)
#
##summary = np.array(summary)
##print(np.max(summary))
##print(np.mean(summary))
##print(unique, total)
##print(maxi)