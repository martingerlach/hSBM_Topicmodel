"""
Computes the NMI between 2 clusterings (partitions) of nodes into different groups using contingency tables.

Author: Alex Tao and Charles Hyland
"""

import numpy as np
from scipy import sparse as sp
import numpy as np
from math import log

def check_clusterings(label_U, label_V):
	"""
	Check that the labels arrays are 1-dimensional and of same dimension.
	
	Parameters
	----------
	label_U : int array, shape = [n_samples]
		The labels for clustering U
	label_V : int array, shape = [n_samples]
		The labels for clustering V
	"""
	# Make sure they are arrays.
	label_U = np.asarray(label_U)
	label_V = np.asarray(label_V)
	# Check dimension sizes.
	if label_U.ndim != 1:
		raise ValueError(f"label_U must be 1D shape. Current shape is: {label_U.shape}")
	if label_V.ndim != 1:
		raise ValueError(f"label_V must be 1D shape. Current shape is f{label_V.shape}")
	if label_U.shape != label_V.shape:
		raise ValueError(f"label_U and label_V must be same size. Current sizes are {label_U.shape[0]} and {label_V.shape[0]}")
	# Pass all test cases!
	return label_U, label_V


def create_contingency_matrix(labels_U, label_V):
	"""
	Build a contingency matrix describing the relationship between labels.
	Parameters
	----------
	labels_U : int array, shape = [number of nodes in network]
		Clustering labels using labeling U.
	label_V : array, shape = [number of nodes in network]
		Clustering labels using labeling V.
	Returns
	-------
	contingency : {array-like, sparse}, shape=[n_classes_U, n_classes_V]
		Matrix C such that entry C_{i, j} is the number of samples in common
		label U class i and in label V class j.
	"""
	# Retrieve different cluster classes for labelings and positions in 1D-array.
	clusters_U, clusters_U_idx = np.unique(labels_U, return_inverse=True)
	clusters_V, cluster_V_idx = np.unique(label_V, return_inverse=True)

	# Count how many unique clusters there are for both labelings.
	n_clusters_U = clusters_U.shape[0]
	n_clusters_V = clusters_V.shape[0]

	# Coo matrix is constructed by (data, row_idx, col_idx) representation.
	# Each data value has its corresponding row and col position located in 
	# the row_idx and col_idx vector. Helps with sparse matrix calculations.
	contingency_matrix = sp.coo_matrix((np.ones(clusters_U_idx.shape[0]),
								 (clusters_U_idx, cluster_V_idx)),
								shape=(n_clusters_U, n_clusters_V),
								dtype=np.int)
	return contingency_matrix


def calculate_average(U, V, average_method):
	"""
	Return a specified mean of two numbers. We generally use arithmetic.
	"""
	if average_method == "min":
		return min(U, V)
	elif average_method == "geometric":
		return np.sqrt(U * V)
	elif average_method == "arithmetic":
		return np.mean([U, V])
	elif average_method == "max":
		return max(U, V)
	else:
		raise ValueError("'average_method' must be 'min', 'geometric', "
						 "'arithmetic', or 'max'")


def calculate_entropy(labels):
	"""
	Calculates the entropy for a given labeling.
	Parameters
	----------
	labels : int array, shape = [number of nodes to partition]
		The labels of interest.
	"""
	if len(labels) == 0:
		# No entropy if no labels.
		return 1.0
	label_idx = np.unique(labels, return_inverse=True)[1]
	pi = np.bincount(label_idx).astype(np.float64)
	pi = pi[pi > 0]
	pi_sum = np.sum(pi)
	# log(a / b) should be calculated as log(a) - log(b) for possible loss of precision.
	return -np.sum((pi / pi_sum) * (np.log(pi) - log(pi_sum)))


def compute_mutual_info_score(contingency_matrix):
		"""
		Computes the Mutual Information between two clusterings.
		Parameters
		----------
		contingency_matrix : {None, array, sparse matrix}, shape = [n_classes_true, n_classes_pred]
				A contingency matrix given by the create_contingency_matrix function.
				If value is ``None``, it will be computed, otherwise the given value is
				used, with ``label_U`` and ``label_V`` ignored.
		Returns
		-------
		mi : float
			 Mutual information, a non-negative value
		"""
		if sp.issparse(contingency_matrix):
			# For a sparse matrix, return (row_idx, col_idx, values) of non-zero values.
			# Returns the COO list where NZ is non-zero.
			NZ_row_idx, NZ_col_idx, NZ_val = sp.find(contingency_matrix)
		else:
			raise ValueError("Unsupported type for given contingency matrix: %s" % type(contingency_matrix))

		contingency_sum = contingency_matrix.sum()
		# Get the sum of each row and column.
		pi = np.ravel(contingency_matrix.sum(axis=1))
		pj = np.ravel(contingency_matrix.sum(axis=0))

		# Compute terms needed for mutual information.
		log_contingency_nm = np.log(NZ_val)
		contingency_nm = NZ_val / contingency_sum

		# Don't need to calculate the full outer product, just for non-zeroes.
		outer = (pi.take(NZ_row_idx).astype(np.int64, copy=False) * pj.take(NZ_col_idx).astype(np.int64, copy=False))
		log_outer = -np.log(outer) + log(pi.sum()) + log(pj.sum())
		# Compute mutual information based on contigency matrix.
		mi = (contingency_nm * (log_contingency_nm - log(contingency_sum)) +
					contingency_nm * log_outer)
		return mi.sum()


def compute_normalised_mutual_information(labels_U, label_V, average_method='arithmetic'):
		"""
		Computes the Normalized Mutual Information (NMI) between two clustering of nodes
		into blocks. Only accepts two partitions of nodes.

		The NMI is a normalization of the Mutual Information (MI) score to scale the 
		results between 0 (no mutual information) and 1 (perfect correlation). 

		Letting H(U) denote the entropy of the random variable U and the MI score 
		I(U,V) = H(V) - H(V|U) = H(U) - H(U|V), the MI score can be normalised by 
		2I(U,V)/(H(U) + H(V)).

		Parameters
		----------
		labels_U : int array, shape = [number of nodes in network]
				A clustering of the data into disjoint subsets.
		label_V : array, shape = [number of nodes in network]
				A clustering of the data into disjoint subsets.
		average_method : string, optional (default: 'arithmetic')
				How to compute the normalizer in the denominator. Possible options
				are 'min', 'geometric', 'arithmetic', and 'max'.
		Returns
		-------
		nmi : float
			 score between 0.0 and 1.0. 1.0 stands for correlation.
		"""
		# Need to make sure clusterings are correct shape.
		labels_U, label_V = check_clusterings(labels_U, label_V)
		
		# Retrieve unique cluster labels in both cases.
		clusters_U = np.unique(labels_U)
		clusters_V = np.unique(label_V)
		
		# Special exception: We have no clustering if only 1 cluster class for both
		# clusterings. This is a perfect match and therefore return 1.0.
		if (clusters_U.shape[0] == clusters_V.shape[0] == 1 or clusters_U.shape[0] == clusters_V.shape[0] == 0):
				return 1.0
		
		# Construct contigency matrix between two labelings.
		contingency_matrix = create_contingency_matrix(labels_U, label_V)

		# Calculate the MI for the two clustering labels.
		mi = compute_mutual_info_score(contingency_matrix)

		# Calculate entropy for each labeling.
		entropy_U, entropy_V = calculate_entropy(labels_U), calculate_entropy(label_V)

		# Compute the denominator needed for NMI_{sum} which is based on arithmetic mean of entropy.
		normalizer = calculate_average(entropy_U, entropy_V, "arithmetic")

		# Avoid 0.0 / 0.0 when either entropy is zero.
		normalizer = max(normalizer, np.finfo('float64').eps)
		
		# Finally compute NMI.
		nmi = mi / normalizer
		return nmi