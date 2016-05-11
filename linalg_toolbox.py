import numpy as np

def get_largest_eigen_vectors(matrix, n=None):
    """
    Computes the eigen vectors of 'matrix' and returns the n largest ones.
    If n is set to None, then return them all in decreasing order.
    """
    _, n_features = matrix.shape
    if n is None:
        n = n_features

    assert n > 0, 'Need to select at least one eigen vector.'

    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    idx = eigen_values.argsort()[::-1] # ordering the eigen values in decreasing order
    ordered_eigen_vectors = eigen_vectors[:,idx] # ordering the eigen vectors
    eigen_vectors = ordered_eigen_vectors[:,range(n)]

    return eigen_vectors


def get_lowest_eigen_vectors(matrix, n=None):
    """
    Computes the eigen vectors of 'matrix' and returns the n lowest ones.
    If n is set to None, then return them all in increasing order.
    """
    _, n_features = matrix.shape
    if n is None:
        n = n_features

    assert n > 0, 'Need to select at least one eigen vector.'

    eigen_values, eigen_vectors = np.linalg.eig(matrix)
    idx = eigen_values.argsort() # ordering the eigen values in decreasing order
    ordered_eigen_vectors = eigen_vectors[:,idx] # ordering the eigen vectors
    eigen_vectors = ordered_eigen_vectors[:,range(n)]

    return eigen_vectors


def as_column_vect(vector):
    """
    Takes a row or a columns vector and makes sure it returns a column vector.
    """
    try:
        nrow, ncol = vector.shape
        assert nrow == 1 or ncol == 1, 'This is not a vector'
    except ValueError: # case where we have an array instead of a matrix.
        vector = np.matrix(vector)
        nrow, ncol = vector.shape

    if ncol == 1:
        return vector
    return vector.T


def whiten_data(data):
    """
    Returns a transformed version of 'data' such that its covariance matrix
    is equal to the identity.
    The data is assumed to be shaped such that:
    rows: the features/variables
    columns: the individuals
    """
    covariance_matrix = data * data.T
    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)
    eig_values = np.abs(eig_values) # some values are so close to 0 that they are
                                    # estimated as negative
    D_m12 = np.diag(1. / np.sqrt(eig_values)) # D_m12 == D^(-1/2)
    whitened_data = (eig_vectors * D_m12 * eig_vectors.T) * data
    return whitened_data


def center_data(data):
    """
    Returns a transformed version of 'data' such that the column-vector of its
    mean is null.
    The data is assumed to be shaped such that:
    rows: the features/variables
    columns: the individuals
    """
    return data - data.mean(axis=1)


def normalize_data(data):
    """
    Center the datapoints and whiten them.
    """
    centered_data = center_data(data)
    whitened_centered_data = whiten_data(centered_data)
    return whitened_centered_data


def filter_low_values(values, threshold=1.e-11):
    """
    Filter low and negative values of the array 'values'.
    If a value is found to be below the threshold: normalize it by 0
    """
    return np.array([v if v > threshold else 0 for v in values])
