import numpy as np
import linalg_toolbox as linalg_tools

def get_projection_matrix(data, ndim, whiten=False):
    """
    Performs a PCA over the data and return the projection matrix.
    data: the data to be projected. Datapoints are given by rows while features
          are given by columns.
    ndim: the dimension of the lower dimentional subspace the data should be
          projected on.
    """
    n_datapoints, n_features = data.shape
    assert ndim <= n_features, 'We can\'t project onto a higher dimentional space.'

    St = compute_St(data)

    # Getting the eigen vectors (they correspond to the projection matrix we are
    # looking for).
    projection_matrix = linalg_tools.get_largest_eigen_vectors(St, ndim)
    if whiten:
        projection_matrix = linalg_tools.normalize_data(projection_matrix.T).T
    return projection_matrix


def project_using_projection_matrix(data_to_transform, projection_matrix):
    """
    Projects given data into lower dimentional subspace using the provided
    projection_matrix.
    """
    projected_data = data_to_transform * projection_matrix;
    return projected_data


def project(data, ndim, whiten=False):
    """
    Projects an array of data onto a lower dimentional subspace.
    The procedure here is the 'naive' one (as seen in the lectures).
    data: the data to be projected. Datapoints are given by rows while features
          are given by columns.
    ndim: the dimension of the lower dimentional subspace the data should be
          projected on.
    """
    projection_matrix = get_projection_matrix(data, ndim, whiten)
    return project_using_projection_matrix(data, projection_matrix)


def compute_St(data):
    """
    Given a dataset, computes the variance matrix of its features.
    """
    n_datapoints, n_features = data.shape

    # Computing the 'mean image'. A pixel at position (x,y) in this image is the
    # mean of all the pixels at position (x,y) of the images in the dataset.
    # This corresponds to the 'mu' we have seen in the lectures.
    mu = data.mean(axis=0) # apply along the rows for each columns.
    centered_data = data - mu

    # Computing the covariance matrix
    St = (1. / n_datapoints) * (centered_data.T * centered_data)
    return St
