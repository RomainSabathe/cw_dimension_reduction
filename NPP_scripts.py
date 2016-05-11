import numpy as np
import linalg_toolbox as linalg_tools
import pickle
import os

def get_projection_matrix(data, ndim, n_neighbors=5):
    """
    Performs a NPP over the data and return the projection matrix.
    data: the data to be projected. Datapoints are given by rows while features
          are given by columns.
    ndim: the dimension of the lower dimentional subspace the data should be
          projected on.
    """
    n_datapoints, n_features = data.shape
    assert ndim <= n_features, 'We can\'t project onto a higher dimentional space.'

    data = data.T # to retrieve the usual notation: datapoints are arranged by columns.
    data = linalg_tools.center_data(data) # this is necessary to unsure that all
                                          # constraints of NPP are satisfied.
                                          # (the transformed data is whitened, for instance.)

    # Getting the distance matrix if required
    #distance_matrix_file = os.path.join(os.getcwd(), 'distance_matrix.pickle')
    #if os.path.isfile(distance_matrix_file):
    #    distance_matrix = pickle.load(open(distance_matrix_file, 'r'))
    #else:
    #    distance_matrix = get_distance_matrix(data)
    #    pickle.dump(distance_matrix, open(distance_matrix_file, 'w'))
    distance_matrix = get_distance_matrix(data)
    W = construct_W(data, distance_matrix, n_neighbors)

    # Calculating L
    diff = np.identity(n_datapoints) - W.T
    L = data * diff * diff.T * data.T

    # Calculating C (covariance matrix)
    C = data * data.T

    # Computing the projection matrix
    main_matrix = np.linalg.inv(C) * L
    projection_matrix = linalg_tools.get_lowest_eigen_vectors(main_matrix, ndim)
    return projection_matrix


def project_using_projection_matrix(data_to_transform, projection_matrix):
    """
    Projects given data into lower dimentional subspace using the provided
    projection_matrix.
    """
    projected_data = projection_matrix.T * data_to_transform.T
    return projected_data.T


def project(data, ndim, n_neighbors):
    """
    Projects an array of data onto a lower dimentional subspace.
    The procedure here is the 'naive' one (as seen in the lectures).
    data: the data to be projected. Datapoints are given by rows while features
          are given by columns.
    ndim: the dimension of the lower dimentional subspace the data should be
          projected on.
    """
    projection_matrix = get_projection_matrix(data, ndim, n_neighbors)
    return project_using_projection_matrix(data, projection_matrix)


def dist_vect(data, num_datapoint1, num_datapoint2):
    """
    Returns the euclidean distance between the datapoints data[num_datapoint1,:] and
    data[num_datapoint2,:].
    """
    diff_vect = data[:,num_datapoint1] - data[:,num_datapoint2]
    #dist = np.linalg.norm(diff_vect)
    dist = diff_vect.T * diff_vect
    return dist


def get_distance_matrix(data):
    """
    Returns a distance matrix of the datapoints in the data.
    """
    # TODO: make this function more efficient.
    n_features, n_datapoints = data.shape
    distance_matrix = np.matrix(np.zeros([n_datapoints, n_datapoints]))
    for dp1 in range(n_datapoints): # dp1 = datapoint 1
        for dp2 in range(dp1 + 1, n_datapoints):
            dist = dist_vect(data, dp1, dp2)
            distance_matrix[dp1, dp2] = dist
            distance_matrix[dp2, dp1] = dist

    return distance_matrix


def get_neighbors(data, distance_matrix, num_datapoint, n_neighbors):
    """
    Returns a matrix containing the neighbors coordinates of the datapoint data[num_datapoint,:].
    """
    n_features, n_datapoints = data.shape
    num_closest_neighbors = get_neighbors_index(data, distance_matrix, num_datapoint, n_neighbors)
    neighbors = np.matrix(np.zeros([n_features, n_neighbors]))
    for i,neighbor in enumerate(num_closest_neighbors):
        neighbors[:,i] = data[:,neighbor]

    return neighbors


def get_neighbors_index(data, distance_matrix, num_datapoint, n_neighbors):
    """
    Returns a matrix containing the neighbors index of the datapoint data[num_datapoint,:].
    """
    distances_to_neighbors = distance_matrix[:,num_datapoint]
    distances_to_neighbors = np.squeeze(np.asarray(distances_to_neighbors)) # We can now iterate
                                                                            # over its elements.
    distances_to_neighbors = [(n_neighbor, dist) for (n_neighbor,dist) in \
                            enumerate(distances_to_neighbors)]
    distances_to_neighbors.sort(key=lambda x: x[1]) # sort based on distance
    num_closest_neighbors = [num for num,_ in \
              distances_to_neighbors[1:n_neighbors+1]] # we ignore the distance to the
                                                       # datapoint itself.

    return num_closest_neighbors



def construct_W(data, distance_matrix, n_neighbors=5):
    """
    Constructs the matrix W of the 'recovered data' from the neighborhood.
    More precisely, if X_i is a datapoint from data, (it's a column vector),
    X_n(i) is a matrix of X_i's neighbors (as given by get_neighbors), and
    W_i is the ith line of W, then the following relation holds:
    X_i =(approx) X_i * W_i.T
    W can approximate the data based on the neighbors of this very data.
    W[i,j] is the weight of the jth neighbor in the reconstruction of the ith datapoint.
    Parameters:
    data:            the data
    distance_matrix: the distance between each point of the dataset, as given
                     by get_distance_matrix
    n_neighbors:     the number of neighbors that should be used in order to recover
                     the data
    """
    n_features, n_datapoints = data.shape
    W = np.matrix(np.zeros([n_datapoints,n_datapoints]))

    for i in range(n_datapoints):
        datapoint = data[:,i]
        neighbors = get_neighbors(data, distance_matrix, i, n_neighbors)
        gram = get_gram_matrix(datapoint, neighbors)
        try:
            gram_inverse = np.linalg.inv(gram)
        except:
            gram_inverse = np.linalg.inv(gram + (1.e-6 * np.identity(n_neighbors)))

        vect_ones = np.matrix(np.ones(n_neighbors)).T
        W_i = gram_inverse * vect_ones
        W_i = W_i / np.sum(W_i) # normalisation

        index_neighbors = get_neighbors_index(data, distance_matrix, i, n_neighbors)
        W[i,index_neighbors] = W_i.T

    return W


def get_gram_matrix(datapoint, neighborhood_matrix):
    """
    Takes a datapoint (as a line vector) and its associated neighborhood matrix
    as given by 'get_neighbors' and returns the corresponding Gram matrix.
    """
    diff_to_neighbors = datapoint - neighborhood_matrix
    gram_matrix = diff_to_neighbors.T * diff_to_neighbors
    return gram_matrix
