import numpy as np
import linalg_toolbox as linalg_tools

def get_projection_matrix(data, classes, ndim, normalize=False):
    """
    Performs a LDA over the data and return the projection matrix.
    data: the data to be projected. Datapoints are given by rows while features
          are given by columns.
    ndim: the dimension of the lower dimentional subspace the data should be
          projected on.
    normalize: set to True if the data should be centered and whitened before
               computing the projection matrix
    """
    n_datapoints, n_features = data.shape
    data = data.T # to retrieve the usual notation: datapoints are arranged by columns.
    if normalize:
        data = linalg_tools.normalize_data(data) # center and whiten the data

    assert ndim <= n_features, 'We can\'t project onto a higher dimentional space.'
    assert len(classes) == n_datapoints, 'Some points have no associated class, or ' \
                                         'too many classes.'

    # Computing the mean of all classes. That is to say, the mean image of
    # every class.
    mu_c = compute_mu_classes(data, classes)

    # Computing the covariance matrix 'between classes': how much are centers
    # of different classes apart from each other
    Sb = compute_Sb(data, classes, mu_c)

    # Computing the covariance matrix 'within classes': how much points within
    # clusters are separated from each other
    Sw = compute_Sw(data, classes, mu_c)

    # This is the matrix we need to compute the eigenvectors from
    fisher_matrix = np.linalg.inv(Sw) * Sb

    # Getting the eigen vectors (they correspond to the projection matrix we are
    # looking for).
    projection_matrix = linalg_tools.get_largest_eigen_vectors(fisher_matrix, ndim)
    return projection_matrix.T


def project_using_projection_matrix(data_to_transform, projection_matrix):
    """
    Projects given data into lower dimentional subspace using the provided
    projection_matrix.
    WARNING: the data is supposed to be in standard data science notation:
    rows: datapoints/individuals
    columns: variables/features
    """
    projected_data = projection_matrix * data_to_transform.T;
    return projected_data.T # the .T is used to get back to standard data-science
                            # layout: datapoints are arranged by rows


def project(data, classes, ndim, normalize):
    """
    Projects an array of data onto a lower dimentional subspace.
    The procedure here is the 'naive' one (as seen in the lectures).
    data: the data to be projected. Datapoints are given by rows while features
          are given by columns.
    ndim: the dimension of the lower dimentional subspace the data should be
          projected on.
    normalize: set to True if the data should be centered and whitened before
               computing the projection matrix
    """
    projection_matrix = get_projection_matrix(data, classes, ndim, normalize)
    return project_using_projection_matrix(data, projection_matrix)


def get_subdata(data, classes, class_to_select):
    """
    Given a dataset and the associate class for each point, returns
    the points whose class is exactly class_to_select).
    """
    subdata = data[:,classes == class_to_select]
    return subdata


def compute_mu_classes(data, classes):
    """
    Given a dataset and the associate class for each point, returns
    the mean of each of the classes as matrix mu_c where:
    each column is associated to a class
    each row is associated to a coordinate
    """
    n_features,_ = data.shape
    unique_classes = np.unique(classes)
    mu_c = np.matrix(np.zeros([n_features,len(unique_classes)]))
    for class_num,class_name in enumerate(unique_classes):
        subdata = get_subdata(data, classes, class_name)
        mu_subdata = subdata.mean(axis=1)
        mu_c[:,class_num] = mu_subdata

    return mu_c


def compute_Sb(data, classes, mu_c):
    """
    Computes the variance 'between' classes. That is to say, how well are
    each class separated from each other.
    data: the dataset
    mu_c: the mean value of each cluster. One can use compute_mu_classes to get it.
    """
    mu = data.mean(axis=1) # mean of the whole dataset

    n_features,_ = data.shape
    unique_classes = np.unique(classes)
    Sb = np.matrix(np.zeros([n_features, n_features]))
    for class_num,class_name in enumerate(unique_classes):
        subdata = get_subdata(data, classes, class_name)
        _,size_subdata = subdata.shape
        mu_subdata = mu_c[:,class_num]
        diff_mu = mu_subdata - mu
        Sb += size_subdata * (diff_mu * diff_mu.T)

    return Sb


def compute_Sw(data, classes, mu_c):
    """
    Computes the variance 'within' classes. That is to say, how spread are each
    class by themselves (how points are separated from each other within their own class).
    data: the dataset
    mu_c: the mean value of each cluster. One can use compute_mu_classes to get it.
    """
    unique_classes = np.unique(classes)
    n_features,_ = data.shape
    Sw = np.matrix(np.zeros([n_features, n_features]))
    for class_num,class_name in enumerate(unique_classes):
        subdata = get_subdata(data, classes, class_name)
        mu_subdata = mu_c[:,class_num]
        _,size_subdata = subdata.shape
        Sw += np.cov(subdata)

    return Sw
