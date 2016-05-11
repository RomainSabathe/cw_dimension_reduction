import numpy as np
import linalg_toolbox as linalg_tools
from PCA_scripts import compute_St
import sklearn.lda as lda
import warnings

def simple_projection(data):
    """
    Projects an array of data onto a lower dimentional subspace.
    data: the data to be projected. Datapoints are given by rows while features
          are given by columns.
    """
    n_datapoints, n_features = data.shape
    data = data.T # to retrieve the usual notation: datapoints are arranged by columns.
    data = linalg_tools.normalize_data(data)

    # We are provided with a 'mixed' dataset. Say X is the mixed dataset, and S
    # are the original sources. We assume a linear mixing such that:
    # X = AS where A is the mixing matrix.
    # The whole idea is to estimate A and invert it to recover S:
    # S = A^-1 * X
    # To do that, we try to maximise negentropy through an iterative algorithm
    weights = np.matrix(np.zeros([n_features, n_datapoints]))
    for p in range(n_datapoints):
        weights[:,p] = init_W(n_features)
        weight_p = weights[:,p]

        for k in range(100):
            term1 = (data * g(weight_p.T * data).T).mean(axis=1)
            term2 = g_prime(weight_p.T * data).mean() * weight_p
            weight_temp = term1 - term2

            # To recover more than one source, we need to procede to a Gram-Schimdt
            # othgonalisation. The matrix 'decorell' just does that.
            decorell = np.matrix(np.zeros(weight_p.shape))
            for j in range(p):
                decorell = weights[:,j] * weight_temp.T * weights[:,j]

            weight_temp = weight_temp - decorell # weight_temps columns are now othogonals.
            weight_temp = weight_temp / np.linalg.norm(weight_temp) # normalisation

            # Evaluating if weight_p has changed.
            diff = np.abs((weight_temp.T * weight_p).sum() - 1)
            weight_p = weight_temp
            if diff < 1.e-9:
                # If not, it means our algorithm has converged
                # and we can move on to the next datapoint
                break

            if k == 99:
                warnings.warn('Did not converge for weight %s. Expect poorer results.' % p)


        weights[:,p] = weight_p

    A = weights.T * data # reminder: data = A*S where S is what we are looking for.

    recovered_images = data * np.linalg.inv(A).T

    return recovered_images.T # the .T is used to get back to standard data-science
                              # layout: datapoints are arranged by rows


def init_W(n_features):
    """
    Initialize a random weight vector W.
    n_features: the number of variables used in the dataset
    """
    W = np.random.normal(0, 1, n_features)
    W = np.matrix(W)
    return W.T # is indeed a column vector


def g(x):
    """
    Function used by FastICA algorithm to estimate negentropy.
    g(x) = x * exp(-x2 / 2)
    """
    assert x.shape[0] == 1 or x.shape[1] == 1, 'This is not a vector.'
    x = np.squeeze(np.asarray(x)) # converting to an array
    return np.asmatrix(np.tanh(x))
    #return np.asmatrix(x * np.exp(-x**2 / 2))


def g_prime(x):
    """
    Function used by FastICA algorithm to estimate negentropy.
    g'(x) = (1 - x2) * exp(-x2 / 2)
    """
    assert x.shape[0] == 1 or x.shape[1] == 1, 'This is not a vector.'
    x = np.squeeze(np.asarray(x)) # converting to an array
    return np.asmatrix(1 - np.tanh(x)**2)
    #return np.asmatrix((1 - x**2) * np.exp(-x**2 / 2))
