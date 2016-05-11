import numpy as np

def normalize_classes(gnd):
    """
    When we use scipy.io, the vector of classes is shaped like array[array[]].
    We only want an array[]. This function does that if required.
    This assumes every datapoint belongs to only one class.
    """
    assert isinstance(gnd, (np.ndarray, list))

    normalized_gnd = []
    for class_ in gnd:
        if isinstance(class_, (np.ndarray, list)):
            assert len(class_) == 1, 'Datapoint belongs to several classes.'
            normalized_gnd.append(class_[0])
        else:
            normalized_gnd.append(class_)
    return np.array(normalized_gnd)


def method_name(filename):
    """
    Analyses the name of a file and returns the associate method of classification.
    Eg: 'detail_Random.csv', 'mean_NPP_20.csv'
    """
    filename, _ = filename.split('.csv') # discard the .csv
    parts = filename.split('_') # parts[0] = mean/detail || parts[1] = PCA etc..
    method_name = parts[1]
    if method_name == 'PCA-w':
        method_name = 'Whiten PCA'
    elif method_name == 'LDA-n':
        method_name = 'LDA with normalised data'
    elif 'NPP' in method_name:
        method_name, n_neighbors = method_name.split('-')
        method_name = 'NPP (%s neighbors)' % int(n_neighbors)

    return method_name
