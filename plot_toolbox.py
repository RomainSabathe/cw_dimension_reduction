import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

plt.style.use('ggplot')

def plot_datapoints(data, dim_x, dim_y, classes=None, save_folder=None, name=None):
    """
    Displays a scatter plot of the data given in argument.
    data:    the data to be plotted. Datapoints should be given as rows and features
             as columns.
    dim_x:   the column number (feature number) to be used as x-axis.
    dim_y:   the column number (feature number) to be used as y-axis.
    classes: is an array which size equals the number of rows of 'data' and which
             associate each datapoint to its corresponding class.
    """
    n_datapoints, n_features = data.shape

    fig = plt.figure(figsize=(15,15))
    color_classes = LabelEncoder().fit_transform(classes)
    plt.scatter(data[:,dim_x], data[:,dim_y], c=color_classes, s=40., alpha=0.65,
                linewidths=1.5, marker='D')

    plt.grid(True)
    plt.xlabel('Dimension %s' % dim_x)
    plt.ylabel('Dimension %s' % dim_y)

    if save_folder is None or name is None:
        plt.show()
    else:
        assert save_folder is not None and name is not None, 'Whether plot folder or' \
                                                             'filename is missing.'
        path = os.path.join(os.getcwd(), save_folder)
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, '%s.png' % name)
        plt.savefig(file_path)


def plot_error_rate(errors, legend):
    dim_subspace_max, n_methods = errors.shape

    fig = plt.figure(figsize=(10,10))
    x_range = range(1, dim_subspace_max+1)
    for method in range(n_methods):
        plt.plot(x_range, errors[:,method], label=legend[method], linewidth=3, alpha=0.8)

    plt.xlim([0, dim_subspace_max+1])
    plt.ylim([0,1])

    plt.xlabel('Number of dimensions used by KNN')
    plt.ylabel('Error rate')
    plt.legend(loc='best')

    plt.savefig('./perf_results/result.png')
    plt.show()
