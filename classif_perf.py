import numpy as np
import scipy.io as sio
import os
import preprocessing
import matplotlib.pyplot as plt
import image_toolbox as img_tools
import plot_toolbox as plot_tools
import PCA_scripts as PCA
import LDA_scripts as LDA
import NPP_scripts as NPP
import FastICA_scripts as FastICA
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier

"""
This file is the principal to be used in order to generate the data for the
classification error of the different techniques. It has been conceived to generate
modular data. Each run creates a result file which can be later
studied, in 'main.py' for instance.
"""

#########################################
#####   PARAMETERS  #####################
#########################################

dataset = 'YaleB_32x32'
n_folds = 20 # n of folds for cross validation
subspace_dim = 150 # dimension of the subspace on which we will project our data
to_be_tested = 'LDA' # 'Random', 'PCA', 'PCA-w', 'LDA', 'LDA-n', 'NPP-n' with n = n_neighbors

###############################################################################
###############################################################################
###############################################################################

#########################################
#####   DATA GATHERING & PREPROSS   #####
#########################################

data_filename = dataset
data_filepath = os.path.abspath('./%s.mat' % data_filename)
datamat = sio.loadmat(data_filepath)

fea = datamat['fea'] # 1 row = 1 image.
fea = np.matrix(fea, dtype='float32')
gnd = datamat['gnd']
gnd = preprocessing.normalize_classes(gnd)

#########################################
#####   CROSS-VAL PREPARATION    ########
#########################################

n_knn = 1 # number of neighbors to consider to classify a datapoint using KNN
n_images = fea.shape[0]
kf = KFold(n=n_images, n_folds=n_folds, shuffle=True, random_state=0) # create folds among the images
errors = np.matrix(np.zeros([subspace_dim, n_folds])) # 6 methods for now

for num_fold, (train_index, test_index) in enumerate(kf):
    train_images = fea[train_index,:]
    train_class = gnd[train_index]
    test_images  = fea[test_index, :]
    test_class = gnd[test_index]

    # Random classifier
    if to_be_tested == 'Random':
        for i,dim in enumerate(range(1,subspace_dim+1)):
            prediction = np.random.randint(1, len(np.unique(gnd)), test_images.shape[0])

            error_rate = 1 - (float((prediction == test_class).sum()) / len(prediction))
            errors[i,num_fold] = error_rate

    # PCA
    if to_be_tested == 'PCA':
        projection_matrix = PCA.get_projection_matrix(train_images, subspace_dim, \
                                                      whiten=False)
        projected_train_images = PCA.project_using_projection_matrix(train_images, \
                                                                    projection_matrix)
        projected_test_images = PCA.project_using_projection_matrix(test_images, \
                                                                    projection_matrix)
        classifier = KNeighborsClassifier(n_knn)
        for i,dim in enumerate(range(1,subspace_dim+1)):
            train_data_ = projected_train_images[:,range(dim)]
            test_data_ = projected_test_images[:,range(dim)]

            classifier.fit(train_data_, train_class)
            prediction = classifier.predict(test_data_)

            error_rate = 1 - (float((prediction == test_class).sum()) / len(prediction))
            errors[i,num_fold] = error_rate

    # Whiten PCA
    if to_be_tested == 'PCA-w':
        projection_matrix = PCA.get_projection_matrix(train_images, subspace_dim, \
                                                      whiten=True)
        projected_train_images = PCA.project_using_projection_matrix(train_images, \
                                                                    projection_matrix)
        projected_test_images = PCA.project_using_projection_matrix(test_images, \
                                                                    projection_matrix)
        classifier = KNeighborsClassifier(n_knn)
        for i,dim in enumerate(range(1,subspace_dim+1)):
            train_data_ = projected_train_images[:,range(dim)]
            test_data_ = projected_test_images[:,range(dim)]

            classifier.fit(train_data_, train_class)
            prediction = classifier.predict(test_data_)

            error_rate = 1 - (float((prediction == test_class).sum()) / len(prediction))
            errors[i,num_fold] = error_rate

    # LDA
    if to_be_tested == 'LDA':
        projection_matrix = LDA.get_projection_matrix(train_images, train_class, subspace_dim, \
                                                      normalize=False)
        projected_train_images = LDA.project_using_projection_matrix(train_images, \
                                                                    projection_matrix)
        projected_test_images = LDA.project_using_projection_matrix(test_images, \
                                                                    projection_matrix)
        classifier = KNeighborsClassifier(n_knn)
        for i,dim in enumerate(range(1,subspace_dim+1)):
            train_data_ = projected_train_images[:,range(dim)]
            test_data_ = projected_test_images[:,range(dim)]

            classifier.fit(train_data_, train_class)
            prediction = classifier.predict(test_data_)

            error_rate = 1 - (float((prediction == test_class).sum()) / len(prediction))
            errors[i,num_fold] = error_rate

    # LDA normalized
    if to_be_tested == 'LDA-n':
        projection_matrix = LDA.get_projection_matrix(train_images, train_class, subspace_dim, \
                                                      normalize=True)
        projected_train_images = LDA.project_using_projection_matrix(train_images, \
                                                                    projection_matrix)
        projected_test_images = LDA.project_using_projection_matrix(test_images, \
                                                                    projection_matrix)
        classifier = KNeighborsClassifier(n_knn)
        for i,dim in enumerate(range(1,subspace_dim+1)):
            train_data_ = projected_train_images[:,range(dim)]
            test_data_ = projected_test_images[:,range(dim)]

            classifier.fit(train_data_, train_class)
            prediction = classifier.predict(test_data_)

            error_rate = 1 - (float((prediction == test_class).sum()) / len(prediction))
            errors[i,num_fold] = error_rate

    # NPP
    if 'NPP' in to_be_tested:
        # Getting the number of neighbors
        _, n_neighbors = to_be_tested.split('-')
        n_neighbors = int(n_neighbors)
        projection_matrix = NPP.get_projection_matrix(train_images, subspace_dim, n_neighbors)
        projected_train_images = NPP.project_using_projection_matrix(train_images, \
                                                                    projection_matrix)
        projected_test_images = NPP.project_using_projection_matrix(test_images, \
                                                                    projection_matrix)
        classifier = KNeighborsClassifier(n_knn)
        for i,dim in enumerate(range(1,subspace_dim+1)):
            train_data_ = projected_train_images[:,range(dim)]
            test_data_ = projected_test_images[:,range(dim)]

            classifier.fit(train_data_, train_class)
            prediction = classifier.predict(test_data_)

            error_rate = 1 - (float((prediction == test_class).sum()) / len(prediction))
            errors[i,num_fold] = error_rate


errors = pd.DataFrame(errors) # easier way to create CSV
errors.index = [i for i in range(1, subspace_dim+1)]
index_label = 'ndim'
header = ['fold_%s' % (i+1) for i in range(n_folds)]
errors.to_csv('./perf_results/detail_%s.csv' % to_be_tested, index_label=index_label, header=header)
errors.mean(axis=1).to_csv('./perf_results/mean_%s.csv' % to_be_tested, \
                           index_label=index_label, header=['mean_error'])
