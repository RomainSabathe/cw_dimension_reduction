import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import method_name
from plot_toolbox import plot_error_rate

###############################################################################
### CLASSIFICATION EXPERIMENTS  ###############################################
###############################################################################

errors = []
method_names = []
order = ['Random',
         'PCA',
         'Whiten PCA',
         'LDA',
         'LDA with normalised data',
         'NPP (10 neighbors)',
         'NPP (30 neighbors)']

# Retrieving the different perf files
root = os.path.join(os.getcwd(), './perf_results/')
for method_toplot in order:
    for file in os.listdir(root):
        if 'detail' in file:
            continue

        mn = method_name(file)
        if mn != method_toplot:
            continue
        method_names.append(mn)

        full_path = os.path.join(root, file)
        data = pd.read_csv(full_path)
        error = data['mean_error']
        errors.append(np.array(error))

errors = np.matrix(errors).T
plot_error_rate(errors, method_names)

###############################################################################
### FAST ICA EXPERIMENTS ######################################################
###############################################################################
#
#n_images = 3
#np.random.seed(0)
#random_images_index = np.random.randint(0, fea.shape[0], n_images)
#random_images_index = [0, 260, 520]
#original_images = fea[random_images_index,:]
#
#mixing_matrix = np.matrix(np.random.rand(n_images, n_images))
#mixed_images = mixing_matrix * original_images
#
##from sklearn.decomposition import FastICA
##coefficients = np.matrix(FastICA(algorithm='deflation').fit_transform(mixed_images))
##recovered_images = coefficients * mixed_images
#recovered_images = FastICA.simple_projection(mixed_images)
#
#for i in range(original_images.shape[0]):
#    img_tools.display_image(original_images[i,:], save_folder='images/FastICA/reconstruction/',
#                            name='original_%s' % i)
#
#for i in range(mixed_images.shape[0]):
#    img_tools.display_image(mixed_images[i,:], save_folder='images/FastICA/reconstruction/',
#                            name='mixed_%s' % i)
#
#for i in range(recovered_images.shape[0]):
#    img_tools.display_image(recovered_images[i,:], save_folder='images/FastICA/reconstruction/',
#                            name='recovered_%s' % i)
