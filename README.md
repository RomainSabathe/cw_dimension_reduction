Coursework - Dimensionality reduction and face classification
=================================================

Coursework where we had to program some dimensionality reduction techniques and apply them to face classification.

Goals and context
--------------------
The goal of this coursework is to gain a deeper understanding at different methods of dimensionality reduction.
Besides seeing how these techniques can be used to understand how the data is shaped, we will further use these
techniques as a way to decrease the complexity of our dataset and yet perform classification with satisfying
results. The last part is dedicated to the application of a technique of independent component analysis: FastICA.
To do that, we will use a dataset provided by the University of Yale ([Yale Faces B](http://cvc.yale.edu/projects/yalefacesB/yalefacesB.html)). It contains roughly 2400 black and white pictures of some 37 people.

Objective and methodology
----------------
Given a dataset, we project it into a subdimensional space using different dimensionality reduction techniques and then perform a really simple classification procedure (here, it is k-nearest-neighborhood with k = 1). We do this while increasing the size of the projection space with the idea that, with more dimensions come more discriminant capabilities. The goal is to see, for a given subspace, which algorithm succeeded in spreading the data the most and therefore realised the best classification performance. We can expect this algorithm to be LDA as it is exactly designed to try to spread clusters away while packing points belonging to the same cluster together.

We performed a 20 folds-cross-validation. For each fold, we create a train set I train and a test set $I_\text{test}$ ($I$ stands for ”images”). A projection matrix is obtained using $I_\text{train}$ only and we use this very projection matrix to project $I_\text{test}$. It is within the latent space that we perform k-NN. For each fold and each dimension of the latent space, a classification error rate is calculated and reported within the folder ”perf results”.

Results
----------------
The final plot can be seen in the "perf_results" folder.

![Final results]
(https://github.com/RomainSabathe/cw_dimension_reduction/blob/master/perf_results/result.png)

About the code
----------------
One must distinguish between the "toolboxes" that are sets of useful functions, the real dimensionality reduction scripts and the general scripts that taken advantage of all of them to produce results.

1. General scripts
  1. *main.py*: Gathers the classification errors produced by *classif_perf.py* or analyses FastICA outputs.
  2. *classif_perf.py*: Proceeds to a dimensionality reduction technique and apply k-nearest-neighbors to evaluate its performance in classification.
2. Dimensionality reduction techniques
  1. *PCA_scripts.py*
  2. *LDA_scripts.py*
  3. *NPP_scripts.py* (neighborhood-preserving projections)
  4. *FastICA_scripts.py*
3. Toolboxes
  1. *preprocessing.py*
  2. *linalg_toolbox.py*: Used to perform data centering, data whitening or compute eigen vectors...
  3. *image_toolbox.py*: Used to display images.
  4. *plots_toolbox.py*: Used to display plots (datapoints in the space of projection or classification performance).


