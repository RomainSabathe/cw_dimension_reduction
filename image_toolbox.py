import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import os

def display_image(vector_image, save_folder=None, name=None):
    """
    Displays a black and white image where;
    vector_image: is a numerical array (values from 0 to 255) of a squared picture
                  in black and white.
    """
    vector_image = normalize_image(vector_image)
    assert np.sqrt(len(vector_image)).is_integer(), 'This is not a squared picture.'
    #assert np.min(vector_image) >= 0 or \
    #       np.max(vector_image) <= 255, 'Invalid color values for this picture.'

    dim_x = dim_y = int(np.sqrt(len(vector_image))) # Number of pixels on the x and y axis.
    matrix_image = np.resize(vector_image, (dim_x, dim_y))
    matrix_image = np.transpose(matrix_image) # otherwise the image would be upside down.

    plt.figure(figsize=[5, 5])
    plt.imshow(matrix_image, cmap='gray')

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


def normalize_image(vector_image, new_min=0., new_max=255.):
    """
    Transforms a vector image which has possibly float values from a min < 0 and
    a max > 255 to a standardized vector image.
    """
    # Getting useful constants.
    min_ = np.min(vector_image)
    max_ = np.max(vector_image)

    # Normalizing (in values range and float->int)
    normalize = lambda x: int((new_max * (x - min_) + new_min) / (max_ - min_))
    normalized_image = np.apply_along_axis(normalize, axis=0, arr=vector_image)

    return normalized_image

