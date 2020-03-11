import sys
import time as t

import numpy as np
from skimage.feature import hog, BRIEF

def hog_single(img, descriptor_size, num_orientations):
    rows, cols = img.shape
    img_padded = np.pad(img, ((int(descriptor_size/2), int(descriptor_size/2)),(int(descriptor_size/2), int(descriptor_size/2))))

    img_hog = np.zeros((rows, cols, num_orientations))

    for i in range(rows):
        for j in range(cols):
            image = img_padded[i:i+descriptor_size, j:j+descriptor_size]
            img_hog[i,j,:] = hog(image, orientations=num_orientations, pixels_per_cell=(descriptor_size, descriptor_size), cells_per_block=(1,1))
    return img_hog

def apply_hog(left, right, descriptor_size, num_orientations):
    """
    computes HOG descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: number of pixels in a hog cell.
    :param num_orientations: number of HOG orientations.
    :return: (H x W x M) array, H = height, W = width and M = num_orientations, of type np.float32.
    """
    # TODO: apply HOG descriptor on left and right images.

    left_rows, left_cols = left.shape
    right_rows, right_cols = right.shape

    left_padded = np.pad(left, ((int(descriptor_size/2), int(descriptor_size/2)),(int(descriptor_size/2), int(descriptor_size/2))))

    left_hog = hog_single(left, descriptor_size, num_orientations)
    right_hog = hog_single(right, descriptor_size, num_orientations)

    return (left_hog, right_hog)



def convert_brief(descriptor):
    return descriptor.dot(2 ** np.arange(descriptor.size)[::-1])

def apply_brief(left, right, descriptor_size, num_elements):
    """
    computes BRIEF descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: size of window of the BRIEF descriptor.
    :param num_elements: length of the feature vector.
    :return: (H x W) array, H = height and W = width, of type np.int64
    """
    # TODO: apply BRIEF descriptor on both images. You will have to convert the BRIEF feature vector to a int64.

    extractor = BRIEF(descriptor_size=num_elements, patch_size=descriptor_size, mode='normal')

    left_rows, left_cols = left.shape

    left_indices = np.empty((left_rows, left_cols, 2))
    left_indices[..., 0] = np.arange(left_rows)[:, None]
    left_indices[..., 1] = np.arange(left_cols)
    left_indices = left_indices.reshape(left_cols * left_rows, 2)

    right_rows, right_cols = right.shape

    right_indices = np.empty((right_rows, right_cols, 2))
    right_indices[..., 0] = np.arange(right_rows)[:, None]
    right_indices[..., 1] = np.arange(right_cols)
    right_indices = right_indices.reshape(right_cols * right_rows, 2)

    extractor.extract(left, left_indices)
    left_desc = extractor.descriptors.astype(np.int64).reshape(left_rows-descriptor_size+1, left_cols-descriptor_size+1, 128)
    extractor.extract(right, right_indices)
    right_desc = extractor.descriptors.astype(np.int64).reshape(right_rows-descriptor_size+1, right_cols-descriptor_size+1, 128)

    left_desc = np.pad(np.apply_along_axis(convert_brief, 2, left_desc),((3,4), (3,4)))
    right_desc = np.pad(np.apply_along_axis(convert_brief, 2, right_desc),((3,4), (3,4)))

    return (left_desc, right_desc)
