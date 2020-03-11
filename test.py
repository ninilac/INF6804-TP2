import numpy as np
from skimage import io
from skimage.feature import (match_descriptors, corner_peaks, corner_fast,
                             plot_matches, BRIEF)
import matplotlib.pyplot as plt

img1 = io.imread("./data/training/image_2/000000_10.png", as_gray=True)
img2 = io.imread("./data/training/image_2/000000_11.png", as_gray=True)
extractor = BRIEF(descriptor_size=128, patch_size=9, mode='normal')

left_rows, left_cols = img1.shape

left_indices = np.empty((left_rows,left_cols,2))
left_indices[..., 0] = np.arange(left_rows)[:, None]
left_indices[..., 1] = np.arange(left_cols)
left_indices = left_indices.reshape(left_cols*left_rows, 2)

#extractor.extract()