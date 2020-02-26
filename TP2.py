import numpy as np
from skimage import io
from skimage.feature import (match_descriptors, corner_peaks, corner_fast,
                             plot_matches, BRIEF)
import matplotlib.pyplot as plt

img1 = io.imread("./data/training/image_2/000000_10.png", as_gray=True)
img2 = io.imread("./data/training/image_2/000000_11.png", as_gray=True)

pc1 = corner_peaks(corner_fast(img1), min_distance=5)
pc2 = corner_peaks(corner_fast(img2), min_distance=5)

plt.figure(figsize=(6,6))
plt.imshow(img1, cmap='gray')
plt.plot(pc1[:, 1], pc1[:, 0], '+r', markersize=3)
#ax.axis((0, 350, 350, 0))
plt.show()

extractor = BRIEF(descriptor_size=128, patch_size=49, mode='normal')

extractor.extract(img1, pc1)
desc1 = extractor.descriptors
extractor.extract(img2,pc2)
desc2 = extractor.descriptors
matches = match_descriptors(desc1, desc2, metric="hamming", cross_check=True)

fig = plt.figure(figsize=(18, 10))
ax0 = plt.subplot()
plot_matches(ax0,img1, img2, pc1, pc2, matches)
ax0.axis('off')
ax0.set_title("Image1 vs. Image2")
plt.show()




