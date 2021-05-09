from scipy.stats import wasserstein_distance
from imageio import imread
import numpy as np

def get_histogram(img):
  '''
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  h, w = img.shape[:2]
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w)

a = imread('2_HR_X = 351.2425_Y =353.2472_Z =91.99998.jpg', pilmode='L')
b = imread('output_2.jpg', pilmode='L')
a_hist = get_histogram(a)
b_hist = get_histogram(b)
dist = wasserstein_distance(a_hist, b_hist)
print(dist)