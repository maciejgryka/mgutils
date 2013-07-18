# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import map_coordinates


def get_image_profile(image, p0, p1, n_samples=None):
    """
    Extract intensity profile from underneath the p0--p1 line in image.
    http://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array

    """
    if n_samples is None:
        n_samples = int(np.linalg.norm(np.array(p0) - np.array(p1)))
    x = np.linspace(p0[0], p1[0], n_samples)
    y = np.linspace(p0[1], p1[1], n_samples)

    if len(image.shape) == 2:
        image = image[:,:,None]
    n_channels = image.shape[2]
    profile = np.zeros([n_samples, n_channels])
    for c in range(n_channels):
        profile[:,c] = map_coordinates(image[:,:,c], np.vstack((x,y)))

    # Extract the values along the line, using cubic interpolation
    return profile
