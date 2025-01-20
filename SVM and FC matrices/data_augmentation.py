# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 10:59:37 2023

@author: Sebastian Moguilner & Carlos Coronel
"""

import numpy as np
import matplotlib.pyplot as plt

# Data Augmentation by Interpolation
def interpolate_matrices(mat1, mat2, alpha):
    """
    Linearly interpolate between two matrices.

    Args:
    - mat1 (ndarray): First matrix.
    - mat2 (ndarray): Second matrix.
    - alpha (float): Interpolation factor (0 gives mat1, 1 gives mat2).

    Returns:
    - ndarray: Interpolated matrix.
    """
    return (1 - alpha) * mat1 + alpha * mat2

def augment_data(data, labels, target_age):
    """
    Interpolate matrices corresponding to the nearest age values.

    Args:
    - data (ndarray): EEG Connectivity matrices.
    - labels (ndarray): Age labels.
    - target_age (int): Target age value for which to generate a matrix.

    Returns:
    - ndarray: Interpolated matrix for the target age.
    """
    # Find the indices of the nearest age values
    lower_idx = np.max(np.where(labels < target_age))
    upper_idx = np.min(np.where(labels > target_age))

    # Compute the interpolation factor
    alpha = (target_age - labels[lower_idx]) / (labels[upper_idx] - labels[lower_idx])

    return interpolate_matrices(data[lower_idx], data[upper_idx], alpha)

    
#%%    
if __name__ == '__main__':

    # Simulating Data
    n_samples = 100
    # Between -1 and 1
    data = 2 * np.random.random_sample((n_samples, 82, 82)) - 1
    
    # Two clustered subject ages, young and old
    labels = np.concatenate([np.random.randint(20, 30, n_samples // 2),
                              np.random.randint(60, 80, n_samples // 2)])
    
    # Test the Augmentation
    target_age = 45
    augmented_matrix = augment_data(data, labels, target_age)
    
    # Visualization
    plt.imshow(augmented_matrix, vmin=-1, vmax=1, cmap='seismic')
    plt.title(f'Interpolated Matrix for Age {target_age}')
    plt.colorbar()
    plt.show()
    
    











