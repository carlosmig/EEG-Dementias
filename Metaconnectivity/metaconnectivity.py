# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:56:20 2023

Arbabyazd, Lucas M., et al. "Dynamic functional connectivity as a 
complex random walk: definitions and the dFCwalk toolbox." 
MethodsX 7 (2020): 101168.

@author: Carlos
"""


import numpy as np


def matrix_recon(x):
    """
    reconstructs a matrix from its upper triangular form
    
    Parameters
    ----------
    x : 1xM numpy array or list, M = number of connections
        vectorized matrix (upper triangular)

    Returns
    -------
    matrix : NxN numpy array, with N nodes
             Connectivity matrix

    """
    npairs = len(x)
    nnodes = int((1 + np.sqrt(1 + 8 * npairs)) // 2)
    
    matrix = np.zeros((nnodes, nnodes))
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            matrix[row, col] = x[idx]
            idx = idx + 1
    matrix = matrix + matrix.T
   
    return(matrix)   


def get_uptri(x):
    """
    Get the upper triangular of a connectivity matrix

    Parameters
    ----------
    x : NxN numpy array, with N nodes
        Connectivity matrix

    Returns
    -------
    vector: 1XM numpy array, with M connections
            vectorized connectivity matrix

    """
    nnodes = x.shape[0]
    npairs = (nnodes**2 - nnodes) // 2
    vector = np.zeros(npairs)
    
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            vector[idx] = x[row, col]
            idx = idx + 1
    
    return(vector)


def meta_strength(Meta_FC, nnodes):
    """
    Compute the metaconnectivity nodal strength. Its equivalent
    to the nodal strength of the pairwise FC matrices.

    Parameters
    ----------
    Meta_FC : MxM numpy array, with M connectivity pairs
              Metaconnectivity matrix
    nnodes : integer
             Number of nodes.

    Returns
    -------
    nodal_strength : 1xN numpy array, with N nodes
                     nodal strength

    """
   
    nodal_strength = np.zeros(nnodes)
    idx0, idx1, idx2 = np.array([]), 0, 0
    npairs = (nnodes**2 - nnodes) / 2
    vector = np.sum(Meta_FC, 0) / npairs
    
    for i in range(0,nnodes):
    
        if i == 1:
            idx0 = np.append(idx0, np.array([0]), 0)
        if i > 1:
            idx0 += 1
            idx0 = np.append(idx0, np.array([idx0[-1] + nnodes - i]))        
        idx0 = idx0.astype(int)           
        idx1 += (nnodes - i + 1) * (i > 0)
        idx2 += nnodes - i
        
        nodal_strength[i] = (np.sum(vector[idx0]) + np.sum(vector[idx1:idx2])) / (nnodes - 1)
    
    return(nodal_strength)


def Meta_Con(data, windows_size, olap):
    """
    Compute the metaconnectivity matrix using sliding windows.

    Parameters
    ----------
    data : TxN numpy array. T for time, N for nodes
           time series of brain activity
    windows_size : integer
                   window size in points
    olap : float
           windows' overlap between 0 and 1 (1 = 100%, 0.5 = 50% 0 = 0%)

    Returns
    -------
    Meta_FC : M x M numpy array, with M pairs of connections
              Metaconnectivity matrix
    FCs : W x M numpy array, with W time windows
          Time resolved functional connectivity

    """
    
    T, N = data.shape
    N_pairs = int((N**2 - N) / 2)
    
    windows_number = int((T // windows_size) / (1 - olap))
    
    FCs = np.zeros((N_pairs, windows_number))
    
    for i in range(0, windows_number):
        idx0, idx1 = int(i * windows_size * (1- olap)), int((i * (1 - olap) + 1) * windows_size) 
             
        FCs[:,i] = get_uptri(np.corrcoef(data[idx0:idx1,:].T))
    
    Meta_FC = np.corrcoef(FCs)
       
    return([Meta_FC, FCs])



    
    
        
        


