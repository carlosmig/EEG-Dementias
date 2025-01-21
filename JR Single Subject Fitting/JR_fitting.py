# -*- coding: utf-8 -*-
"""
Created on Mon June 20 14:57:25 2025

This script performs optimization of the Jansen-Rit neural mass model parameters to fit 
empirical functional connectivity data.
"""

import numpy as np
from scipy import signal
import time
from skimage.metrics import structural_similarity as ssim
import JansenRitModelMulti as JR
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

def get_uptri(x):
    """
    Extract the upper triangular part of a matrix as a vector.
    
    Parameters:
    x : numpy array
        Input square matrix.
    
    Returns:
    vector : numpy array
        Upper triangular elements of the input matrix.
    """
    nnodes = x.shape[0]
    npairs = (nnodes**2 - nnodes) // 2
    vector = np.zeros(npairs)
    
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            vector[idx] = x[row, col]
            idx += 1
    
    return vector

# Load empirical functional connectivity (FC) data
FC_emp = np.load('FC_south_avg.npy')

# Load structural connectivity (SC) data
SC_CN = np.load('SC_south_avg.npy')

# Set simulation parameters
JR.dt = 1E-3  # Integration step size
JR.teq = 10    # Equilibrium time (in seconds)
JR.tmax = 600  # Signal length (in seconds)
JR.downsamp = 10  # Downsampling factor to reduce data size

# Model parameters
JR.sigma = 1  # Noise scaling factor
JR.nnodes = SC_CN.shape[0]  # Number of nodes
nnodes = JR.nnodes
JR.norm = 1
JR.p = 220 * np.ones(nnodes)  # External input parameter
JR.alpha = 0.5 * np.ones(nnodes)  # Excitatory synaptic gain
JR.gamma = 1 - JR.alpha  # Inhibitory synaptic gain

# Plasticity parameters
JR.plasticity_on = 1  # Enable plasticity
JR.tau_C4 = 2  # Time constant for plasticity dynamics
JR.target = 2.5 * np.ones(nnodes)  # Target firing rate

# Define filter parameters for EEG signal processing
resolution = JR.dt * JR.downsamp
Fmin, Fmax = 0.5, 40  # Frequency range (Hz)
a0, b0 = signal.bessel(3, [2 * resolution * Fmin, 2 * resolution * Fmax], btype='bandpass')

# Number of random seeds for simulation
seeds = 20

# Parameter search space
G_vec = np.linspace(0, 0.5, 51)  # Range of global coupling values

# Optimization parameters for excitatory/inhibitory balance
iters_rE = 51
delta_rE = 0.05  # Step size for parameter adjustment

# Optimization parameters for effective connectivity
iters_SC = 51
delta_SC = 0.1  # Step size for connectivity update

JR.update()

# Start optimization
init = time.time()

# Step 1: Fit global coupling (G)
GoF_Gs = np.zeros((len(G_vec), seeds))  # Store goodness-of-fit values for each G and seed
      
for gx in range(len(G_vec)):
    for sx in range(seeds):
        JR.seed = sx  # Set random seed for reproducibility
        JR.K = G_vec[gx]  # Set global coupling parameter
        JR.M = np.copy(SC_CN)  # Copy initial structural connectivity
        
        # Run the Jansen-Rit model simulation
        y, time_vector = JR.Sim(verbose=False)
        
        # Compute EEG signals as the difference between excitatory and inhibitory components
        EEG_signals = (JR.alpha * y[:,1,:] + JR.gamma * y[:,7,:]) - (JR.alpha * y[:,2,:] + JR.gamma * y[:,8,:])
        
        # Filter the EEG signals using a Bessel filter
        EEG_filt = signal.filtfilt(a0, b0, EEG_signals, axis=0)
        
        # Compute the functional connectivity matrix
        FC_sim = np.corrcoef(EEG_filt.T)
        
        # Evaluate the similarity between simulated and empirical FC using SSIM
        GoF_Gs[gx, sx] = ssim(FC_emp, FC_sim, data_range=1)

GoF_Gs = np.nanmean(GoF_Gs, axis=1)  # Compute the mean goodness-of-fit for each G
G_best = G_vec[np.argmax(GoF_Gs)]  # Select the best G value based on maximum SSIM
JR.K = G_best  # Update model with best global coupling

# Step 2: Fit excitatory/inhibitory balance (rE)
results = np.zeros((seeds, iters_rE))  # Store SSIM results per seed and iteration
results_re = np.zeros((iters_rE, nnodes))  # Store regional firing rate optimization

for j in range(iters_rE):
    FCs_temp = np.zeros((nnodes, nnodes, seeds))  # Store FC matrices across seeds
    
    for sx in range(seeds):
        JR.seed = sx  # Set seed for each simulation
        y, time_vector = JR.Sim(verbose=False)
        
        # Compute EEG signals and filter them
        EEG_signals = (JR.alpha * y[:,1,:] + JR.gamma * y[:,7,:]) - (JR.alpha * y[:,2,:] + JR.gamma * y[:,8,:])
        EEG_filt = signal.filtfilt(a0, b0, EEG_signals, axis=0)
        
        # Compute functional connectivity matrix
        FC = np.corrcoef(EEG_filt.T)
        FCs_temp[:, :, sx] = FC.copy()
        results[sx, j] = ssim(FC_emp, FC, data_range=2)  # Evaluate similarity
    
    FC_avg = np.mean(FCs_temp, axis=2)  # Compute mean FC across seeds
    nodal_diff = np.mean(FC_emp - FC_avg, axis=0)  # Compute nodal strength difference
    results_re[j, :] = JR.target  # Store current target firing rates
    
    # Update firing rates based on nodal differences
    JR.target += nodal_diff * delta_rE
    JR.target[JR.target < 0] = 0  # Ensure firing rates are non-negative

rE_best = results_re[np.argmax(np.nanmean(results, axis=0)), :]  # Get optimal firing rates

# Step 3: Fit structural connectivity (SC)
results = np.zeros((seeds, iters_SC))  # Store SSIM results per seed and iteration
results_sc = np.zeros((iters_SC, nnodes, nnodes))  # Store updated SC matrices

for j in range(iters_SC):
    FCs_temp = np.zeros((nnodes, nnodes, seeds))  # Store FC matrices across seeds
    
    for sx in range(seeds):
        JR.seed = sx  # Set seed for each simulation
        y, time_vector = JR.Sim(verbose=False)
        
        # Compute EEG signals and filter them
        EEG_signals = (JR.alpha * y[:,1,:] + JR.gamma * y[:,7,:]) - (JR.alpha * y[:,2,:] + JR.gamma * y[:,8,:])
        EEG_filt = signal.filtfilt(a0, b0, EEG_signals, axis=0)
        
        # Compute functional connectivity matrix
        FC = np.corrcoef(EEG_filt.T)
        FCs_temp[:, :, sx] = FC.copy()
        results[sx, j] = ssim(FC_emp, FC, data_range=2)
    
    FC_avg = np.mean(FCs_temp, axis=2)  # Compute mean FC across seeds
    FC_diff = FC_emp - FC_avg  # Compute FC difference
    results_sc[j, :, :] = JR.M  # Store current SC matrix
    
    # Update structural connectivity based on FC difference
    JR.M += FC_diff * delta_SC
    JR.M[JR.M < 0] = 0  # Ensure SC values are non-negative
    JR.M[JR.M > 1] = 1  # Cap SC values at 1

    ###Some suggestions
    ##JR.M = JR.M + FC_diff * (SC_CN > 0.07) * delta_SC #replace 0.07 for any threshold you want.
    ### This is just for ensuring that you're changing only pre-existing connections in your SC matrix

SC_best = get_uptri(results_sc[np.argmax(np.nanmean(results, axis=0)), :])  # Get best SC

# Store fitting results
my_fit = np.array([GoF_Gs.max(), np.nanmax(results_re), np.nanmax(results_sc), G_best])
my_fit = np.append(my_fit, rE_best)
my_fit = np.append(my_fit, SC_best)

end = time.time()

print("Fitting completed. Time elapsed = %.2f sec" % (end - init))

