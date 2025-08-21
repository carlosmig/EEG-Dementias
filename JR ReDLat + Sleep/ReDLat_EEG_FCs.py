# -*- coding: utf-8 -*-
"""
Run 5 seeds and save averaged FC matrices for EEG model
  - NO ISP  (plasticity_off)
  - WITH ISP (plasticity_on)

It reproduces your filtering/FC pipeline and averages across seeds.

Author: Carlos Coronel
"""

import numpy as np
from scipy import signal
from skimage.metrics import structural_similarity as ssim
import JansenRitModelMulti as JR
import matplotlib.pyplot as plt

# ----------------------------
# Common data / parameters
# ----------------------------
# Healthy controls SC matrix
SC_CN = np.load('SC_CN_ReDLat.npy')

#Empirical FCs
FC_emp = np.load('FCs_CN_Dementia.npy')
FC_emp = np.mean(FC_emp,3)[:,:,0:4] #averaged FCs across subjects, from delta tobeta

JR.M = SC_CN
JR.nnodes = len(JR.M)
nnodes = JR.nnodes

# Simulation parameters
JR.dt = 1e-3
JR.teq = 20 #lower simulation time as example
JR.tmax = 120 #lower simulation time as example
JR.downsamp = 10
JR.tau_C4 = 2
JR.p = 220 * np.ones(nnodes)
JR.norm = 1 #This just scale the global coupling
JR.sigma = 1

# Bands (Hz)
bands = [(0.5, 4), (4, 8), (8, 13), (13, 30)]



# ----------------------------
# Helper: run condition & average FCs across seeds
# ----------------------------
def average_FC_over_seeds(seeds, K, alpha_scalar, plasticity_on, target_hz=2.5):
    """
    Runs the model for each seed and returns/saves the average FC per band.
    """
    n_bands = len(bands)
    ssims = np.zeros((len(seeds),n_bands))
    FCs = np.zeros((nnodes,nnodes,len(seeds),n_bands))

    for seed in seeds:
        # Set condition-specific params
        JR.K = K
        JR.alpha = alpha_scalar * np.ones(nnodes)
        JR.gamma = 1.0 - JR.alpha
        JR.sigma = 1.0
        JR.plasticity_on = int(plasticity_on)
        JR.target = target_hz * np.ones(nnodes)
        JR.seed = int(seed)
        
        JR.update()

        # Simulate
        y, t = JR.Sim(verbose=False)
        EEG = (JR.alpha * y[:, 1, :] + JR.gamma * y[:, 7, :]) - (JR.alpha * y[:, 2, :] + JR.gamma * y[:, 8, :])

        # Envelope-FC per band
        for bi, (fmin, fmax) in enumerate(bands):
            # Band-pass (digital, normalized to Nyquist)
            Wn = 2 * JR.dt * JR.downsamp * np.array([fmin, fmax])
            a0, b0 = signal.bessel(3, Wn, btype='bandpass')
            Vfilt = signal.filtfilt(a0, b0, EEG, axis=0)[3000:-3000] #removing filter artifacts

            # Envelope -> lowpass at 0.5 Hz
            envelopes = np.abs(signal.hilbert(Vfilt, axis=0))
            Wlp = 2 * JR.dt * JR.downsamp * 0.5
            a1, b1 = signal.bessel(3, [Wlp], btype='low')
            Efilt = signal.filtfilt(a1, b1, envelopes, axis=0)
            
            #FCs by band
            FC_sim = np.corrcoef(Efilt.T)
            FCs[:,:,seed,bi] = FC_sim.copy()
            
            #goodness of fit 
            ssims[seed,bi] = ssim(FC_emp[:,:,bi], FC_sim, data_range = 2)
    
    return FCs, ssims

# ----------------------------
# Run 5 seeds per condition
# ----------------------------
seeds5 = np.arange(5)

# (A) NO ISP (plasticity_off), your network exploration example
FC_avg_noISP, ssim_no_ISP = average_FC_over_seeds(
    seeds=seeds5,
    K=0.425,
    alpha_scalar=1.0,
    plasticity_on=False,
    target_hz=2.5,
)

# (B) WITH ISP (plasticity_on), your commented exploration values
FC_avg_ISP, ssim_ISP = average_FC_over_seeds(
    seeds=seeds5,
    K=0.675,
    alpha_scalar=0.575,
    plasticity_on=True,
    target_hz=2.5,
)

#%%

#PLOTTING

#this is for re-ordering ROIs in FC matrices
left = np.arange(0,82,2)
right = np.arange(1,82,2)[::-1]
symm_order = np.append(left, right)

# --- Compute averages---
FC_noISP_mean = np.mean(FC_avg_noISP, axis=2)[symm_order,:,:][:,symm_order,:]  # (nnodes, nnodes, n_bands)
FC_ISP_mean   = np.mean(FC_avg_ISP,   axis=2)[symm_order,:,:][:,symm_order,:]  # (nnodes, nnodes, n_bands)


ssim_noISP_vals = ssim_no_ISP  # (n_seeds, n_bands)
ssim_ISP_vals   = ssim_ISP     # (n_seeds, n_bands)

band_names = [r'$\delta$ (0.5–4 Hz)', r'$\theta$ (4–8 Hz)', 
              r'$\alpha$ (8–13 Hz)', r'$\beta$ (13–30 Hz)']

# --- Plot params ---
cmap = "rainbow"
vmin, vmax = 0, 1

plt.rcParams.update({"font.size": 15})
fig, axes = plt.subplots(4, 4, figsize=(16, 12), constrained_layout=True)

# --- Row 1: SSIM violin plots ---
for bi in range(4):
    ax = axes[0, bi]
    parts = ax.violinplot([ssim_noISP_vals[:, bi], ssim_ISP_vals[:, bi]], 
                          positions=[0, 1], showmeans=True)
    ax.set_xticks([0, 1], ["Single", "Multi"])
    ax.set_ylim(0, 1.05)
    ax.set_title(band_names[bi])  # titles only on first row
    if bi == 0:
        ax.set_ylabel("SSIM")

# --- Row 2: Avg FC (No ISP) ---
for bi in range(4):
    ax = axes[1, bi]
    im = ax.imshow(FC_noISP_mean[:, :, bi], vmin=vmin, vmax=vmax, cmap=cmap)
    if bi == 0:
        ax.set_ylabel("Single")
    ax.set_xticks([]); ax.set_yticks([])
    if bi == 3:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# --- Row 3: Avg FC (ISP) ---
for bi in range(4):
    ax = axes[2, bi]
    im = ax.imshow(FC_ISP_mean[:, :, bi], vmin=vmin, vmax=vmax, cmap=cmap)
    if bi == 0:
        ax.set_ylabel("Multi")
    ax.set_xticks([]); ax.set_yticks([])
    if bi == 3:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# --- Row 4: Empirical ---
for bi in range(4):
    ax = axes[3, bi]
    im = ax.imshow(FC_emp[:, :, bi][symm_order,:][:,symm_order], vmin=vmin, vmax=vmax, cmap=cmap)
    if bi == 0:
        ax.set_ylabel("Empirical")
    ax.set_xticks([]); ax.set_yticks([])
    if bi == 3:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()

plt.suptitle("EEG Envelope-FC: SSIM and Average FCs (No ISP vs ISP) vs Empirical", fontsize=16, y=1.02)
plt.show()
plt.rcParams.update({"font.size": 10})  # reset to default
