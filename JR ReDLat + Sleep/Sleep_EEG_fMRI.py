# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 16:20:24 2025

Non-parallel sleep FC pipeline (BOLD from EEG rates)
- Runs two conditions (as in your script)
- Averages FCs across seeds
- Plots mean FCs and their difference

Requires:
  - JansenRitModelMulti as JR
  - BOLDModel as BD
  - SCmatrices88healthy.mat

author: Carlos Coronel

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import JansenRitModelMulti as JR
import BOLDModel as BD

warnings.filterwarnings("ignore")

# ----------------------------
# Helpers / setup
# ----------------------------
def fill_antidiagonal(matrix, value):
    n = matrix.shape[0]
    out = matrix.copy()
    for i in range(n):
        out[i, n - 1 - i] = value
    return out


# Load SC and build network (match your code)
SC = np.load('NVGPs_optimized.npy')
np.fill_diagonal(SC, 0)


JR.M = SC
JR.nnodes = SC.shape[0]
nnodes = JR.nnodes
JR.norm = 1

# Base model params (as in your script)
JR.dt = 1e-3
JR.teq = 10
stab_time = 60
JR.tmax = 660
JR.downsamp = 10
N_stab = int(stab_time / JR.dt / JR.downsamp)
JR.plasticity_on = 1
JR.tau_C4 = 2
JR.alpha = 0.5 * np.ones(nnodes)
JR.gamma = 1 - JR.alpha
JR.sigma = 1
JR.p = 220 * np.ones(nnodes)
JR.norm = 1
JR.update()

# BOLD filtering (0.01–0.08 Hz) after decimation step
BOLD_dt = 2.08
bold_bp = (0.01, 0.08)
a_bp, b_bp = signal.bessel(3, [2 * BOLD_dt * bold_bp[0], 2 * BOLD_dt * bold_bp[1]], btype='bandpass')

#this is for re-ordering ROIs in FC matrices
left = np.arange(0,82,2)
right = np.arange(1,82,2)[::-1]
symm_order = np.append(left, right)

# Emp FCs
FCs_zzz = np.load('FCs_sleep_116_ROIs.npy')
FC_AWAKE = np.mean(FCs_zzz[0:90,0:90,0,:],-1)[symm_order,:][:,symm_order]
FC_N3 = np.mean(FCs_zzz[0:90,0:90,3,:],-1)[symm_order,:][:,symm_order]

# ----------------------------
# Runner for one condition across seeds (now returns FCs + PSDs)
# ----------------------------
def run_condition_get_FCs_and_PSDs(condition_id, seeds):
    """
    condition_id: 0 or 1
      0 (awake)   -> K=1.44, target=2.56 Hz
      1 (N3 sleep)-> K=0.52, target=2.10 Hz
    Returns:
      FC_list : list of (n_nodes x n_nodes) FC matrices (one per seed)
      freqs   : frequency vector from Welch
      PSD_mat : array (n_freqs x n_seeds), mean PSD across nodes per seed
    """
    if condition_id == 0:
        JR.K = 1.44
        JR.target = 2.56 * np.ones(nnodes)
    elif condition_id == 1:
        JR.K = 0.52
        JR.target = 2.10 * np.ones(nnodes)
    else:
        raise ValueError("condition_id must be 0 or 1.")

    FC_list = []
    freqs_out = None
    psd_per_seed = []

    # Welch params consistent with your earlier code
    fs_eeg   = 1000 // JR.downsamp
    nperseg  = 4000 // JR.downsamp
    noverlap = 2000 // JR.downsamp

    for sd in seeds:
        JR.seed = int(sd)
        JR.update()

        # Simulate JR -> EEG
        y, t = JR.Sim(verbose=False)
        EEG = (JR.alpha * y[:, 1, :] + JR.gamma * y[:, 7, :]) - (JR.alpha * y[:, 2, :] + JR.gamma * y[:, 8, :])

        # --- Welch PSD (average across nodes) ---
        # EEG shape expected: (time, nodes). Welch returns (freqs, PSD) with PSD shape (n_freqs, nodes)
        freqs, PSDs = signal.welch(
            EEG, fs=fs_eeg, window='hann',
            nperseg=nperseg, noverlap=noverlap,
            axis=0, scaling='density'
        )
        if freqs_out is None:
            freqs_out = freqs
        PSD_mean_nodes = np.mean(PSDs, axis=1)   # (n_freqs,)
        psd_per_seed.append(PSD_mean_nodes)

        # Convert to BOLD
        rates = JR.s(EEG, 0.56)              # firing rate nonlinearity
        BOLD = BD.Sim(rates, nnodes, JR.dt * JR.downsamp)

        # Stabilization and downsample to ~2.08 s
        BOLD = BOLD[N_stab:, :]
        BOLD = signal.decimate(BOLD, n=3, q=int(BOLD_dt / (JR.dt * JR.downsamp)), axis=0)

        # Band-pass 0.01–0.08 Hz
        BOLD_filt = signal.filtfilt(a_bp, b_bp, BOLD, axis=0)

        # FC as Pearson correlation
        FC_sim = np.corrcoef(BOLD_filt.T)
        FC_list.append(FC_sim)
        
        if sd == 0:
            EEG_to_save = EEG.copy()

    PSD_mat = np.column_stack(psd_per_seed)  # (n_freqs x n_seeds)
    return FC_list, freqs_out, PSD_mat, EEG_to_save

#%%

# ----------------------------
# Run small set of seeds and average (FCs + PSDs)
# ----------------------------
seeds = np.arange(5)  # run 5 seeds

# Condition 0 = awake
FCs_cond0, freqs0, PSDs_cond0, EEG_cond0 = run_condition_get_FCs_and_PSDs(condition_id=0, seeds=seeds)

# Condition 1 = N3 sleep
FCs_cond1, freqs1, PSDs_cond1, EEG_cond1 = run_condition_get_FCs_and_PSDs(condition_id=1, seeds=seeds)

# --- FCs ---
FC_mean_0 = np.mean(np.stack(FCs_cond0, axis=2), axis=2)  # (n_nodes, n_nodes)
FC_mean_1 = np.mean(np.stack(FCs_cond1, axis=2), axis=2)  # (n_nodes, n_nodes)

# --- PSDs ---
PSD_mean_0 = np.mean(PSDs_cond0, axis=1)  # (n_freqs,)
PSD_mean_0 = PSD_mean_0 / np.max(PSD_mean_0)
PSD_mean_1 = np.mean(PSDs_cond1, axis=1)  # (n_freqs,)
PSD_mean_1 = PSD_mean_1 / np.max(PSD_mean_1)


#%%

# ----------------------------
# Plot: Empirical vs Simulated FCs + EEG traces + PSDs
# ----------------------------
plt.rcParams.update({"font.size": 15})  # bump fontsize

fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)

cmap_fc = "rainbow"
vmin, vmax = 0, 1

# -------- Row 1: FCs --------
# Empirical FC awake
im0 = axes[0,0].imshow(np.clip(FC_AWAKE, vmin, vmax), vmin=vmin, vmax=vmax, cmap=cmap_fc)
axes[0,0].set_title("Empirical FC (Awake)")
axes[0,0].set_xticks([]); axes[0,0].set_yticks([])
fig.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04)

# Simulated FC condition 0 (awake)
im1 = axes[0,1].imshow(np.clip(FC_mean_0, vmin, vmax), vmin=vmin, vmax=vmax, cmap=cmap_fc)
axes[0,1].set_title("Simulated FC (Awake, cond 0)")
axes[0,1].set_xticks([]); axes[0,1].set_yticks([])
fig.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04)

# Empirical FC N3
im2 = axes[0,2].imshow(np.clip(FC_N3, vmin, vmax), vmin=vmin, vmax=vmax, cmap=cmap_fc)
axes[0,2].set_title("Empirical FC (N3)")
axes[0,2].set_xticks([]); axes[0,2].set_yticks([])
fig.colorbar(im2, ax=axes[0,2], fraction=0.046, pad=0.04)

# Simulated FC condition 1 (N3)
im3 = axes[0,3].imshow(np.clip(FC_mean_1, vmin, vmax), vmin=vmin, vmax=vmax, cmap=cmap_fc)
axes[0,3].set_title("Simulated FC (N3, cond 1)")
axes[0,3].set_xticks([]); axes[0,3].set_yticks([])
fig.colorbar(im3, ax=axes[0,3], fraction=0.046, pad=0.04)

# -------- Row 2: EEG + PSD --------
# Take 1 s segment from EEG (fs = 1000//downsamp Hz)
fs_eeg = int(1000 // JR.downsamp)
t_axis = np.arange(fs_eeg) / fs_eeg  # 1 second

axes[1,0].plot(t_axis, EEG_cond0[:fs_eeg, 0], color="blue")
axes[1,0].set_title("EEG trace (Awake, cond 0)")
axes[1,0].set_xlabel("Time (s)")
axes[1,0].set_ylabel("Amplitude")

axes[1,1].plot(t_axis, EEG_cond1[:fs_eeg, 0], color="red")
axes[1,1].set_title("EEG trace (N3, cond 1)")
axes[1,1].set_xlabel("Time (s)")
axes[1,1].set_ylabel("Amplitude")

# PSDs (log–log scale)
axes[1,2].loglog(freqs0, PSD_mean_0, label="Sim Awake", color="blue")
axes[1,2].loglog(freqs1, PSD_mean_1, label="Sim N3", color="red")
axes[1,2].set_xlabel("Frequency (Hz)")
axes[1,2].set_ylabel("Normalized Power")
axes[1,2].legend()
axes[1,2].set_title("Simulated PSDs")

# Empty last subplot
axes[1,3].axis("off")

# Reset fontsize back to default
plt.rcParams.update({"font.size": 10})



