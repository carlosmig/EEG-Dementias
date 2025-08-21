# -*- coding: utf-8 -*-
"""
Single-run showcase:
- Condition 0 (awake): K=1.44, target=2.56 Hz
- One seed
Plots:
  1) BOLD FC (0.01–0.08 Hz)
  2) EEG PSD (Welch, normalized)
  3) 1-second EEG trace (one node)
  4) BOLD signals (several nodes)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
import JansenRitModelMulti as JR
import BOLDModel as BD

warnings.filterwarnings("ignore")

# ----------------------------
# Load/connectome & base model setup
# ----------------------------
SC = np.load('NVGPs_optimized.npy')     # your structural connectivity
np.fill_diagonal(SC, 0)

#SC matrix
JR.M = SC
JR.nnodes = SC.shape[0]
nnodes = JR.nnodes

# Sampler / timing
JR.dt = 1e-3          # 1 ms internal step
JR.downsamp = 10      # outputs at 100 Hz
JR.teq = 10           # transient (s)
JR.tmax = 660         # total time (s) (incl. transient)
stab_time = 60        # discard this much at start for BOLD
N_stab = int(stab_time / (JR.dt * JR.downsamp))

# Model toggles / parameters
JR.plasticity_on = 1
JR.tau_C4 = 2
JR.alpha = 0.5 * np.ones(nnodes)
JR.gamma = 1 - JR.alpha
JR.sigma = 1.0
JR.p = 220 * np.ones(nnodes)
JR.norm = 1.0

# ----------------------------
# Condition 0 (awake) parameters + seed
# ----------------------------
JR.K = 1.44
JR.target = 2.56 * np.ones(nnodes)
JR.seed = 0
JR.update()

# ----------------------------
# Simulate JR -> EEG
# ----------------------------
y, t = JR.Sim(verbose=False)
EEG = (JR.alpha * y[:, 1, :] + JR.gamma * y[:, 7, :]) \
    - (JR.alpha * y[:, 2, :] + JR.gamma * y[:, 8, :])

# ----------------------------
# EEG PSD (Welch, averaged across nodes, normalized)
# ----------------------------
fs_eeg = int(1000 // JR.downsamp)   # 100 Hz given downsamp=10
nperseg  = 4000 // JR.downsamp      # like your pipeline (400 samples)
noverlap = 2000 // JR.downsamp      # 200 samples
freqs, PSDs = signal.welch(
    EEG, fs=fs_eeg, window='hann',
    nperseg=nperseg, noverlap=noverlap,
    axis=0, scaling='density'
)
PSD_mean = np.mean(PSDs, axis=1)
PSD_mean = PSD_mean / PSD_mean.max()

# ----------------------------
# Convert to BOLD and compute FC (0.01–0.08 Hz)
# ----------------------------
rates = JR.s(EEG, 0.56)                     # firing rates
BOLD = BD.Sim(rates, nnodes, JR.dt * JR.downsamp)

# discard stabilization, then downsample to ~2.08 s TR (like your code)
BOLD = BOLD[N_stab:, :]
BOLD_dt = 2.08
BOLD = signal.decimate(BOLD, n=3, q=int(BOLD_dt / (JR.dt * JR.downsamp)), axis=0)

# band-pass 0.01–0.08 Hz (BOLD space)
a_bp, b_bp = signal.bessel(3, [2 * BOLD_dt * 0.01, 2 * BOLD_dt * 0.08], btype='bandpass')
BOLD_filt = signal.filtfilt(a_bp, b_bp, BOLD, axis=0)

FC_sim = np.corrcoef(BOLD_filt.T)          # (nodes x nodes)

#%%

# ----------------------------
# Build plots
# ----------------------------
plt.rcParams.update({"font.size": 15})  # bump fontsize

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2], hspace=0.28, wspace=0.25)

# (1) BOLD FC
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(np.clip(FC_sim, 0, 1), vmin=0, vmax=1, cmap="rainbow")
ax1.set_title("BOLD FC (0.01–0.08 Hz)")
ax1.set_xticks([]); ax1.set_yticks([])
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label("Correlation")

# (2) EEG PSD (normalized)
ax2 = fig.add_subplot(gs[0, 1])
ax2.loglog(freqs, PSD_mean)
ax2.set_title("EEG PSD (Welch, normalized)")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Normalized Power")
ax2.grid(True, which='both', ls=':')

# (3) 1-second EEG trace (one node)
ax3 = fig.add_subplot(gs[1, 0])
one_sec = fs_eeg  # 1 second at 100 Hz
node_idx = 10
ax3.plot(np.arange(one_sec)/fs_eeg, EEG[:one_sec, node_idx])
ax3.set_title(f"EEG trace (1 s, node {node_idx})")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Amplitude")

# (4) BOLD signals (a few nodes)
ax4 = fig.add_subplot(gs[1, 1])
T_bold = BOLD_filt.shape[0]
t_bold = np.arange(T_bold) * BOLD_dt
nodes_to_plot = np.arange(0, min(5, nnodes))  # first up to 5 nodes
offset = 3.0 * np.std(BOLD_filt)              # vertical offset for clarity
for k, n in enumerate(nodes_to_plot):
    ax4.plot(t_bold, BOLD_filt[:, n] + k*offset, label=f"Node {n}")
ax4.set_title("BOLD signals (band-passed 0.01–0.08 Hz)")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Amplitude (offset per node)")

plt.suptitle("Single-run showcase (Condition 0: Awake)", y=0.98)
plt.show()

# Reset font size to default
plt.rcParams.update({"font.size": 10})


