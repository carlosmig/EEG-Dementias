# -*- coding: utf-8 -*-
"""
Figure 3 – Single-node dynamics
A) Power spectrum as a function of r_alpha for a single-node.
B) Noise-free single-node simulations (r_alpha = 0.5):
   target firing rate (ρ) vs. time-averaged simulated firing rate
C) Same setup:
   target firing rate (ρ) vs. time-averaged feedback inhibition (C4)

Notes
- The Jansen–Rit multi-population model is configured for one node (JR.M = [[0]]).
- Plasticity is enabled (JR.plasticity_on = 1).
- We sweep target rates (rE_vec) and external input (p_vec), collect steady-state outputs.

Author: Carlos Coronel
"""

import warnings
warnings.filterwarnings("ignore")  # silence non-critical runtime warnings
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import JansenRitModelMulti as JR

# ---------------------------------------------------------------------
# 1) MODEL & SIMULATION SETUP
# ---------------------------------------------------------------------
# Single-node connectivity
JR.M = np.array([[0]])
JR.nnodes = len(JR.M)
nnodes = JR.nnodes

# Node (synaptic) parameters (kept as in your code)
# These are used internally by JR.update() / JR.Sim() via the JR module globals
JR.a1 = 120   # EPSP inverse time (alpha), 1/s
JR.b1 = 60    # IPSP inverse time (alpha), 1/s
JR.a2 = 660   # EPSP inverse time (gamma), 1/s
JR.b2 = 330   # IPSP inverse time (gamma), 1/s
JR.A1 = 32.5 * JR.a1 / 1000  # EPSP amplitude (alpha)
JR.B1 = 440  * JR.b1 / 1000  # IPSP amplitude (alpha)
JR.A2 = 32.5 * JR.a2 / 1000  # EPSP amplitude (gamma)
JR.B2 = 440  * JR.b2 / 1000  # IPSP amplitude (gamma)

# Simulation parameters
JR.dt      = 1e-3   # integration step (s)
JR.teq     = 10     # equilibration time (s)
JR.tmax    = 15     # total time (s)
JR.downsamp= 5      # downsample factor

# Plasticity flag and dependent updates
JR.update()  # apply changes

#%%
# ---------------------------------------------------------------------
# 2) POWER SPECTRUM
# ---------------------------------------------------------------------

seeds      = np.arange(0, 5, 1) #more seeds will require more time
alpha_vec  = np.linspace(0, 1, 41)
target_vec = np.array([0])           # no target for plasticity_off

# Storage
freqs_out = None
PSD_alpha_norm = None  # shape: (len(alpha_vec), n_freqs)

# Fixed parameters per sweep
JR.C4_min = 0
JR.p = 220 * np.ones(JR.nnodes)

# Plasticity flag and dependent updates
JR.plasticity_on  = 0
JR.update()  # apply changes

# Welch params (derived from your snippet)
fs = 1000 // JR.downsamp
nperseg = 4000 // JR.downsamp
noverlap = 2000 // JR.downsamp

# --- Loop over alpha and seeds ---
all_psd = []  # temporary collector to size arrays after first run

for ia, a in enumerate(alpha_vec):
    JR.alpha = a * np.ones(JR.nnodes)
    JR.gamma = 1 - JR.alpha

    psd_seeds = []

    for seed in seeds:
        JR.seed = int(seed)

        y, t = JR.Sim(verbose=False)
        EEG = (JR.alpha * y[:, 1, :] + JR.gamma * y[:, 7, :]) - (JR.alpha * y[:, 2, :] + JR.gamma * y[:, 8, :])

        freqs, PSDs = signal.welch(
            EEG, fs=fs, window='hann',
            nperseg=nperseg, noverlap=noverlap,
            axis=0, scaling='density'
        )
        PSD = np.mean(PSDs, axis=1).ravel()  # (n_freqs,)

        psd_seeds.append(PSD)

    # Average PSD across seeds for this alpha
    psd_mean = np.mean(np.vstack(psd_seeds), axis=0)  # (n_freqs,)

    # Normalize by max (avoid divide-by-zero)
    max_val = np.max(psd_mean)
    if max_val > 0:
        psd_mean_norm = psd_mean / max_val
    else:
        psd_mean_norm = psd_mean  # all zeros

    # Allocate on first pass
    if PSD_alpha_norm is None:
        PSD_alpha_norm = np.zeros((len(alpha_vec), len(psd_mean_norm)))
        freqs_out = freqs

    PSD_alpha_norm[ia, :] = psd_mean_norm
    
    print(ia)

#%%

# ---------------------------------------------------------------------
# 3) SWEEP SETUP
# ---------------------------------------------------------------------
# Target excitatory rates (Hz) and external drive p (a.u.)
rE_vec = np.linspace(2, 5, 31)         # 31 values from 2 to 5 Hz
p_vec  = np.array([125, 250, 500, 1000])

# Storage arrays: shape [len(p_vec), len(rE_vec)]
sim_rE  = np.zeros((len(p_vec), len(rE_vec)))  # simulated mean firing rate
real_rE = np.zeros((len(p_vec), len(rE_vec)))  # target (ground-truth) rate
sim_C4  = np.zeros((len(p_vec), len(rE_vec)))  # time-avg feedback inhibition

# Fixed mixture (r_alpha = 0.5) used in the single-node readout
JR.C4_min = 0
JR.alpha  = 0.5 * np.ones(JR.nnodes)
JR.gamma  = 1 - JR.alpha

# Default external input (overwritten inside the loop)
JR.p = 220 * np.ones(nnodes)

#plasticity activated
JR.plasticity_on = 1
JR.update()  # apply changes


#%%

# ---------------------------------------------------------------------
# 4) MAIN SWEEP: run simulations and collect steady-state metrics
# ---------------------------------------------------------------------
for i, target_rate in enumerate(rE_vec):
    for j, p_in in enumerate(p_vec):

        # Set target and input
        JR.target = target_rate * np.ones(JR.nnodes)
        JR.p      = p_in * np.ones(JR.nnodes)

        # Run simulation
        y, t = JR.Sim(verbose=False)

        # EEG-like signal: weighted pyramidal output (alpha/gamma subpopulations)
        eeg = (JR.alpha * y[:, 1, :] + JR.gamma * y[:, 7, :]) \
            - (JR.alpha * y[:, 2, :] + JR.gamma * y[:, 8, :])

        # Time-averaged inhibitory feedback (last state assumed to store C4)
        sim_C4[j, i] = np.mean(y[:, -1, :])

        # Convert “eeg” to firing-rate proxy via sigmoid s(v, v0=0.56),
        # then average over time (and nodes)
        rE = np.mean(JR.s(eeg, 0.56))

        # Store
        sim_rE[j, i]  = rE
        real_rE[j, i] = target_rate

    print(f"Done rE index {i+1}/{len(rE_vec)} (target={target_rate:.2f} Hz)")


#%%

plt.rcParams.update({'font.size': 16})

# ---------------------------------------------------------------------
# 5) PLOTS
# ---------------------------------------------------------------------

# A) Plot PSD map
plt.figure(1, figsize=(6, 5.25))

# use imshow for alpha (x) vs frequency (y)
plt.imshow(
    PSD_alpha_norm.T, 
    aspect='auto', 
    origin='lower',
    extent=[alpha_vec.min(), alpha_vec.max(), freqs_out.min(), freqs_out.max()],
    cmap='jet'
)

plt.colorbar(label="Normalized PSD", shrink=0.9)
plt.xlabel(r"Neural proportion $r^{\alpha}$")
plt.ylabel("Frequency (Hz)")
plt.title("Single node frequency")

plt.ylim(0, 100)   # match your example (cut at 100 Hz)
plt.tight_layout()
plt.show()

# B) Target vs. simulated firing rates (convergence)
plt.figure(2, figsize=(6, 5.25))
plt.clf()
for j, p_in in enumerate(p_vec):
    plt.scatter(real_rE[j, :], sim_rE[j, :], label=f"p = {p_in}")

# Identity line
lims = [min(real_rE.min(), sim_rE.min()),
        max(real_rE.max(), sim_rE.max())]
plt.plot(lims, lims, 'k--', lw=2, label="Identity")

plt.xlabel("Target firing rate (Hz)")
plt.ylabel("Simulated firing rate (Hz)")
plt.title("Firing rate convergence")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

# C) Target rate vs. steady-state feedback inhibition (C4)
plt.figure(3, figsize=(6, 5.25))
plt.clf()
for j, p_in in enumerate(p_vec):
    plt.scatter(real_rE[j, :], sim_C4[j, :], label=f"p = {p_in}")

plt.xlabel("Target firing rate (Hz)")
plt.ylabel(r"Simulated C$_4$")
plt.title("Steady state feedback inhibition")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.rcdefaults()

