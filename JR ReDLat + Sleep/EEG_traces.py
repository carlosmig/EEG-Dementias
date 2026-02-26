import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import JansenRitModelMulti as JR

# Load SC and configure model
SC = np.load('SC_CN_ReDLat.npy')

JR.M = SC
JR.nnodes = len(JR.M)
nnodes = JR.nnodes

# Simulation parameters
JR.dt = 1e-3
JR.teq = 20
JR.tmax = 120
JR.downsamp = 10

# Fixed model params
JR.norm = 1
JR.sigma = 1.0
JR.K = 0.675
JR.norm = 1
JR.p = 220 * np.ones(nnodes)
JR.alpha = 0.575 * np.ones(nnodes)
JR.gamma = 1.0 - JR.alpha
JR.target = 2.5 * np.ones(nnodes)

JR.plasticity_on = 1   
JR.seed = 0

JR.update()

# Run simulation
y, t = JR.Sim(verbose=False)

# EEG-like signals
EEG = (JR.alpha * y[:, 1, :] + JR.gamma * y[:, 7, :]) - (JR.alpha * y[:, 2, :] + JR.gamma * y[:, 8, :])
x = EEG.mean(axis=1)
fs = 1.0 / (JR.dt * JR.downsamp)
mask = t >= JR.teq
t_ss = t[mask]
x_ss = x[mask]
x_ss = x_ss - np.mean(x_ss)

# Plotting window
snippet_sec = 10.0
t0 = t_ss[0]
snip_mask = t_ss <= (t0 + snippet_sec)
t_plot = t_ss[snip_mask] - t0
x_plot = x_ss[snip_mask]


# Band filtering
bands = [
    ("δ", 0.5, 4),
    ("θ", 4, 8),
    ("α", 8, 13),
    ("β", 13, 30),
    ("γ", 30, 40),
]

filtered = []
for name, fmin, fmax in bands:
    Wn = np.array([fmin, fmax]) / (fs / 2.0)  # normalized to Nyquist
    b, a = signal.bessel(3, Wn, btype='bandpass', analog=False, norm='phase')
    xf = signal.filtfilt(b, a, x_ss)
    xf = xf - np.mean(xf)
    filtered.append(xf)
    
# Alpha amplitude fluctuations (Hilbert envelope + lowpass)
alpha_filt = filtered[2]

# Envelope (amplitude)
alpha_env = np.abs(signal.hilbert(alpha_filt))

# Low-pass the envelope to highlight slow amplitude fluctuations
env_lp_hz = 0.5
Wlp = env_lp_hz / (fs / 2.0)
b_lp, a_lp = signal.bessel(3, Wlp, btype='low', analog=False, norm='phase')
alpha_env_lp = signal.filtfilt(b_lp, a_lp, alpha_env)

# Center envelopes
alpha_env_c = alpha_env - np.mean(alpha_env)
alpha_env_lp_c = alpha_env_lp - np.mean(alpha_env_lp)

# Snippet for plotting
alpha_env_plot = alpha_env_c[snip_mask]
alpha_env_lp_plot = alpha_env_lp_c[snip_mask]

# Optional quantification
alpha_env_cv = np.std(alpha_env_lp) / (np.mean(alpha_env) + 1e-12)
print(f"Alpha envelope CV (slow, lp<{env_lp_hz} Hz): {alpha_env_cv:.4f}")

# For plotting snippet
filtered_plot = [xf[snip_mask] for xf in filtered]

# Same y-limits across band subplots
ymax = max(np.max(np.abs(xf)) for xf in filtered_plot)
ymax = 1.05 * ymax if ymax > 0 else 1.0

# Relative band power (Welch PSD + band integration)
nperseg = int(fs * 4)   # 4-second segments
noverlap = int(fs * 2)  # 50% overlap

freqs, Pxx = signal.welch(
    x_ss, fs=fs, window="hann",
    nperseg=nperseg, noverlap=noverlap,
    scaling="density"
)

band_power = []
for name, fmin, fmax in bands:
    m = (freqs >= fmin) & (freqs < fmax)
    bp = np.trapz(Pxx[m], freqs[m])
    band_power.append(bp)

band_power = np.array(band_power)
rel_power = band_power / (band_power.sum() + 1e-12)

#%%

# Plot: 5 band signals + alpha-envelope
plt.rcParams.update({"font.size": 12})
plt.rcParams["svg.fonttype"] = "none"

nrows = len(bands) + 1
fig, axes = plt.subplots(nrows, 1, figsize=(10, 12), sharex=True, constrained_layout=True)

# time-series panels y-limit
ymax_ts = max(
    [np.max(np.abs(xf)) for xf in filtered_plot] +
    [np.max(np.abs(alpha_env_lp_plot))]
)
ymax_ts = 1.05 * ymax_ts if ymax_ts > 0 else 1.0

row = 0

# delta, theta, alpha
for bi in [0, 1, 2]:
    name, fmin, fmax = bands[bi]
    ax = axes[row]
    ax.plot(t_plot, filtered_plot[bi], linewidth=1.0)
    ax.set_ylabel(f"{name} ({fmin}-{fmax:g} Hz)")
    ax.set_ylim([-ymax_ts, ymax_ts])
    ax.grid(True, alpha=0.3)
    row += 1

# beta, gamma
for bi in [3, 4]:
    name, fmin, fmax = bands[bi]
    ax = axes[row]
    ax.plot(t_plot, filtered_plot[bi], linewidth=1.0)
    ax.set_ylabel(f"{name} ({fmin}-{fmax:g} Hz)")
    ax.set_ylim([-ymax_ts, ymax_ts])
    ax.grid(True, alpha=0.3)
    row += 1

# Alpha envelope fluctuations
ax = axes[row]
ax.plot(t_plot, alpha_env_lp_plot, linewidth=1.2)
ax.set_ylabel(r"$|\alpha|$ env" + f"\n(lp<{env_lp_hz} Hz)")
ax.grid(True, alpha=0.3)
row += 1

plt.tight_layout()
