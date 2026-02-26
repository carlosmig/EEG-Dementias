import numpy as np
import matplotlib.pyplot as plt
import JansenRitModelMulti as JR

# Fixed parameters
JR.K = 0.5
JR.alpha = 0.5 * np.ones(JR.nnodes)
JR.gamma = 1.0 - JR.alpha
JR.plasticity_on = 1
JR.target = 2.5 * np.ones(JR.nnodes)

JR.sigma = 0.0   # no noise (strongly recommended)
JR.seed = 0

# Simulation parameters (keep your values)
JR.dt = 1e-3
JR.teq = 0
JR.tmax = 120
JR.downsamp = 10

tau_list = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100], dtype=float)

# windows (seconds)
init_window_sec   = 2.0     # for C0
steady_window_sec = 120.0    # for C_infinity
fit_start_sec     = 0.0     # set to JR.teq if you want to start after equilibration

tau_eff = np.full(len(tau_list), np.nan)

for i, tau in enumerate(tau_list):
    JR.tau_C4 = float(tau)
    JR.update()

    y, t = JR.Sim(verbose=False)

    # ROI-averaged C4(t)
    C4_avg = y[:, -1, :].mean(axis=1)

    # masks
    t_end = t.max()
    m0 = (t >= fit_start_sec) & (t <= fit_start_sec + init_window_sec)
    m_inf = t >= (t_end - steady_window_sec)

    if (not np.any(m0)) or (not np.any(m_inf)):
        raise RuntimeError("init/steady windows incompatible with simulation length.")

    C0 = C4_avg[m0].mean()
    Cinf = C4_avg[m_inf].mean()

    dC = Cinf - C0
    if np.abs(dC) < 1e-9:
        # basically no adaptation observed
        tau_eff[i] = np.nan
        continue

    # normalized progress toward final value (handle decreasing too)
    prog = (C4_avg - C0) / dC
    if dC < 0:
        prog = (C0 - C4_avg) / (C0 - Cinf)

    # only consider times after fit_start_sec
    mfit = t >= fit_start_sec
    t_fit = t[mfit]
    prog_fit = prog[mfit]

    # find first time it reaches 1-1/e ~= 0.632
    target = 1.0 - 1.0/np.e
    idx = np.where(prog_fit >= target)[0]

    if len(idx) == 0:
        # did not reach target within sim time -> lower bound
        tau_eff[i] = t_fit[-1]
    else:
        tau_eff[i] = t_fit[idx[0]]

#%%

# Plot
plt.rcParams["svg.fonttype"] = "none"
plt.figure(figsize=(7, 5))
plt.plot(tau_list, tau_eff, marker='o', linewidth=2)
plt.xscale('log')
plt.yscale('log')  # usually helpful here
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.xlabel(r'Plasticity timescale $\tau_{C4}$', fontsize = 16)
plt.ylabel(r'Estimated convergence time (sec)',
           fontsize = 16)
plt.title("Convergence vs ISP timescale", fontsize = 16)
plt.tight_layout()
plt.show()

