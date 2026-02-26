import numpy as np
import matplotlib.pyplot as plt
import JansenRitModelMulti as JR

# Load SC and set common params

SC_CN = np.load('SC_CN_ReDLat.npy')

JR.M = SC_CN
JR.nnodes = len(JR.M)
nnodes = JR.nnodes

# Simulation parameters
JR.dt = 1e-3
JR.teq = 5
JR.tmax = 15
JR.downsamp = 10
JR.tau_C4 = 2
JR.p = 220 * np.ones(nnodes)
JR.norm = 1
JR.sigma = 1

# Plasticity settings
JR.plasticity_on = 1
target_hz = 2.5

# Seeds
seeds = np.arange(10)

# Discard time for computing mean C4 post-transient
discard_sec = JR.teq

# Nodal strength from SC
strength = JR.M.sum(axis=1)


# SWEEP
K_vals = np.linspace(0.1, 1.0, 10)
alpha_list = np.array([0.50, 0.75, 1.0])

# Storage:
C4mean_byAK = np.zeros((len(alpha_list), len(K_vals), nnodes))
slope_seed  = np.zeros((len(alpha_list), len(K_vals), len(seeds)))

for ai, r_alpha in enumerate(alpha_list):
    for ki, K in enumerate(K_vals):
        print(f"[Sweep] r_alpha={r_alpha:.2f} ({ai+1}/{len(alpha_list)}), "
              f"K={K:.3f} ({ki+1}/{len(K_vals)})")

        C4means_seeds = np.zeros((len(seeds), nnodes))

        for si, seed in enumerate(seeds):
            JR.K = float(K)
            JR.alpha = float(r_alpha) * np.ones(nnodes)
            JR.gamma = 1.0 - JR.alpha
            JR.target = target_hz * np.ones(nnodes)
            JR.seed = int(seed)

            JR.update()

            y, t = JR.Sim(verbose=False)

            # C4(t) in last state variable: shape (nt, nnodes)
            C4_hist = y[:, -1, :]

            # discard transient based on returned t
            mask = t >= discard_sec
            if not np.any(mask):
                raise RuntimeError(
                    f"discard_sec={discard_sec}s beyond simulation time. Max t={t.max():.3f}s"
                )

            # Post-transient mean C4 per node
            C4_mean = C4_hist[mask, :].mean(axis=0)
            C4means_seeds[si, :] = C4_mean

            # slope for this seed: y = m*x + b
            m, b = np.polyfit(strength, C4_mean, 1)
            slope_seed[ai, ki, si] = m

        # average across seeds
        C4mean_byAK[ai, ki, :] = C4means_seeds.mean(axis=0)

# Average slope across seeds
slope_byAK = slope_seed.mean(axis=2)
slope_std  = slope_seed.std(axis=2, ddof=1)


#%%

# PLOTTING
plt.rcParams["svg.fonttype"] = "none"

x = strength
xline = np.linspace(x.min(), x.max(), 200)

alpha_to_plot = 0.5
ai_plot = int(np.where(np.isclose(alpha_list, alpha_to_plot))[0][0])

plt.figure(figsize=(7, 6))
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, len(K_vals)))

for ki, K in enumerate(K_vals):
    y = C4mean_byAK[ai_plot, ki, :] 

    # scatter
    plt.scatter(x, y, s=18, alpha=0.30, color=colors[ki])

    # regression line
    m, b = np.polyfit(x, y, 1)
    plt.plot(xline, m * xline + b, color=colors[ki], linewidth=2, label=f"K={K:.2f}")

plt.xlabel("Nodal strength")
plt.ylabel("Mean C4 (post-transient)")
plt.title(f"Strength vs mean C4 (K sweep, r_alpha={alpha_to_plot})")
plt.grid(True)
plt.tight_layout()
plt.legend(ncol=2, fontsize=9, frameon=False)
plt.show()

# K versus slopes for r_alpha = 0, 0.5, 1  (three curves)
plt.figure(figsize=(7, 5))

for ai, r_alpha in enumerate(alpha_list):
    plt.plot(K_vals, slope_byAK[ai, :], marker='o', linewidth=2, label=f"r_alpha={r_alpha:.1f}")

plt.xlabel("K")
plt.ylabel("Slope of regression: C4 ~ strength")
plt.title("K vs slope (strength -> mean C4), for r_alpha = 0, 0.5, 1")
plt.grid(True)
plt.tight_layout()
plt.legend(frameon=False)
plt.show()






