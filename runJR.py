# -*- coding: utf-8 -*-
"""

Modified version of the Jansen & Rit model [1] with inhibitory synaptic plasticity [2].
Two subpopulations of neural masses were used to simulate the neural activity of 
individual brain regions, according to [3].

[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked
 potential generation in a mathematical model of coupled cortical columns. 
 Biological cybernetics, 73(4), 357-366.

[2] Abeysuriya, R. G., Hadida, J., Sotiropoulos, S. N., Jbabdi, S.,
Becker, R., Hunt, B. A., ... & Woolrich, M. W. (2018). A biophysical 
model of dynamic balancing of excitation and inhibition in fast 
oscillatory large-scale networks. PLoS computational biology, 14(2), 
e1006007.

[3] Otero, M., Lea-Carnall, C., Prado, P., Escobar, M. J., & El-Deredy, W. 
(2022). Modelling neural entrainment and its persistence: influence of 
frequency of stimulation and phase at the stimulus offset. Biomedical 
Physics & Engineering Express, 8(4), 045014. 


@author: Carlos Coronel
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import JansenRitModelMulti as JR

#Simulations parameters
JR.tmax = 120 #max sim time
JR.teq = 20 #eq time for reaching steady-state dynamics
JR.dt = 0.001 #integration step
JR.downsamp = 2 #downsampling for reducing memory consumption

#networks parameters
JR.M = np.loadtxt('Structural_Deco_AAL.txt') #SC matrix
JR.nnodes = len(JR.M) #number of nodes
JR.norm = 1 #normalization factor

#Noise an inputs
JR.sigma = 1 #noise scaling factor
JR.p = 220 * np.ones(JR.nnodes) #inputs for individual nodes

#Plasticity
JR.plasticity_on = 1 #1: activated, 0: disabled
JR.target = 2.5 * np.ones(JR.nnodes) #target firing rate in Hz
JR.tau_p = 2 #time constant for plasticity (in seconds)

#Global coupling
JR.K = 0.32

#Proportion of alpha neurons
JR.alpha = 0.5 * np.ones(JR.nnodes)
JR.gamma = 1 - JR.alpha

#updating some parameters
JR.update() #avoid this in extensive simulations (call it increases memory consumption).
            #just use it one time before parallelization

#Simulation starts here
y, t = JR.Sim(verbose = True)
#Simulations end here

#EEG-like signals
EEG = (JR.alpha * y[:,1,:] + JR.gamma * y[:,7,:]) - (JR.alpha * y[:,2,:] + JR.gamma * y[:,8,:])

#Power spectrum using Welch 
freqs, PSDs = signal.welch(EEG, 1000//JR.downsamp, 'hann', 4000//JR.downsamp, 2000//JR.downsamp, axis = 0, scaling = 'density')
PSD = np.mean(PSDs, axis = 1)

#filtering and computing FCs
bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40), (0.5, 40)]
FCs = np.zeros((JR.nnodes, JR.nnodes, 6))

for i in range(0,6):
    fmin, fmax = bands[i]
    a0,b0 = signal.bessel(3, 2 * JR.dt * JR.downsamp * np.array([fmin, fmax]), btype = 'bandpass')
    Vfilt = signal.filtfilt(a0, b0, EEG, 0)
    FCs[:,:,i] = np.corrcoef(Vfilt.T)
    

#%%

###some plots

plt.figure(1)
plt.clf()

plt.subplot(2,1,1)
plt.plot(t[30000:40000], EEG[30000:40000,10:12])
plt.xlabel('Time (seconds)')
plt.ylabel('EEG (mV)')
plt.title('EEG-like signals (Raw)')

plt.subplot(2,2,3)
plt.loglog(freqs,PSD)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (A.U.)')
plt.title('Averaged Power Spectrum')

plt.subplot(2,2,4)
plt.imshow(FCs[:,:,-1], vmin = 0, vmax = 1, cmap = 'rainbow')
plt.xticks([])
plt.yticks([])
plt.title('FC matrix (broadband)')

plt.tight_layout()


plt.figure(2)
plt.clf()

my_title = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'broadband']
for i in range(0,6):
    plt.subplot(2,3,1+i)
    plt.imshow(FCs[:,:,i], vmin = 0, vmax = 1, cmap = 'rainbow')    
    plt.title(my_title[i])
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
        
    
    
    
    
    
    
    
    
    
    









