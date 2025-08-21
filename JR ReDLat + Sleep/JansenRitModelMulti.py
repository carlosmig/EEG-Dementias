# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:57:25 2018

@author: Carlos Coronel

Modified version of the Jansen and Rit Neural Mass Model [1]. 

[1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked 
potential generation in a mathematical model of coupled cortical columns. 
Biological cybernetics, 73(4), 357-366.

[2] Deco, G., Cruzat, J., Cabral, J., Knudsen, G. M., Carhart-Harris, R. L., Whybrow, 
P. C., ... & Kringelbach, M. L. (2018). Whole-brain multimodal neuroimaging 
model using serotonin receptor maps explains non-linear functional effects of LSD. 
Current biology, 28(19), 3065-3074.


"""

import numpy as np
from numba import jit,float64, int32, vectorize
from numba.core.errors import NumbaPerformanceWarning
import warnings
import networkx as nx
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


#Simulation parameters
dt = 1E-3 #Integration step
teq = 10 #Simulation time for stabilizing the system
tmax = 30 #Simulation time
downsamp = 1 #Downsampling to reduce the number of points        
seed = 0 #Random seed

#Networks parameters

#Structural connectivity
nnodes = 90 #number of nodes
M = np.loadtxt('structural_Deco_AAL.txt')
norm = np.mean(np.sum(M,0)) #Normalization factor

#Node parameters
a1 = 120 #Inverse of the characteristic time constant for EPSPs (1/sec) Alpha neurons
b1 = 60 #Inverse of the characteristic time constant for IPSPs (1/sec) Alpha neurons
a2 = 660 #Inverse of the characteristic time constant for EPSPs (1/sec) Gamma neurons
b2 = 330 #Inverse of the characteristic time constant for IPSPs (1/sec) Gamma neurons
p = 220 * np.ones(nnodes) #Basal input to pyramidal population
sigma = 0.032 #Scaling noise factor

C = 135 #Global synaptic connectivity
C1 = C * 1 #Connectivity between pyramidal pop. and excitatory pop.
C2 = C * 0.8 #Connectivity between excitatory pop. and pyramidal pop.
C3 = C * 0.25 #Connectivity between pyramidal pop. and inhibitory pop.
C4_0 = C * 0.25 #Connectivity between inhibitory pop. and pyramidal pop.

A1 = 32.5 * a1 / 1000 #Amplitude of EPSPs Alpha neurons
B1 = 440 * b1 / 1000 #Amplitude of IPSPs Alpha neurons
A2 = 32.5 * a2 / 1000 #Amplitude of EPSPs Gamma neurons
B2 = 440 * b2 / 1000 #Amplitude of IPSPs Gamma neurons

alpha = 1 * np.ones(nnodes) #Proportion of alpha neurons

#Global copuling
K = 0 #Long-range pyramidal-pyramidal coupling

#Plasticity
target = 2.5 * np.ones(nnodes) #Target firing rate of pyramidal neurons
tau_C4 = 2 #Time constant for plasticity (inverse of learning rate)
plasticity_on = 0 #0 = no plasticity, 1 = with plasticity
C4_min = 0
beta = 1

#Sigmoid function parameters
e0 = 2.5 #Half of the maximum firing rate
v0 = 6 #V1/2
r0, r1, r2 = 0.56, 0.56, 0.56 #Slopes of sigmoid functions

#Recompile functions if their parameters changed
def update():
    f1.recompile()
    noise.recompile()


@vectorize([float64(float64,float64)],nopython=True)
#Sigmoid function
def s(v,r0):
    return (2 * e0) / (1 + np.exp(r0 * (v0 - v)))


@jit(float64[:,:](float64[:,:],float64,float64,float64[:],float64[:,:],float64,float64[:],float64,float64[:]),nopython=True)
#Jansen & Rit multicolumn model (intra-columnar outputs)
def f1(y,t,K,alpha,M,norm,target,tau_C4,p):
    x0_1, x1_1, x2_1, y0_1, y1_1, y2_1, x0_2, x1_2, x2_2, y0_2, y1_2, y2_2, C4 = y

    gamma = 1 - alpha
    x0  = alpha * x0_1 + gamma * x0_2 
    x1  = alpha * x1_1 + gamma * x1_2
    x2  = alpha * x2_1 + gamma * x2_2   
    
    x0_1_dot = y0_1
    y0_1_dot = A1 * a1 * (s(x1 - x2, r0)) - \
             2 * a1 * y0_1 - a1**2 * x0_1 
    x1_1_dot = y1_1
    y1_1_dot = A1 * a1 * (p + C2 * s(C1 * x0, r1) + K * C * M @ s(x1 - x2, r0) / norm) - \
             2 * a1 * y1_1 - a1**2 * x1_1
    x2_1_dot = y2_1
    y2_1_dot = B1 * b1 * (C4 * s(C3 * x0, r2)) - \
             2 * b1 * y2_1 - b1**2 * x2_1
 
    x0_2_dot = y0_2
    y0_2_dot = A2 * a2 * (s(x1 - x2, r0)) - \
             2 * a2 * y0_2 - a2**2 * x0_2 
    x1_2_dot = y1_2
    y1_2_dot = A2 * a2 * (p + C2 * s(C1 * x0, r1) + K * C * M @ s(x1 - x2, r0) / norm) - \
             2 * a2 * y1_2 - a2**2 * x1_2
    x2_2_dot = y2_2
    y2_2_dot = B2 * b2 * (C4 * s(C3 * x0, r2)) - \
             2 * b2 * y2_2 - b2**2 * x2_2

    C4_dot = s(C3 * x0, r2) * (s(x1 - x2, r0) - target) / tau_C4 * plasticity_on * (C4 / C - C4_min / C) ** beta      

    return(np.vstack((x0_1_dot, x1_1_dot, x2_1_dot, y0_1_dot, y1_1_dot, y2_1_dot,
                      x0_2_dot, x1_2_dot, x2_2_dot, y0_2_dot, y1_2_dot, y2_2_dot,
                      C4_dot)))


@jit(float64[:,:](float64),nopython=True)
#Noise
def noise(sigma):
    
    y1_1_dot = A1 * a1 * np.random.normal(0, sigma, nnodes)  
    y1_2_dot = A2 * a2 * np.random.normal(0, sigma, nnodes)
    
    return(np.vstack((y1_1_dot, y1_2_dot)))


@jit(int32(int32),nopython=True)
#This function is just for setting the random seed
def set_seed(seed):
    np.random.seed(seed)
    return(seed)


def Sim(verbose = True):
    """
    Run a network simulation with the current parameter values.
    
    Note that the time unit in this model is seconds.

    Parameters
    ----------
    verbose : Boolean, optional
        If True, some intermediate messages are shown.
        The default is False.

    Raises
    ------
    ValueError
        An error raises if the dimensions of M and the number of nodes
        do not match.

    Returns
    -------
    y : ndarray
        Time trajectory for the six variables of each node.
    time_vector : numpy array (vector)
        Values of time.

    """
    global M, tau_C4
         
    if M.shape[0]!=M.shape[1] or M.shape[0]!=nnodes:
        raise ValueError("check M dimensions (",M.shape,") and number of nodes (",nnodes,")")
    
    if M.dtype is not np.dtype('float64'):
        try:
            M=M.astype(np.float64)
        except:
            raise TypeError("M must be of numeric type, preferred float")    

    set_seed(seed); #Set the random seed   
    
    
    Nsim = int(tmax / dt) #Total simulation time points
    Neq = int(teq / dt ) #Number of points to discard
    Nmax = int(tmax / dt / downsamp) #Number of points of final simulated recordings
   
    #Initial conditions
    ic = np.ones((1, nnodes)) * np.array([0.131,  0.171, 0.343, 0.21, 3.07, 2.96, 
                                          0.131,  0.171, 0.343, 0.21, 3.07, 2.96, 
                                          C4_0])[:, None]     
   
    #Time vector
    time_vector = np.linspace(0, tmax, Nmax)

    row = 13 #Number of variables of the Jansen & Rit model
    col = nnodes #Number of nodes
    y_temp = np.copy(ic) #Temporal vector to update y values
    y = np.zeros((Nmax, row, col)) #Matrix to store values
    
    old_tau_C4 = tau_C4
    tau_C4 = 0.01
    
    if verbose == True:
        for i in range(1,Neq):
            y_temp += dt * f1(y_temp, i*dt, K, alpha, M, norm, target, tau_C4, p) 
            y_temp[[4,10],:] += np.sqrt(dt) * noise(sigma)        
        tau_C4 = old_tau_C4 / 2
        for i in range(1,Neq):
            y_temp += dt * f1(y_temp, i*dt, K, alpha, M, norm, target, tau_C4, p) 
            y_temp[[4,10],:] += np.sqrt(dt) * noise(sigma)        
        tau_C4 = old_tau_C4        
        y[0,:,:] = y_temp
        
        for i in range(1,Nsim):  
            y_temp += dt * f1(y_temp, i*dt, K, alpha, M, norm, target, tau_C4, p) 
            y_temp[[4,10],:] += np.sqrt(dt) * noise(sigma)            
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp
            if (i % (10 / dt)) == 0:
                print('Elapsed time: %i seconds'%(i * dt)) #this is for impatient people
    else:
        for i in range(1,Neq):
            y_temp += dt * f1(y_temp, i*dt, K, alpha, M, norm, target, tau_C4, p) 
            y_temp[[4,10],:] += np.sqrt(dt) * noise(sigma)        
        tau_C4 = old_tau_C4 / 2
        for i in range(1,Neq):
            y_temp += dt * f1(y_temp, i*dt, K, alpha, M, norm, target, tau_C4, p) 
            y_temp[[4,10],:] += np.sqrt(dt) * noise(sigma)        
        tau_C4 = old_tau_C4        
        y[0,:,:] = y_temp
        
        for i in range(1,Nsim):  
            y_temp += dt * f1(y_temp, i*dt, K, alpha, M, norm, target, tau_C4, p) 
            y_temp[[4,10],:] += np.sqrt(dt) * noise(sigma)           
            if (i % downsamp) == 0:
                y[i//downsamp,:,:] = y_temp
     
        
    return(y, time_vector)


def ParamsNode():
    pardict={}
    for var in ('a','b','A','B','r0',
                'r1','r2','e0','v0','C','C1','C2','C3',
                'C4','K','p','sigma'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsNet():
    pardict={}
    for var in ('nnodes', 'M'):
        pardict[var]=eval(var)
        
    return pardict

def ParamsSim():
    pardict={}
    for var in ('tmax','teq','dt','downsamp'):
        pardict[var]=eval(var)
        
    return pardict

#%%
# if __name__=='__main__':   

#     import matplotlib.pyplot as plt
#     from scipy import signal
#     from scipy.io import loadmat



#     odd = np.arange(1,82,2)[::-1]
#     even = np.arange(0,82,2)
#     SC_idx = np.append(even,odd)
    
#     bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    
#     all_SCs = loadmat('Z:/Data_Alz/sc_aal90_cn_ad_dft.mat')
#     # all_SCs = loadmat('/home/carlosmig/Z/Data_Alz/sc_aal90_cn_ad_dft.mat')
#     SC_CN = all_SCs['ave_sc'][1][0]
#     SC_CN = np.delete(SC_CN, np.arange(70,78,1), axis = 0)
#     SC_CN = np.delete(SC_CN, np.arange(70,78,1), axis = 1)
#     SC_CN = SC_CN[:,SC_idx][SC_idx,:]

#     M = SC_CN
#     nnodes = 82
#     p = 220 * np.ones(nnodes) * 1

#     teq = 10 #Simulation time for stabilizing the system
#     tmax = 60 #Simulation time    
#     plasticity_on = 1
#     target = 2.5 * np.ones(nnodes)

    

#     alpha = 0.5 * np.ones(nnodes) #Proportion of alpha neurons
    
#     #Global copuling
#     K = 0.25 #Long-range pyramidal-pyramidal coupling
#     norm = 1
#     seed = 2
    
#     update()
    
#     y,t = Sim(verbose = True)
    
#     EEG = (alpha * y[:,1,:] + (1 - alpha) * y[:,7,:]) - (alpha * y[:,2,:] + (1 - alpha) * y[:,8,:])
#     samp_rate = 1 / dt / downsamp
    
#     freqs, PSDs = signal.welch(EEG, samp_rate, 'hann', samp_rate*4, samp_rate*2, axis = 0, scaling = 'density')
#     PSD = np.mean(PSDs, axis = 1)
    
#     plt.figure(1)
#     plt.clf()
#     plt.subplot(2,2,1)
#     plt.plot(EEG[:1000,0])
#     plt.subplot(2,2,2)
#     plt.plot(freqs,PSD)
#     plt.xlim(0.1,80)
#     plt.subplot(2,2,3)
#     plt.plot(EEG[:1000,:])
#     plt.subplot(2,2,4)
#     plt.imshow(np.corrcoef(EEG.T), vmin = 0, vmax = 1, cmap = 'jet')
    
#     plt.figure(2)
#     plt.clf()
#     for i in range(0,5):
#         a0,b0 = signal.bessel(3,[2/samp_rate*bands[i][0],2/samp_rate*bands[i][1]],btype='bandpass')
#         Vfilt = signal.filtfilt(a0,b0,EEG,0)
#         plt.subplot(2,3,i+1)
#         FC = np.corrcoef(Vfilt.T)
#         plt.imshow(FC,vmin=0,vmax=1,cmap='jet')
#         print(np.mean(FC))