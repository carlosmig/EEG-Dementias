# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:47:50 2025

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm
import bct
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings to keep output clean

def matrix_recon(x):
    """
    Function to reconstruct a connectivity matrix from its vectorized form.
    
    Inputs:
        x: numpy array, vector with connectivity values.
    Output:
        matrix: numpy array, connectivity matrix.
    
    """
    npairs = len(x)
    nnodes = int((1 + np.sqrt(1 + 8 * npairs)) // 2)
    
    matrix = np.zeros((nnodes, nnodes))
    idx = 0
    for row in range(0, nnodes - 1):
        for col in range(row + 1, nnodes):
            matrix[row, col] = x[idx]
            idx = idx + 1
    matrix = matrix + matrix.T  # Ensure symmetry
   
    return(matrix)   

def cohen_d(x,y):
    """
    Function for computing Cohen's D effect size.
    
    Inputs:
        x,y: numpy arrays, vectors with observations.
    Output:
        Effect size.
    
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x) ** 2 + (ny-1)*np.std(y) ** 2) / dof)


#%% Load functional connectivity (FC) and age data

#Load the 3 sets of FCs for SVM training
SVM_FCs_1 = np.load('SVM_FCs_1.npy')  
SVM_FCs_2 = np.load('SVM_FCs_2.npy')  
SVM_FCs_3 = np.load('SVM_FCs_3.npy')  
SVM_FCs = np.concatenate((SVM_FCs_1, SVM_FCs_2, SVM_FCs_3), axis = 2) # FC matrices for SVM analysis

#Load the 2 sets of FCs for north and south
north_FCs_1 = np.load('north_FCs_1.npy')  
north_FCs_2 = np.load('north_FCs_2.npy')  
north_FCs = np.concatenate((north_FCs_1, north_FCs_2), axis = 2) # North

south_FCs_1 = np.load('south_FCs_1.npy')  
south_FCs_2 = np.load('south_FCs_2.npy')  
south_FCs = np.concatenate((south_FCs_1, south_FCs_2), axis = 2) # south

# Patients data
AD_FCs = np.load('AD_FCs.npy')  # FC matrices for Alzheimer's patients
FTD_FCs = np.load('FTD_FCs.npy')  # FC matrices for frontotemporal dementia patients

SVM_ages = np.load('SVM_ages.npy')  # Age labels for SVM
north_ages = np.load('north_ages.npy')
south_ages = np.load('south_ages.npy')
AD_ages = np.load('AD_ages.npy')
FTD_ages = np.load('FTD_ages.npy')

# Create a mask of significant connections based on thresholded average FC
mask = bct.threshold_proportional(np.mean(SVM_FCs, 2), 0.25) > 0

# Create an upper triangular matrix to extract unique connectivity values
# Dummy matrix to get upper triangular indices
dummy_mat = np.zeros((82,82))
for i in range(0,81):
    for j in range(1+i,82):
        dummy_mat[i,j] = 1
triu_idx = dummy_mat == 1

# Apply the mask to the upper triangular indices
mask = mask[dummy_mat == 1]

# Number of repetitions for SVM
n_splits = 10  # Number of cross-validation folds
reps = 20  # Number of repetitions

# Initialize arrays to store results
rreps = np.zeros(reps)  # Correlation results
e_reps = np.zeros(reps)  # Mean absolute error results

Y = SVM_ages.copy()  # Target variable (ages)

# Storage for test results
test_pool_reps = []
Y_pred_pool_reps = []

# Vectorize FC matrices using the upper triangular mask
vectorized = (SVM_FCs[triu_idx,:].T * mask).T
vectorized_north = (north_FCs[triu_idx,:].T * mask).T
vectorized_south = (south_FCs[triu_idx,:].T * mask).T
vectorized_AD = (AD_FCs[triu_idx,:].T * mask).T
vectorized_FTD = (FTD_FCs[triu_idx,:].T * mask).T

#%% Prepare storage for error and correlation analyses

gap_north = np.zeros((n_splits,reps,vectorized_north.shape[1]))
gap_south = np.zeros((n_splits,reps,vectorized_south.shape[1]))
gap_AD = np.zeros((n_splits,reps,vectorized_AD.shape[1]))
gap_FTD = np.zeros((n_splits,reps,vectorized_FTD.shape[1]))

r_north = np.zeros((n_splits,reps))
error_north = np.zeros((n_splits,reps,vectorized_north.shape[1]))
r_south = np.zeros((n_splits,reps))
error_south = np.zeros((n_splits,reps,vectorized_south.shape[1]))
error_AD = np.zeros((n_splits,reps,vectorized_AD.shape[1]))
error_FTD = np.zeros((n_splits,reps,vectorized_FTD.shape[1]))

min_corr = 0.3  # Minimum correlation threshold

for k in range(0, reps):
    # Lists to store correlations and errors across folds
    rtemp_pool = []     
    error_pool = []
    
    # Initialize Support Vector Regression model
    regr = svm.SVR(max_iter=10000, C=2, kernel='poly', degree=2, epsilon=0.0001)
    
    # Perform K-Fold cross-validation
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=k)
    
    counter = 0  # Counter for tracking fold indices
    for train, test in cv.split(Y, Y):
        Y_train = Y[train]
        Y_test = Y[test]
        
        # Compute correlations and threshold values
        corr_vec = np.array([stats.pearsonr(vectorized[x0, train], Y_train)[0] for x0 in range(0, 3321)])
        corr_vec[np.isnan(corr_vec)] = 0
        corr_mat = matrix_recon(corr_vec)
        
        # Select significant features based on correlation
        corr_vec_pool_idx = np.abs(corr_vec) >= min_corr
        X_pool_train = vectorized[corr_vec_pool_idx, :][:, train]
        X_pool_test = vectorized[corr_vec_pool_idx, :][:, test]

        # Train and test the SVR model
        regr.fit(X_pool_train.T, Y_train)    
        Y_pred = regr.predict(X_pool_test.T)
        
        # Calculate error and correlation
        rtemp_pool.append(stats.pearsonr(Y_test, Y_pred)[0])  
        error_pool.append(np.mean(np.abs(Y_pred - Y_test)))

        counter += 1
    
    rreps[k] = np.mean(rtemp_pool)
    e_reps[k] = np.mean(error_pool)
    print(k)

# Display mean results
print(np.mean(rreps[:]))  # Average correlation
print(np.mean(e_reps[:]))  # Average error

#%%
###PLOTTING

Y_pred_all = np.zeros((np.sum(SVM_ages > 0),reps))  # Initialize array to store predicted ages
for k in range(0,reps):
    Y_pred_all[test_pool_reps[k],k] = Y_pred_pool_reps[k]  # Assign predicted values for each repetition
Y_pred_all = np.sum(Y_pred_all,1) / np.sum(Y_pred_all > 0, 1)  # Compute the average prediction ignoring zeros

# Combined model visualization
plt.figure(1, figsize = (5,4.5))
plt.clf()
plt.plot(Y, Y_pred_all, 'bo')  # Scatter plot of actual vs predicted ages
a, b, r = stats.linregress(Y, Y_pred_all)[0:3]  # Linear regression
lines = b + a * Y

plt.plot(Y, lines, color = 'crimson', lw = 1.5, ls = 'dashed')  # Plot regression line
plt.xlabel('Chronological age (years)')
plt.ylabel('Predicted age (years)')
plt.title("Pearson's r = %.3f (cross validation)"%np.mean(rreps))  # Display correlation coefficient

plt.xlim(-10,110)
plt.ylim(-10,110)

###Boxplot analysis of Brain Age Gap (BAG)

plt.figure(2, figsize = (13,4))
plt.clf()

# Filter out outliers beyond 2 standard deviations
filt_south = np.abs(gap_south - np.mean(gap_south)) < 2 * np.std(gap_south)
filt_north = np.abs(gap_north - np.mean(gap_north)) < 2 * np.std(gap_north)

plt.subplot(1,2,1)
plt.boxplot([gap_south[filt_south], gap_north[filt_north]])  # Boxplots for healthy controls
plt.xticks([1,2], ['HCs South', 'HCs North'])
plt.xlabel('Groups')
plt.ylabel('BAG (years)')
plt.title('BAG in HCs')

# Statistical comparison: North vs South healthy controls
t, p = stats.ttest_ind(gap_south[filt_south], gap_north[filt_north])
d = cohen_d(gap_south[filt_south], gap_north[filt_north])
print('North Vs South')
print('t = %.3f, p = %.3e, d  %.3f'%(t,p,d))

# Filter and boxplot for aged south group and patients
gap_south_aged = gap_south[south_ages > 62]
filt_south_aged = np.abs(gap_south_aged - np.mean(gap_south_aged)) < 2 * np.std(gap_south_aged)
filt_AD = np.abs(gap_AD - np.mean(gap_AD)) < 2 * np.std(gap_AD)
filt_FTD = np.abs(gap_FTD - np.mean(gap_FTD)) < 2 * np.std(gap_FTD)

plt.subplot(1,2,2)
plt.boxplot([gap_south_aged[filt_south_aged], gap_AD[filt_AD], gap_FTD[filt_FTD]])
plt.xticks([1,2,3], ['HCs South', 'AD', 'bvFTD'])
plt.xlabel('Groups')
plt.ylabel('BAG (years)')
plt.title('BAG in Patients')

plt.tight_layout()

# Statistical tests for patient vs control groups
print('Ages difference patients versus controls')
print(stats.ttest_ind(AD_ages, south_ages[south_ages > 62]))
print(stats.ttest_ind(FTD_ages, south_ages[south_ages > 62]))

t, p = stats.ttest_ind(gap_AD[filt_AD], gap_south_aged[filt_south_aged])
d = cohen_d(gap_AD[filt_AD], gap_south_aged[filt_south_aged])
print('South Vs AD')
print('t = %.3f, p = %.3e, d  %.3f'%(t,p,d))

t, p = stats.ttest_ind(gap_FTD[filt_FTD], gap_south_aged[filt_south_aged])
d = cohen_d(gap_FTD[filt_FTD], gap_south_aged[filt_south_aged])
print('South Vs FTD')
print('t = %.3f, p = %.3e, d  %.3f'%(t,p,d))



