# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:47:50 2025

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm
import bct
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
rmse_reps = np.zeros(reps)

Y = SVM_ages.copy()  # Target variable (ages)

# Vectorize FC matrices using the upper triangular mask
vectorized = (SVM_FCs[triu_idx,:].T * mask).T
vectorized_north = (north_FCs[triu_idx,:].T * mask).T
vectorized_south = (south_FCs[triu_idx,:].T * mask).T
vectorized_AD = (AD_FCs[triu_idx,:].T * mask).T
vectorized_FTD = (FTD_FCs[triu_idx,:].T * mask).T

# Storage for test results
Y_pred_pool_reps = np.zeros((vectorized.shape[1], n_splits, reps)) + np.nan 
test_pool_reps = np.zeros((vectorized.shape[1], n_splits, reps)) + np.nan

#%%
#Nested cross-validation for hyper parameter tuning

# Generate log-spaced C values between 0.01 and 10
C_values = np.logspace(-2, 2, 11)
C_values = np.insert(C_values, 6, 2)

min_corr = 0.3  # Minimum correlation threshold

# Results matrices
n_inner_splits = n_splits  # Using the same number of splits for inner CV
mae_matrix = np.zeros((reps, n_splits, n_inner_splits, len(C_values)))
r_matrix = np.zeros((reps, n_splits, n_inner_splits, len(C_values)))
rmse_matrix = np.zeros((reps, n_splits, n_inner_splits, len(C_values)))

for k in range(reps):
    # Outer loop for cross-validation
    cv_outer = KFold(n_splits=n_splits, shuffle=True, random_state=k)
    
    outer_fold = 0  # To keep track of outer fold index
    for train_idx, test_idx in cv_outer.split(Y, Y):
        Y_train = Y[train_idx]

        # Compute correlations and threshold values
        corr_vec = np.array([stats.pearsonr(vectorized[x0, train_idx], Y_train)[0] for x0 in range(3321)])
        corr_vec[np.isnan(corr_vec)] = 0
        corr_vec_pool_idx = np.abs(corr_vec) >= min_corr
        
        X_pool_train = vectorized[corr_vec_pool_idx, :][:, train_idx]

        # Inner loop for hyperparameter tuning
        cv_inner = KFold(n_splits=n_inner_splits, shuffle=True, random_state=k)
        
        inner_fold = 0
        for inner_train_idx, inner_test_idx in cv_inner.split(Y_train, Y_train):
            Y_inner_train = Y_train[inner_train_idx]
            Y_inner_test = Y_train[inner_test_idx]
            X_inner_train = X_pool_train[:, inner_train_idx]
            X_inner_test = X_pool_train[:, inner_test_idx]
            
            for c_idx, C in enumerate(C_values):
                # Train the model
                model = svm.SVR(max_iter=10000, C=C, kernel='poly', degree=2, epsilon=0.0001)
                model.fit(X_inner_train.T, Y_inner_train)
                Y_inner_pred = model.predict(X_inner_test.T)
                
                # Compute performance metrics
                mae = mean_absolute_error(Y_inner_test, Y_inner_pred)
                rmse = np.sqrt(mean_squared_error(Y_inner_test, Y_inner_pred))
                r_value, _ = stats.pearsonr(Y_inner_test, Y_inner_pred)
                
                # Store the metrics
                mae_matrix[k, outer_fold, inner_fold, c_idx] = mae
                rmse_matrix[k, outer_fold, inner_fold, c_idx] = rmse
                r_matrix[k, outer_fold, inner_fold, c_idx] = r_value
            
            inner_fold += 1
        
        print(f"Rep {k+1}, Outer Fold {outer_fold+1} completed.")
        outer_fold += 1

print("\nCompleted all repetitions.")

#%%
#Plotting best hyperparameter

# Average across inner and outer folds
mae_avg = np.mean(mae_matrix, axis=(1, 2))
rmse_avg = np.mean(rmse_matrix, axis=(1, 2))
r_avg = np.mean(r_matrix, axis=(1, 2))

# Calculate overall averages across repetitions
mae_overall_avg = np.mean(mae_avg, axis=0)
rmse_overall_avg = np.mean(rmse_avg, axis=0)
r_overall_avg = np.mean(r_avg, axis=0)

# Find the best C for each metric (minimum for mae and rmse, maximum for r)
best_mae_idx = np.argmin(mae_overall_avg)
best_rmse_idx = np.argmin(rmse_overall_avg)
best_r_idx = np.argmax(r_overall_avg)

# Calculate percentage change relative to the best value
percent_diff_mae = 100 * (mae_overall_avg - mae_overall_avg[best_mae_idx]) / mae_overall_avg[best_mae_idx]
percent_diff_rmse = 100 * (rmse_overall_avg - rmse_overall_avg[best_rmse_idx]) / rmse_overall_avg[best_rmse_idx]
percent_diff_r = 100 * (r_overall_avg - r_overall_avg[best_r_idx]) / r_overall_avg[best_r_idx]

# Set the best index to 0 as requested
percent_diff_mae[best_mae_idx] = 0
percent_diff_rmse[best_rmse_idx] = 0
percent_diff_r[best_r_idx] = 0

# Plot the raw Pearson r values and the % change
plt.figure(figsize=(10, 4.5))

plt.subplot(1, 2, 1)
plt.plot(C_values, r_overall_avg, marker='o', color='red')
plt.xscale('log')
plt.title("Pearson Correlation (Raw Values)", fontsize=15)
plt.xlabel('C', fontsize=15)
plt.ylabel("Pearson r", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(C_values, percent_diff_r, marker='o', color='darkred')
plt.axhline(y=-5, color='black', linestyle='--', linewidth=1.5, label='5% Threshold')
plt.xscale('log')
plt.title("Percent Change in Pearson Correlation", fontsize=15)
plt.xlabel('C', fontsize=15)
plt.ylabel("% Change", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)

plt.tight_layout()
plt.show()

#%% 
#Training model and computing BAGs

#Prepare storage for error and correlation analyses
gap_north = np.zeros((n_splits,reps,vectorized_north.shape[1]))
gap_south = np.zeros((n_splits,reps,vectorized_south.shape[1]))
gap_AD = np.zeros((n_splits,reps,vectorized_AD.shape[1]))
gap_FTD = np.zeros((n_splits,reps,vectorized_FTD.shape[1]))


min_corr = 0.3  # Minimum correlation threshold

for k in range(0, reps):
    # Lists to store correlations and errors across folds
    rtemp_pool = []     
    error_pool = []
    rmse_pool = []
    
    # Initialize Support Vector Regression model
    regr = svm.SVR(max_iter=10000, C = 2, kernel='poly', degree=2, epsilon=0.0001)
    
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
        
        #age bias correction
        gap_train = regr.predict(X_pool_train.T) - Y_train
        a, b = stats.linregress(Y_train, gap_train)[0:2]
        
        # Correct the predictions for the age bias
        Y_pred_corrected = Y_pred - (a * Y_test + b)
        
        # Calculate error and correlation
        rtemp_pool.append(stats.pearsonr(Y_test, Y_pred)[0])  
        error_pool.append(np.mean(np.abs(Y_pred_corrected - Y_test)))
        rmse_pool.append(np.sqrt(mean_squared_error(Y_pred_corrected, Y_test)))        

        # Test north
        X_north = vectorized_north[corr_vec_pool_idx, :]
        Y_pred_north = regr.predict(X_north.T)
        gap_north[counter,k,:] = Y_pred_north - north_ages
        gap_north[counter,k,:] = gap_north[counter,k,:] - (a * north_ages + b)
        
        # Test south
        X_south = vectorized_south[corr_vec_pool_idx, :]
        Y_pred_south = regr.predict(X_south.T)
        gap_south[counter,k,:] = Y_pred_south - south_ages
        gap_south[counter,k,:] = gap_south[counter,k,:] - (a * south_ages + b)
        
        # Test AD
        X_AD = vectorized_AD[corr_vec_pool_idx, :]
        Y_pred_AD = regr.predict(X_AD.T)
        gap_AD[counter,k,:] = Y_pred_AD - AD_ages
        gap_AD[counter,k,:] = gap_AD[counter,k,:] - (a * AD_ages + b)    
        
        # Test FTD
        X_FTD = vectorized_FTD[corr_vec_pool_idx, :]
        Y_pred_FTD = regr.predict(X_FTD.T)
        gap_FTD[counter,k,:] = Y_pred_FTD - FTD_ages
        gap_FTD[counter,k,:] = gap_FTD[counter,k,:] - (a * FTD_ages + b)

        #store values for test
        test_pool_reps[test,counter,k] = Y_test
        Y_pred_pool_reps[test,counter,k] = (Y_pred)

        counter += 1
    
    rreps[k] = np.mean(rtemp_pool)
    e_reps[k] = np.mean(error_pool)
    rmse_reps[k] = np.mean(rmse_pool)
    print(k)

# Display mean results
print(np.mean(rreps[:]))  # Average correlation
print(np.mean(e_reps[:]))  # Average absolute error
print(np.mean(rmse_reps[:]))  # Average root mean square error

#average gaps across repetitions and folds
gap_south = np.nanmean(np.nanmean(gap_south,0),0)
gap_north = np.nanmean(np.nanmean(gap_north,0),0)
gap_AD = np.mean(np.mean(gap_AD,0),0)
gap_FTD = np.mean(np.mean(gap_FTD,0),0)


#%%
###PLOTTING

Y_pred_all = np.nanmean(np.nanmean(Y_pred_pool_reps,1),1)  
Y_test_all = np.nanmean(np.nanmean(test_pool_reps,1),1) 

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



