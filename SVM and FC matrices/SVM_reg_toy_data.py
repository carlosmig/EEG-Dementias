# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:54:06 2023

SVM regression using toy data

@author: Carlos Coronel
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn.model_selection import KFold


def matrix_recon(x):
    """
    Function to re-built a connectivity matrix from its vectorized form.
    
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
    matrix = matrix + matrix.T
   
    return(matrix)   

def standardize(x, axis=None):
    return (x - x.mean(axis, keepdims=True)) / x.std(axis, keepdims=True)

#%%
###TOY FUNCTIONAL CONNECTIVITY DATA

N = 90 #Number of Brain Areas
M = (N**2-N)//2 #Number of connections
S = 500 #Subjects

np.random.seed(1)

#Ages
ages_G1 = np.random.uniform(20,90,S)
ages_G2 = np.random.uniform(30,80,S)

#Toy FCs
FCs_G1 = np.random.uniform(-0.5,0.5,((S,N,N)))
FCs_G2 = np.random.uniform(-0.5,0.5,((S,N,N)))

#This is for partially correlating (artificially) a subset of connections with age
mask = np.random.binomial(1, 0.025, M)

#another mask for vectorizing matrices
dummy_mat = np.zeros((N,N))
for i in range(0,N-1):
    for j in range(1+i,N):
        dummy_mat[i,j] = 1
triu_idx = dummy_mat == 1

#Vectorized FCs
vectorized_G1 = FCs_G1[:,triu_idx].T
vectorized_G2 = FCs_G2[:,triu_idx].T    

#correlating some of the connections with age
vectorized_G1[mask == 1,:] += 0.2 - 0.005 * ages_G1 + np.random.normal(0,0.15,(np.sum(mask),S))
vectorized_G2[mask == 1,:] += 0.16 - 0.005 * ages_G2 + np.random.normal(0,0.15,(np.sum(mask),S)) 
#different intercepts would create an artifical BAG between G1 and G2

#normalizing
vectorized_G1 /= np.max(vectorized_G1)
vectorized_G1[vectorized_G1 > 1]  = 1                        
vectorized_G1[vectorized_G1 < -1]  = -1                        

vectorized_G2 /= np.max(vectorized_G2)
vectorized_G2[vectorized_G2 > 1]  = 1                        
vectorized_G2[vectorized_G2 < -1]  = -1                        


#%%

#Number of repetitions for SVM
reps = 20 # >= 20 should be okay

#for saving the results
rreps = np.zeros(reps) #correlation
ereps = np.zeros(reps) #mean absolute error

#X and Y for SVM reg (Training with G1 data)
vectorized = np.copy(vectorized_G1)
Y = np.copy(ages_G1)
# np.random.shuffle(Y) #shuffling for surrogate data
Y = Y.astype(int)

#standardizing
mean, std = np.mean(vectorized, 1), np.std(vectorized, 1)
vectorized = ((vectorized.T - mean) / std).T
vectorized_G1 = ((vectorized_G1.T - mean) / std).T
vectorized_G2 = ((vectorized_G2.T - mean) / std).T

#concordance matrices (links most correlated with age across folds and repetitions)
CM_pos = np.zeros((N,N,reps))
CM_neg = np.zeros((N,N,reps))

#Number of splits for cross validation
n_splits = 5

#Brain Age Gaps
gap_G1 = np.zeros((n_splits,reps,vectorized_G1.shape[1]))
gap_G2 = np.zeros((n_splits,reps,vectorized_G2.shape[1]))

#Min absolute correlation threshold (connectivity versus age)
min_corr = 0.2

#Saving test data results across folds and iterations
test_pool_reps = []
Y_pred_pool_reps = []

for k in range(0,reps):
    
    #to store correlations and erros across folds
    rtemp_pool = []     
    error_pool = []
    
    #Linear SVM-reg
    regr = svm.LinearSVR(max_iter = 1000, C = 1, fit_intercept = True, random_state = k,
                            intercept_scaling = 1)
        
    #5-fold cross validation
    cv = KFold(n_splits = n_splits, shuffle = True, random_state = k)
        
    counter = 0 #for indexing across folds
    for train, test in cv.split(Y, Y):
        
        #train and test ages
        Y_train = Y[train]
        Y_test = Y[test]    
        
        #feature selection
        corr_vec = np.array([stats.pearsonr(vectorized[x0,train], Y_train)[0] for x0 in range(0,M)])    
        corr_vec[np.isnan(corr_vec)] = 0
        corr_mat = matrix_recon(corr_vec)
        
        #thresholding
        corr_vec_pool_idx = np.abs(corr_vec) >= min_corr
        corr_mat_pos = corr_mat * (corr_mat >= min_corr)
        corr_mat_neg = corr_mat * (corr_mat <= -min_corr) 
        
        #train and test connectivity values
        X_pool_train = vectorized[corr_vec_pool_idx,:][:,train]
        X_pool_test = vectorized[corr_vec_pool_idx,:][:,test]

        #updating concordance matrix
        CM_pos[:,:,k] += (corr_mat >= min_corr) * 1 / n_splits
        CM_neg[:,:,k] += (corr_mat <= -min_corr) * 1 / n_splits     
        

        #Training SVMs    
        regr.fit(X_pool_train.T, Y_train)    
        #Prediction
        Y_pred = regr.predict(X_pool_test.T)
        #Performance (correlation and mean absolute error)
        rtemp_pool.append(stats.pearsonr(Y_test, Y_pred)[0])  
        error_pool.append(np.mean(np.abs(Y_test - Y_pred)))
        #saving test data
        test_pool_reps.append(test)
        Y_pred_pool_reps.append(Y_pred)    
        
        # Predict on training data to compute BAGs for training set
        Y_pred_train = regr.predict(X_pool_train.T)
        gap_train = Y_pred_train - Y_train  # BAGs for training data
        
        # Apply linear regression to compute a and b
        a, b = stats.linregress(Y_train, gap_train)[:2]
        
        ###TEST G1 USING BOTH
        X_test_G1 = vectorized_G1[corr_vec_pool_idx,:]
        Y_test_G1 = ages_G1
        Y_pred_G1 = regr.predict(X_test_G1.T)
        gap_G1[counter,k,:] = (Y_pred_G1 - Y_test_G1) - (a * Y_test_G1 + b)
        
        ###TEST G2 USING BOTH
        X_test_G2 = vectorized_G2[corr_vec_pool_idx,:]
        Y_test_G2 = ages_G2
        Y_pred_G2 = regr.predict(X_test_G2.T)
        gap_G2[counter,k,:] = (Y_pred_G2 - Y_test_G2) - (a * Y_test_G2 + b)
        
        counter += 1
        
    rreps[k] = np.mean(rtemp_pool)
    ereps[k] = np.mean(error_pool)
    
    print(k)

#BAGs averaged across folds and iterations
gap_G2 = np.mean(np.mean(gap_G2,0),0)
gap_G1 = np.mean(np.mean(gap_G1,0),0)

print('Performance')
print('Correlation: %.3f'%np.mean(rreps))
print('MAE: %.3f'%np.mean(ereps))


#%%
###Plotting concordance matrices

plt.figure(1)
plt.clf()
plt.subplot(1,2,1)
plt.imshow(np.mean(CM_pos,2) * (np.mean(CM_pos,2) >= 0))
plt.title('Concordance mat for pos')
plt.subplot(1,2,2)
plt.imshow(np.mean(CM_neg,2) * (np.mean(CM_neg,2) >= 0))
plt.title('Concordance mat for neg')
plt.tight_layout()



#%%

###Plotting model's predictions

Y_pred_all = np.zeros((S,reps))
for k in range(0,reps):
    Y_pred_all[test_pool_reps[k],k] = Y_pred_pool_reps[k]
Y_pred_all = np.sum(Y_pred_all,1) / np.sum(Y_pred_all > 0, 1)

plt.figure(2, figsize = (5,4.5))
plt.clf()
plt.plot(Y, Y_pred_all, 'bo')
a, b, r = stats.linregress(Y, Y_pred_all)[0:3]
lines = b + a *Y

plt.plot(Y, lines, color = 'crimson', lw = 1.5, ls = 'dashed')

plt.xlabel('Chronological age (years)')
plt.ylabel('Predicted age (years)')
plt.title("Pearson's r = %.3f (cross validation)"%np.mean(rreps))

plt.xlim(-10,110)
plt.ylim(-10,110)

#%%

###Plotting BAGs

plt.figure(3)
plt.clf()
plt.boxplot([gap_G1, gap_G2])
plt.title('Brain Age Gap')
plt.ylabel('BAG (years)')
plt.xlabel('Groups')
plt.xticks([1,2], ['G1', 'G2'])
plt.tight_layout()











