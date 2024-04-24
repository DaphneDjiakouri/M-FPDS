#Dafni Tziakouri
#NLA_Project 2

# First we will upload the packages we need 

import pandas as pd
import numpy as np
from numpy import genfromtxt, vstack, sqrt, std, concatenate, reshape, dot
from numpy.linalg import norm, svd
from numpy.core.fromnumeric import argmin
import sys
from scipy.linalg import solve_triangular, qr
import matplotlib.pyplot as plt
import imageio 
from imageio import imread, imsave
from pandas import read_csv, DataFrame, concat

##########################
# 1. Least Squares problem
##########################

# Let's now solve the least squares problem by using SVD and QR factorization

# Creating a function using SVD and returning the least squares solution
def svd_LS(A, b):
    # Step 1: SVD of matrix A
    U, Sigma, VT = np.linalg.svd(A, full_matrices=False)

    # Step 2: Calculate the inverse of the diagonal matrix of singular values
    Sigma_inv = np.diag(1 / Sigma)

    # Step 3: Compute the least squares solution using SVD components
    # Combine the transposed right singular vectors, the inverse of singular values, 
    # and the transposed left singular vectors to form the solution matrix
    x_svd = VT.T @ Sigma_inv @ U.T @ b

    # Step 4: Return the least squares solution obtained using SVD
    return x_svd

# Creating a function using the QR factorization with rank deficient 
#and returning the least squares solution
def qr_LS(A, b):
    # Step 1: Rank of A matrix
    Rank = np.linalg.matrix_rank(A)

    # Initialize x_qr
    x_qr = None

    # Step 2: Case for Full-rank matrix
    if Rank == A.shape[1]:
        # Substep 2.1: QR factorization for full-rank matrix
        Q_fullr, R_fullr = np.linalg.qr(A)
        
        # Substep 2.2: Solve the system using back substitution
        y_aux = np.transpose(Q_fullr).dot(b)
        x_qr = solve_triangular(R_fullr, y_aux)

    # Step 3: Case for Rank-deficient matrix
    else:
        # Substep 3.1: QR factorization with pivoting for rank-deficient matrix
        Q, R, P = qr(A, mode='economic', pivoting=True)  
        
        # Substep 3.2: Extract the relevant part of R for rank-deficient case
        R_def = R[:Rank, :Rank]
        
        # Substep 3.3: Solve the system for rank-deficient case
        c = np.transpose(Q).dot(b)[:Rank]
        u = solve_triangular(R_def, c)
        v = np.zeros((A.shape[1] - Rank))
        
        # Substep 3.4: Combine solutions for rank-deficient case
        x_qr = np.linalg.solve(np.transpose(np.eye(A.shape[1])[:, P]), np.concatenate((u, v)))
        
    # Step 4: Return the solution vector x_qr
    return x_qr

# Now, we will upload the data to check our functions

# Load datasets
def datafile(degree):
    data = genfromtxt("dades.txt", delimiter="   ")
    points, b = data[:, 0], data[:, 1]

    # Form A from the points given in dataset, for a given degree
    A = vstack([points ** d for d in range(degree)]).T

    return A, b

def datafile2(degree):
    # Read matrices and return
    data = genfromtxt('dades_regressio.csv', delimiter=',')
    A, b = data[:, :-1], data[:, -1]
    return A, b

#--------------------------------------------------------------------#

# Load datasets
svd_errors = []
degrees=range(3,10)

# Call functions with datafile
for degree in range(3,10):
    A, b = datafile(degree)
    x_svd = svd_LS(A, b)
    x_qr = qr_LS(A, b)
    svd_errors.append(norm(A.dot(x_svd) - b))

min_svd_error_pos = argmin(svd_errors)
best_degree = min_svd_error_pos+3

# Compare results for datafile
print("Results for datafile:")
print("Best degree:", best_degree)
A, b = datafile(best_degree)
x_svd = svd_LS(A, b)
x_qr = qr_LS(A, b)
print("\n")
print("LS solution using SVD:", x_svd)
print("Solution norm:", norm(x_svd))
print("Error:", norm(A.dot(x_svd)-b))
print("\n")
print("LS solution using QR:", x_qr)
print("Solution norm:", norm(x_qr))
print("Error:", norm(A.dot(x_qr)-b))
print("\n")

#--------------------------------------------------------------------#

# Call functions with datafile2
# Compare results for datafile2
print("Results for datafile2:")
print("Best degree:", best_degree)
A, b = datafile2(best_degree)
x_svd = svd_LS(A, b)
x_qr = qr_LS(A, b)
print("\n")
print("LS solution using SVD:", x_svd)
print("Solution norm:", norm(x_svd))
print("Error:", norm(A.dot(x_svd)-b))
print("\n")
print("LS solution using QR:", x_qr)
print("Solution norm:", norm(x_qr))
print("Error:", norm(A.dot(x_qr)-b))
print("\n")


#########################
# 2. Graphics compression
#########################

#Let's load the images

# An image without text is going to be a black and white panda
image1 = imageio.imread('panda_image.jpg')

# And an image with text is going to be the alphabet
image2 = imageio.imread('Alphabet_new.jpg')

Image1 = image1[:,:,1]
Image2 = image2[:,:,1]

#--------------------------------------------------------------------#

def compress_image(matrix, output_prefix='compressed'):
    U, sigma, V = np.linalg.svd(matrix)
    rank= [1, 5, 25, 50, 100]

    for i in rank:
        A = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
        
        # Relative error
        relative_error = np.sum(sigma[i:]**2) / np.sum(sigma**2)
        
        # Create new files with the name and Frobenius norm captured in it, to differentiate
        percentage_captured = np.linalg.norm(A) / np.linalg.norm(matrix)

        # Save the compressed image with a name reflecting the percentage captured
        output_filename = f"{output_prefix}_rank_{i}_capture_{percentage_captured:.2f}.jpg"
        imageio.imwrite(output_filename, np.clip(A, 0, 255).astype(np.uint8))
    
        # Write and Save
        #imageio.imwrite(path, np.clip(A, 0, 255).astype(np.uint8))
    
        # Print information
        print(f"Rank {i} - Percentage Captured: {percentage_captured:.2f}% - Relative Error: {relative_error:.4f}")

#--------------------------------------------------------------------#

# Apply the function to our images
print("Results for Image1 (panda_image.jpg):")
compress_image(Image1, output_prefix='panda_compressed')
print("\n")
print("Results for Image2 (Alphabet_image.jpg):")
compress_image(Image2, output_prefix='alphabet_compressed')        


#################################
# 3. Principal Component Analysis
#################################

# First we will read the datasets

# Function that read data from exaple.dat
def read_txt():
    X = np.genfromtxt('example.dat', delimiter = ' ')
    return X.T

# Function that read data from the csv file
def read_csv():
    X = np.genfromtxt('RCsGoff.csv', delimiter = ',')
    # Get rid of unnecessary variables
    return X[1:,1:].T 

#--------------------------------------------------------------------#

# We will creat a function that apply PCA analysis
def PCA(matrix_choice, file_choice):
    
    # Choose the data
    if file_choice == 1:# text dataset
        X = read_txt()
    else:# csv dataset
        X = read_csv()
    
    # Substract the mean
    X = X - np.mean(X, axis = 0)
    n = X.shape[0]
    # Choose the matrix and complete the program
    if matrix_choice == 1:# covariance matrix
        Y = (1 / (np.sqrt( n - 1))) * X.T
        U,S,VH = np.linalg.svd(Y, full_matrices = False)
        
        # Portion of the total variance accumulated in each of the PC
        total_var = S**2 / np.sum(S**2)
        # Standard deviation of each of the PC
        # Observe that the matrix V contains the eigenvectors of Cx
        standard_dev = np.std(VH, axis = 0)
        
        # Expression of the original dataset in the new PCA coordinates
        new_expr_PCA_coord = np.matmul(VH,X).T
    else:# correlation matrix
        X = (X.T / np.std(X, axis = 1)).T
        Y = (1 / (np.sqrt( n - 1))) * X.T
        U,S,VH = np.linalg.svd(Y, full_matrices = False)
        
        # Portion of the total variance accumulated in each of the PC
        total_var = S**2 / np.sum(S**2)
        
        # Standard deviation of each of the PC
        # Observe that the matrix V contains the eigenvectors of Cx
        standard_dev = np.std(VH.T, axis = 0)
        
        # Expression of the original dataset in the new PCA coordinates
        new_expr_PCA_coord = np.matmul(VH,X).T
    return total_var, standard_dev, new_expr_PCA_coord, S

#--------------------------------------------------------------------#

# Also, we are creating the following functions for later use
# Ploting the Scree Plot
# Using Kaiser method 
# Using the 3/4 rule

def Scree_plot(S,number_figure,matrix_type):
    if matrix_type == 1:#covariance matrix
        plt.figure(number_figure)
        plt.plot(range(len(S)), S)
        for i in range(len(S)):
            plt.scatter(i,S[i],color='purple')
        plt.title('Scree plot for the covariance matrix')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.savefig("scree_plot_cov.jpg")
        plt.show()
    else:#correlation matrix
        plt.figure(number_figure)
        plt.plot(range(len(S)), S)
        for i in range(len(S)):
            plt.scatter(i,S[i],color='purple')
        plt.title('Scree plot for the correlation matrix')
        plt.xlabel('Principal Components')
        plt.ylabel('Eigenvalues')
        plt.savefig("scree_plot_corr.jpg")
        plt.show()    

def Kasier(S):
    count = 0
    for i in range(len(S)):
        if S[i]>1:
            count += 1
    return count

def rule_34(var):
    total_var = sum(var)
    new_var = []
    i = 0
    
    while sum(new_var) < 3*total_var/4:
        new_var.append(var[i])
        i += 1

    return len(new_var)

#--------------------------------------------------------------------#

# Analysis of the fist dataset: "example.dat"

# Covariance matrix   
print('Covariance matrix')
total_var,standar_dev,new_expr,S = PCA(1,1)
print('\n')
print('Accumulated total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,1,1)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var))
print('\n')

#--------------------------------------------------------------------#

# Correlation matrix 
print('Correlation matrix')
total_var,standar_dev,new_expr,S = PCA(0,1)
print('\n')
print('Accumulated total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,2,0)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var)) 
print('\n')      

#--------------------------------------------------------------------#

# Analysis of the second dataset: "RCsGoff.csv"

# Covariance matrix   
print('Covariance matrix')
total_var,standar_dev,new_expr,S = PCA(1,0)
print(new_expr.shape)
print('\n')
print('Accumulated total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,3,1)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var))
print('\n')

#--------------------------------------------------------------------#

# Save results to a file
# Call the read_csv function
X_RCsGoff = read_csv()  

# Convert NumPy arrays to DataFrame
data_df = pd.DataFrame(data=new_expr[:20, :].T, columns=[f"PC{i}" for i in range(1, 21)])
variance_df = pd.DataFrame(data=reshape(total_var, (20, 1)), columns=["Variance"])

# Assuming 'gene' column exists, drop it
if 'gene' in data_df.columns:
    data_df = data_df.drop('gene', axis=1)

# Assuming 'Sample' is the index
data_df.index.name = "Sample"

# Add the variance column to the DataFrame
data_df["Variance"] = variance_df["Variance"]

# Save to a text file
data_df.to_csv("rcsgoff_covariance.txt", sep='\t')

#--------------------------------------------------------------------#

# Correlation matrix 
print('Correlation matrix')
total_var,standar_dev,new_expr,S = PCA(0,0)
print(new_expr.shape)
print('\n')
print('Accumulated total variance in each principal component: ',total_var)
print('\n')
print('Standard deviation of each principal component: ',standar_dev)
print('\n')
print('PCA coordinates of original dataset: ',new_expr)
Scree_plot(S,4,0)
print('\n')
print('Kasier rule:',Kasier(S))
print('3/4 rule:',rule_34(total_var)) 
print('\n') 

#--------------------------------------------------------------------#

# Save results to a file
# Call the read_csv function
X_RCsGoff = read_csv()  

# Convert NumPy arrays to DataFrame
data_df = pd.DataFrame(data=new_expr[:20, :].T, columns=[f"PC{i}" for i in range(1, 21)])
variance_df = pd.DataFrame(data=reshape(total_var, (20, 1)), columns=["Variance"])

# Assuming 'gene' column exists, drop it
if 'gene' in data_df.columns:
    data_df = data_df.drop('gene', axis=1)

# Assuming 'Sample' is the index
data_df.index.name = "Sample"

# Add the variance column to the DataFrame
data_df["Variance"] = variance_df["Variance"]

# Save to a text file
data_df.to_csv("rcsgoff_correlation.txt", sep='\t')