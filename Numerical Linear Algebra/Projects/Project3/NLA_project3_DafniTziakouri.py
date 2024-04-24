# Dafni Tzakouri
# NLA_Project 3
# PageRank implementations

# We created functions for each method 
# (storing the matrices or not storing the matrices)
# Also, functions to create the neccessary matrices 


# Import necessary packages:
import numpy as np
import time
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


#######################################################################


# Function to create a D matrix
def Matr_G(matrix):
    # Ensure division by zero or NaN results in 0
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate the reciprocal of the sum of each column
        G = np.divide(1., np.asarray(matrix.sum(axis=0)).reshape(-1))
    # Replace NaN values with 0
    np.ma.masked_array(G, ~np.isfinite(G)).filled(0)
    # Create a sparse diagonal matrix from the calculated values
    return sp.diags(G)


#######################################################################


# Function: Power Method Storing Matrices
def PowerMethod_Storing(G, m=0.15, Tol=1e-05):
    # Number of nodes in the graph
    n = G.shape[0]
    # Record the starting time for performance measurement
    start = time.time()
    # Initialize a vector of ones
    e = np.ones((n, 1))
    # Initialize the vector representing the probability of jumping to each node
    Zj = []

    # Calculate the probability of jumping for each node
    for i in range(n):
        # Count the number of non-zero elements in the i-th column
        cont = G[:, i].count_nonzero()
        if cont > 0:
            # If there are non-zero elements, assign m/n
            Zj.append(m / n)
        else:
            # If the column is all zeros, assign 1/n
            Zj.append(1. / n)

    # Convert the list to a NumPy array for efficiency
    Zj_arr = np.asarray(Zj)
    # Initialize the PageRank vector
    Xk = np.ones((n, 1))
    # Initialize the previous iteration's PageRank vector
    Xk_1 = np.ones((n, 1)) / n

    # Iteratively update the PageRank vector until convergence
    while np.linalg.norm(Xk_1 - Xk, np.inf) > Tol:
        Xk = Xk_1
        # Calculate the vector representing the probability of jumping to each node
        Zxk = Zj_arr.dot(Xk)
        # Update the PageRank vector using the power iteration formula
        Xk_1 = (1 - m) * G.dot(Xk) + e * Zxk

    # Print the time consumption for the storing matrices method
    print("Time consumption when storing matrices: ", time.time() - start, "seconds")
    # Normalize the final PageRank vector and return
    Xk_1 = Xk_1 / np.sum(Xk_1)
    return Xk_1


#######################################################################


# Function to create an L matrix
def Matr_L(nrows, ncols):
    # Initialize a dictionary to store indices of non-zero elements in each column
    ind = {}
    # Iterate through the non-zero elements of the matrix
    for i in range(len(ncols)):
        ind_col = ncols[i]
        # If the column index is already in the dictionary, append the row index to the existing array
        if ind_col in ind:
            ind[ind_col] = np.append(ind[ind_col], nrows[i])
        else:
            # If the column index is not in the dictionary, create a new array with the row index
            ind[ind_col] = np.asarray([nrows[i]])
    # Return the dictionary containing column indices and corresponding row indices
    return ind


#######################################################################


# Function: Power Method Not Storing Matrices
def PowerMethod_NotStoring(matrix, m, Tol):
    # Number of nodes in the graph
    n = matrix.shape[0]
    # Record the starting time for performance measurement
    start = time.time()
    # Create a dictionary to store column indices and corresponding row indices of non-zero elements
    ind = Matr_L(matrix.nonzero()[0], matrix.nonzero()[1])
    # Initialize the PageRank vector
    x = np.ones((n, 1)) / n
    # Initialize the previous iteration's PageRank vector
    xc = np.ones((n, 1))

    # Iteratively update the PageRank vector until convergence
    while np.linalg.norm(x - xc, np.inf) > Tol:
        xc = x
        # Initialize a vector of zeros for the new iteration
        x = np.zeros((n, 1))

        # Update the new iteration vector based on the non-stored matrix multiplication
        for j in range(n):
            if j in ind:
                # If the column index is in the dictionary, update the corresponding rows
                if len(ind[j]) != 0:
                    x[ind[j]] = x[ind[j]] + xc[j] / len(ind[j])
                else:
                    x = x + xc[j] / n
            else:
                x += xc[j] / n

        # Update the PageRank vector using the power iteration formula
        x = (1 - m) * x + m / n

    # Print the time consumption for the not storing matrices method
    print("Time consumption when not storing matrices: ", time.time() - start, "seconds")
    # Normalize the final PageRank vector and return
    return x / np.sum(x)


#######################################################################


# Result Printing: Selection of the algorithm parameters

# Read the data file and initialize matrices
matr = sio.mmread("Data/p2p-Gnutella30.mtx")
Matr = sp.csr_matrix(matr)
D = Matr_G(Matr)
A = sp.csr_matrix(Matr.dot(D))

# User input for damping factor and tolerance
DampFact = float(input("\nDefine the Damping factor (Mandatorily between 0 and 1 (0.15 is recommended))\n"))

# Validate damping factor input
if not (0 < DampFact < 1):
    DampFact = 0.15
    print("Damping factor set to default: 0.15")

Tol = float(input("\nDefine the Tolerance (Mandatorily between 1e-04 and 1e-10 (1e-05 is recommended))\n"))

# Validate tolerance input
if not (1e-11 < Tol < 1e-04):
    Tol = 1e-05
    print("Tolerance set to default: 1e-05")


#######################################################################


# Initialize variable for method selection
Method = -1

# Loop until a valid method is selected (1 or 2)
while Method not in [1, 2]:
    # Prompt user to choose a method
    Method = int(input("\nChoose the method to use:\n 1. Power Method Storing Matrices\n 2. Power Method Not Storing Matrices \n"))

    # Check the selected method and perform the corresponding computation
    if Method == 1:
        print("\nOutput:")
        # Call the PowerMethod_Storing function with specified parameters
        PR = PowerMethod_Storing(A, DampFact, Tol)
        print("Normalized PR Vector Storing Matrices:\n", np.round(PR, 6))

    elif Method == 2:
        print("\nOutput:")
        # Call the PowerMethod_NotStoring function with specified parameters
        PR = PowerMethod_NotStoring(A, DampFact, Tol)
        print("Normalized PR Vector Not Storing Matrices:\n", np.round(PR, 6))

    else:
        # Print a message if an invalid input is provided
        print("Input 1 or 2")    