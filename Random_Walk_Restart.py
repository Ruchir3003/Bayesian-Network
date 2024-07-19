# This code runs the random walk with restart algorithm. 
# The formula for random walk is derived from https://www.cell.com/ajhg/pdf/S0002-9297(08)00172-9.pdf

# Input : The input parameters are : 
# a) Initial probability vector - Numpy Array
# b) Adjacency matrix - Numpy Array
# c) Probability of restart - Float 

# Output : Final Probability vector - Numpy array
import numpy as np

def random_walk_restart(initial_prob,adjacency_matrix,prob_restart,steps=10e5):
    po = initial_prob # initial probability vector
    p_final = np.array(len(po)) # Final probability vector
    A = adjacency_matrix # adjacency matrix of the network 
    A_shape = A.shape
    A_rows=A_shape[0]
    A_columns=A_shape[1]
    D=np.zeros((A_rows,A_columns))   # D is the diagonal matrix as per https://arxiv.org/pdf/2008.03639.pdf
    if(A_rows == A_columns):
        for i in range(A_rows):
            sum_row = 0
            for j in range(A_columns):
                sum_row+=A[i][j]
            if(sum_row==0):
                 D[i][i]=0      # In case of no edges
            else: 
                D[i][i]=1/sum_row
        M=D@A
        pt=po  # pt is the probability vector at t time - initialised with initial probability vector (po)
        count=0 # Number of iterations
        while True : 
            pt_1= ((M.transpose())@pt)*(1-prob_restart) + prob_restart*po # pt+1 is the probability vector at t+1 time
            if (np.linalg.norm((pt_1 - pt), ord=1)<10e-6):
                 # We use L1 norm to create a threshold for convergence - https://www.cell.com/ajhg/pdf/S0002-9297(08)00172-9.pdf
                p_final=pt_1
                return(p_final)
                break
            if(count>steps-1): # Stop the loop after 10e5 iterations
                return(pt_1)
                break
            pt=pt_1
            count+=1
    else:
        print("Invalid input")