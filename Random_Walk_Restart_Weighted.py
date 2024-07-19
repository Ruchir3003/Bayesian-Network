# This script is to run a random walk with restart on a weighted graph (edges are weighted). 

# Inputs : a) Inital probability vector b) Edge Weights c) Steps os the random walk d) Restart Probability 
# Output : Final Probability vector - Numpy array

import numpy as np

def random_walk_restart_weighted (initial_prob, edge_weights, prob_restart, steps=10e5 ):
    po = initial_prob # initial probability vector
    p_final = np.array(len(po)) # Final probability vector
    transition_matrix = np.zeros((len(edge_weights),len(edge_weights)))
    edge_weight_sum = 0 # Sum of edge weights of a particular node
    for i in range(len(transition_matrix)):
        edge_weight_sum=0
        for j in range(len(transition_matrix)):
            edge_weight_sum+=edge_weights[i][j]
        for k in range(len(transition_matrix)):
            if edge_weight_sum==0:
                transition_matrix[i][k] = 0 # In case of no edges
            else: 
                transition_matrix[i][k] = edge_weights[i][k]/edge_weight_sum # Transition matrix calculated as per https://home.icts.res.in/~rbasu/TIDP(1).pdf
    pt=po  # pt is the probability vector at t time - initialised with initial probability vector (po)
    count=0 # Number of iterations
    while True : 
        pt_1= ((transition_matrix.transpose())@pt)*(1-prob_restart) + prob_restart*po # pt+1 is the probability vector at t+1 time
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