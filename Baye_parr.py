import cmapPy
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress.parse import parse # Package needed to read gctx files
import networkx as nx
import json
import matplotlib.pyplot as plt
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.models import BayesianModel
from pgmpy.estimators import BDeuScore
import warnings
warnings.filterwarnings('ignore')
import multiprocessing

data = pd.read_pickle('discrete_data.pkl')


# Function to learn the structure of a Bayesian network for a given chunk of data
def learn_structure(chunk):
    hc = HillClimbSearch(chunk)
    return hc.estimate()

def process_chunk(chunk):
    # Learn the structure for each chunk
    return learn_structure(chunk)

def main():
    # Define the chunk size
    chunk_size = 10000

    df=data

    # Initialize multiprocessing pool with 20 processes
    pool = multiprocessing.Pool(processes=30)

    # List to store the learned structures
    learned_structures = []

    # Process data in chunks
    for chunk_start in range(0, len(df), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk = df.iloc[chunk_start:chunk_end]
        # Apply the process_chunk function to each chunk in parallel
        result = pool.apply_async(process_chunk, args=(chunk,))
        learned_structures.append(result)

    # Close the multiprocessing pool
    pool.close()
    pool.join()

    # Get the results from the asynchronous processes
    learned_structures = [result.get() for result in learned_structures]

    # Combine results from parallel processes
    combined_edges = set()
    for structure in learned_structures:
        combined_edges.update(structure.edges())

    # Instantiate a BayesianModel object with the combined edges
    bayesian_network = BayesianModel(list(combined_edges))

    # Print the edges of the Bayesian network
    print("Edges of the Bayesian Network:")
    print(bayesian_network.edges())

if __name__ == "__main__":
    main()
# Store the Bayesian model object using pickle
with open('bayesian_model.pkl', 'wb') as f:
    pickle.dump(bayesian_network, f)