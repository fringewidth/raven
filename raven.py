import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx

def minmax(x: float, new_min: float, new_max: float, old_min: float, old_max: float) -> float:
    """
        Applies min-max normalisation

        Args:
            x (float): The value to be normalised
            new_min (float): The minimum value of the new range
            new_max (float): The maximum value of the new range
            old_min (float): The minimum value of the old range
            old_max (float): The maximum value of the old range
        
        Returns:
            x (float): The normalised value
    """

    return (x - old_min) * (new_max - new_min) / (old_max - old_min) + new_min


def make_graph(r_squared: np.ndarray, tau: float) -> nx.Graph:
    """
    Creates a graph from the R^2 matrix.
    
    Args:
        r_squared (np.ndarray): The R^2 matrix.
        tau (float): The threshold of correlation for an edge to be present between two features.

    Returns:
        G (nx.Graph): The correlation graph

    Raises:
        ValueError: If r_squared is not a square matrix.
    """
    if r_squared.shape[0] != r_squared.shape[1]:
        raise ValueError("r_squared must be a square matrix")
    
    G = nx.Graph()
    for i in range(r_squared.shape[0]):
        for j in range(r_squared.shape[1]):
            if r_squared[i, j] > tau:
                G.add_edge(i, j, weight=minmax(r_squared[i, j], 0.5, 1, tau, 1))
    
    return G


def raven(df: pd.DataFrame, tau: float = 0.95, sample_size: int = 100) -> list:
    """
    Implements the Raven algorithm that identifies redundant features in a dataset.

    Args: 
        df (pd.DataFrame): Independent features in the dataset.
        tau (float): The threshold value for the correlation coefficient. Default is 0.95.
        sample_size (int): The number of samples to use for the calculation. Default is 100.
    
    Returns:
        redundant(list): Names of the redundant features.

    Raises
        ValueError: If any of the input params are invalid.
    """
    # Validate tau
    if tau <= 0 or tau >= 1:
        raise ValueError("tau must be greater than 0 and lesser than 1")
    
    # Validate Sample Size
    if sample_size < 1:
        raise ValueError("sample_size must be greater than 0")
    if sample_size > len(df):
        raise ValueError("sample_size must be lesser than the number of rows in the DataFrame")
    
    # Validate DataFrame
    if len(df.columns) < 2:
        raise ValueError("DataFrame must have at least 2 columns")
    if not all(df.dtypes == np.number):
        raise ValueError("DataFrame must contain only numeric columns")
    
    # Convert to numpy array
    data = df.to_numpy()

    # sample data
    sample = data[np.random.choice(data.shape[0], sample_size),:]

    # Create pairs
    pairs = np.array(list(combinations(range(sample.shape[1]), 2)))

    # calculate R^2 for each pair of features
    r_squared = np.empty((sample.shape[1], sample.shape[1]))
    for first, second in pairs:
        cov_mat = np.cov(sample[:, [first, second]], rowvar=False)
        if all(cov_mat[i, i] != 0 for i in range(2)):
            r_squared[first, second] = cov_mat[0, 1] ** 2 / (cov_mat[0, 0] * cov_mat[1, 1])
        else:
            r_squared[first, second] = 0
    
    # Create graph and extract components
    G = make_graph(r_squared, tau)

    connected_components = list(nx.connected_components(G))

    essential_ind = np.empty((len(connected_components),), dtype=int)

    for i, component in enumerate(connected_components):
        subgraph = G.subgraph(component)
        max_degree_node, _ = max(subgraph.degree, key=lambda x: x[1])
        essential_ind[i] = max_degree_node
    
    redundant = [df.columns[i] for i in essential_ind]

    return redundant


data = pd.read_csv('train1000.csv').drop(columns=['sample_id', 'Unnamed: 0']).iloc[:, :556]

print(raven(data))