from itertools import combinations
import numpy as np
import networkx as nx
import openml
import pandas as pd

def raven(data, sample_size=50, tau=0.95, target_col=None, verbose=True):
    """
    Implements the Raven algorithm that identifies redundant features in a dataset.

    Args: 
        data (pd.DataFrame | int | str): DataFrame or OpenML dataset ID/name.
        tau (float): Threshold for correlation coefficient. Default is 0.95.
        sample_size (int): Number of samples to use. Default is 50.
        target_col (str, optional): Target column to drop.
    Returns:
        essential (list): Names of selected (non-redundant) features.
        redundant (list): Names of redundant features.
    """

    # --- Load dataset from OpenML if ID or name given ---
    if isinstance(data, (int, str)):
        dataset = openml.datasets.get_dataset(data)
        df, *_ = dataset.get_data(dataset_format="dataframe")
        if target_col is None and dataset.default_target_attribute:
            target_col = dataset.default_target_attribute
        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])
        dataset_name = dataset.name
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        dataset_name = "Custom DataFrame"
    else:
        raise ValueError("Input must be a pandas DataFrame or an OpenML dataset ID/name.")

    # --- Keep only numeric columns ---
    df = df.select_dtypes(include=[np.number])
    total_features = len(df.columns)

    # --- Validate parameters ---
    if tau <= 0 or tau >= 1:
        raise ValueError("tau must be greater than 0 and lesser than 1")
    if sample_size < 1:
        raise ValueError("sample_size must be greater than 0")
    if sample_size > len(df):
        raise ValueError("sample_size must be lesser than the number of rows in the DataFrame")
    if total_features < 2:
        raise ValueError("DataFrame must have at least 2 numeric columns")

    # --- Convert to numpy sample ---
    sample = df.sample(sample_size, random_state=42).to_numpy()
    r2_scores = {}
    col_idx = {col: df.columns.get_loc(col) for col in df.columns}

    # --- Compute R^2 between feature pairs ---
    for first, second in combinations(df.columns, 2):
        f_i, s_i = col_idx[first], col_idx[second]
        cov = np.cov(sample[:, f_i], sample[:, s_i])
        r2_scores[first, second] = cov[1, 0]**2 / (cov[1, 1] * cov[0, 0]) if np.all(cov) else 0

    # --- Build correlation graph ---
    def make_graph(scores, tau):
        def get_weight(r2): return (r2 - tau)/(1 - tau) * 0.5 + 0.5
        G = nx.Graph()
        for (a, b), r2 in scores.items():
            if r2 >= tau:
                G.add_edge(a, b, weight=get_weight(r2))
        return G

    G = make_graph(r2_scores, tau)
    del sample, r2_scores

    # --- Identify essential and redundant features ---
    essential = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        max_deg_node, _ = max(sub.degree(), key=lambda x: x[1])
        essential.append(max_deg_node)

    redundant = [node for node in G.nodes() if node not in essential]
    
    connected_features = set(G.nodes())
    all_features = set(df.columns)
    isolated = list(all_features - connected_features)
    essential = essential + isolated
    return essential, redundant
