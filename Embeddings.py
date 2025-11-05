import numpy as np
import Utils

def mainTPP_T(G, L, C, T=1):
    """
    General mainTPP_T function
    ----------
    G : networkx.Graph
        Input network graph
    L : int
        Sequence length (including the target node and its neighbors)
    C : float
        Coefficient for calculating the infection probability beta
    T : int
        Feature order (1~5)
    ----------
    Returns:
    data_D : dict[node -> feature matrix (list of list)]
    """
    data_D = {}

    # --- Step 1. Network parameters ---
    node_list = list(G.nodes())
    dc_list = np.array(list(dict(G.degree()).values()))
    K_mean = float(dc_list.mean())
    beta = C * (K_mean / (float((dc_list ** 2).mean()) - K_mean))
    K = 1 / K_mean

    # --- Step 2. Compute feature vectors for all nodes ---
    DF = Utils.get_future_features(G, beta, K, T)

    # --- Step 3. Build temporal sequence matrix for each node ---
    for node in node_list:
        matrix = []
        # Get first-order neighbors and sort them by degree (descending)
        one_order = list(G.adj[node])
        sorted_neighbors = sorted(one_order, key=lambda n: G.degree[n], reverse=True)

        # Target node features
        DDF = DF.get(node)
        matrix.append(DDF)

        # Append features of top L-1 neighbors
        for count, neighbor in enumerate(sorted_neighbors):
            if count >= L - 1:
                break
            DDF = DF.get(neighbor)
            matrix.append(DDF)

        # Pad with zeros if sequence length < L
        feature_dim = len(matrix[0])
        if len(matrix) < L:
            matrix.extend([[0] * feature_dim] * (L - len(matrix)))

        # Transpose the matrix so that each row represents one feature dimension
        matrix = list(zip(*matrix))
        data_D[node] = matrix

    return data_D
