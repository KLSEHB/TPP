import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
from tqdm import tqdm

# Set random seed
def setup_seed(seed):
    """Fix random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ---------------- SIR Model ----------------
def SIR(G, infected, beta=0.1, miu=1):
    """
    SIR epidemic simulation for a given initial infected set.
    
    Parameters:
        G : networkx.Graph
            Input network.
        infected : list
            Initially infected nodes.
        beta : float
            Infection probability.
        miu : float
            Recovery probability.
    
    Returns:
        Average infected size after 1000 simulations.
    """
    N = 1000
    total_infected = 0
    
    while N > 0:
        inf = set(infected)  # currently infected nodes
        R = set()            # recovered nodes
        while inf:
            newInf = []
            for i in inf:
                for j in G.neighbors(i):
                    if random.uniform(0,1) < beta and j not in inf and j not in R:
                        newInf.append(j)
                # Recovery check
                if random.uniform(0,1) <= miu:
                    R.add(i)
                else:
                    newInf.append(i)
            inf = set(newInf)
        total_infected += len(R) + len(inf)
        N -= 1
    return total_infected / 1000.0

def SIR_dict(G, beta=0.1, miu=1, real_beta=None, a=1.5):
    """
    Compute SIR results for all nodes in the network.
    
    Parameters:
        G : networkx.Graph
        beta : float
        miu : float
        real_beta : bool, optional
        a : float, scaling factor for beta if real_beta is True
    
    Returns:
        Dictionary mapping node -> average infection size.
    """
    node_list = list(G.nodes())
    SIR_dic = {}
    if real_beta:
        dc_list = np.array(list(dict(G.degree()).values()))
        beta = a * (float(dc_list.mean()) / (float((dc_list**2).mean()) - float(dc_list.mean())))

    print('beta:', beta)
    for node in tqdm(node_list):
        sir = SIR(G, infected=[node], beta=beta, miu=miu)
        SIR_dic[node] = sir

    return SIR_dic

def save_sir_dict(dic, path):
    """
    Save SIR results to CSV.
    
    Parameters:
        dic : dict
            Node -> SIR value.
        path : str
            Target CSV path.
    """
    df = pd.DataFrame({'Node': list(dic.keys()), 'SIR': list(dic.values())})
    df.to_csv(path, index=False)

def SIR_betas(G, a_list, root_path):
    """
    Run SIR for different beta scaling factors.
    
    Parameters:
        G : networkx.Graph
        a_list : list of float
            Multipliers for beta.
        root_path : str
            Path to save CSV files.
    
    Returns:
        List of SIR dictionaries for each beta.
    """
    sir_list = []
    for idx, a in enumerate(a_list):
        sir_dict = SIR_dict(G, real_beta=True, a=a)
        sir_list.append(sir_dict)
        path = root_path + str(idx) + '.csv'
        save_sir_dict(sir_dict, path)
    return sir_list

# ---------------- TPP DataLoader ----------------
def Get_TPP_DataLoader(data, label, batch_size ,seq_len):
    """
    Create a LSTM DataLoader, ensuring the feature tensor has shape (batch_size, seq_len, L).

    Parameters:
        data: dict
            Dictionary mapping each node to its feature matrix.
        label: dict
            Dictionary mapping each node to its label.
        batch_size: int
            Size of each batch.

    Returns:
        loader: DataLoader
            PyTorch DataLoader for batch training.
    """
    # Get the feature matrix shape
    _, input_size, seq_len = len(data), seq_len, len(next(iter(data.values()))[0])

    # Construct feature tensor of shape (num_samples, seq_len, L)
    torch_set = torch.empty(len(data), input_size, seq_len, dtype=torch.float32)
    for inx, matrix in enumerate(data.values()):
        torch_set[inx] = torch.tensor(matrix, dtype=torch.float32)

    # Construct label tensor of shape (num_samples, 1)
    sir_torch = torch.tensor(list(label.values()), dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    dataset = TensorDataset(torch_set, sir_torch)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return loader


def train_model(loader, model, num_epochs, lr, path=None):
    """
    Train an LSTM model on the given DataLoader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        count = 0
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device).float()
            pred = model(data)
            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        avg_loss = running_loss / count
        loss_list.append(avg_loss)

    return model, loss_list

# ---------------- Evaluation ----------------
def compute_mi(ranking):
    """
    Compute Monotonicity Index (MI) for node ranking.
    """
    if isinstance(ranking, dict):
        ranking_values = [v.item() if isinstance(v, torch.Tensor) else v for v in ranking.values()]
    elif isinstance(ranking, list):
        ranking_values = ranking
    else:
        raise ValueError("ranking must be dict or list")

    unique, counts = np.unique(ranking_values, return_counts=True)
    N_Z = len(ranking_values)
    sum_Nw = np.sum(counts * (counts - 1))
    MI = (1 - (sum_Nw / (N_Z * (N_Z - 1)))) ** 2 if N_Z > 1 else 1
    return MI

def Jaccard(pred_rank, true_rank, top_percentage=100):
    """
    Compute Jaccard Index between predicted and true ranking.
    """
    N = min(len(pred_rank), len(true_rank))
    if N == 0:
        return 0.0
    top_k = max(1, int(N * (top_percentage / 100)))
    pred_top_k = set(pred_rank[:top_k])
    true_top_k = set(true_rank[:top_k])
    intersection_size = len(pred_top_k & true_top_k)
    union_size = len(pred_top_k | true_top_k)
    return intersection_size / union_size if union_size > 0 else 0.0

# ---------------- Graph Utils ----------------
def load_graph(path):
    """Load graph from edge list."""
    return nx.read_edgelist(path, create_using=nx.Graph())

def load_sir_list(path, num_files=10):
    """Load multiple SIR CSV results."""
    sir_list = []
    for i in range(num_files):
        df = pd.read_csv(path + str(i) + '.csv')
        sir_list.append(dict(zip(df['Node'].astype(str), df['SIR'])))
    return sir_list

def precompute_degrees(graph):
    """Precompute node degrees."""
    return dict(graph.degree())

def precompute_neighbor(graph):
    """Precompute node neighbors."""
    neighbor = {}
    for node in graph.nodes():
        neighbor[node] = set(graph.neighbors(node))
    return neighbor

def get_future_features(G, beta, K, order=1):
    """
    Compute node propagation features up to a specified order.
    
    Parameters:
        G : networkx.Graph
            Input network.
        beta : float
            Infection probability.
        K : float
            Scaling factor for feature normalization.
        order : int, optional (default=1)
            Propagation order (1~5). Determines how many propagation steps to include.
    
    Returns:
        dict: Mapping node -> list of features [T1, T2, ..., T_order].
    """
    degree_cache = precompute_degrees(G)      # Cache node degrees
    neighbor_cache = precompute_neighbor(G)   # Cache neighbors for each node
    features = {}

    for node in G.nodes():
        features[node] = compute_T_features(node, beta, neighbor_cache, degree_cache, K, order)

    return features

def compute_T_features(node, beta, neighbor_cache, degree_cache, K, order):
    """
    Compute propagation features for a single node up to the specified order.
    
    Parameters:
        node : hashable
            Target node.
        beta : float
            Infection probability.
        neighbor_cache : dict
            Node -> set of neighbors.
        degree_cache : dict
            Node -> degree.
        K : float
            Scaling factor for feature normalization.
        order : int
            Propagation order (1~5).
    
    Returns:
        list: Feature vector [T1, T2, ..., T_order].
    """
    T = []

    # T1: scaled node degree
    T1 = degree_cache[node]
    T.append(T1 * K)

    if order == 1:
        return T

    # T2: sum of neighbors' degrees scaled by beta
    first_order = neighbor_cache[node]
    T2 = sum(degree_cache[n] for n in first_order) * beta
    T.append(T2 * K)

    if order == 2:
        return T

    # T3: second-order neighbors
    second_order = set().union(*[neighbor_cache[n] for n in first_order]) - {node}
    B2 = {x: 1 - (1 - beta * beta) ** len(neighbor_cache[x].intersection(first_order)) for x in second_order}
    T3 = sum(B2[x] * degree_cache[x] for x in second_order)
    T.append(T3 * K)

    if order == 3:
        return T

    # T4: third-order neighbors
    third_order = set().union(*[neighbor_cache[n] for n in second_order]) - first_order
    B3 = {}
    T4 = 0
    for x in third_order:
        Z = neighbor_cache[x].intersection(second_order)
        np_prod = 1
        for z in Z:
            np_prod *= (1 - B2[z] * beta)
        B3[x] = 1 - np_prod
        T4 += (1 - np_prod) * degree_cache[x]
    T.append(T4 * K)

    if order == 4:
        return T

    # T5: fourth-order neighbors
    fourth_order = set().union(*[neighbor_cache[n] for n in third_order]) - second_order
    T5 = 0
    for x in fourth_order:
        Z = neighbor_cache[x].intersection(third_order)
        np_prod = 1
        for z in Z:
            np_prod *= (1 - B3[z] * beta)
        T5 += (1 - np_prod) * degree_cache[x]
    T.append(T5 * K)

    return T
