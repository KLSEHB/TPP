import numpy as np
import torch
import Embeddings
import Utils
from scipy import stats


def to_torchLSTM(data):
    """
    Convert dictionary data into a PyTorch tensor for LSTM input.
    
    Parameters:
        data : dict
            Mapping from node -> feature matrix (list of lists)
    
    Returns:
        torch_data : torch.Tensor
            Tensor of shape (num_nodes, input_size, seq_len)
    """
    # Automatically infer input dimension and sequence length
    sample_matrix = next(iter(data.values()))
    input_size, seq_len = len(sample_matrix), len(sample_matrix[0])

    # Initialize an empty tensor with shape (len(data), input_size, seq_len)
    torch_data = torch.empty(len(data), input_size, seq_len, dtype=torch.float32)

    # Fill the tensor with feature matrices
    for idx, matrix in enumerate(data.values()):
        torch_data[idx] = torch.tensor(matrix, dtype=torch.float32)

    return torch_data


def nodesRank(rank):
    """
    Convert a node ranking list into numerical ranks.
    
    Parameters:
        rank : list
            List of node identifiers ordered by predicted importance.
    
    Returns:
        re : list
            Numerical rank list corresponding to the input order.
    """
    sorted_rank = sorted(rank)
    indices = []
    for node in sorted_rank:
        indices.append(rank.index(node))
    return indices


def order(result):
    """
    Sort dictionary items by value in ascending order and assign ranks.
    
    Parameters:
        result : dict
            Key = node, Value = score.
    
    Returns:
        list
            Rank list for all nodes.
    """
    n = len(result) - 1
    for idx, (k, v) in enumerate(sorted(result.items(), key=lambda x: x[1], reverse=False)):
        result[k] = n - idx
    return list(result.values())


def tau_TPP(G, L, sir_list, TPP, C, T):
    """
    Compute Kendall’s tau correlation between TPP predictions and SIR results.
    
    Parameters:
        G : networkx.Graph
            Input network.
        L : int
            Sequence length.
        sir_list : list of dict
            List of node infection results from multiple SIR simulations.
        TPP : torch.nn.Module
            Trained TPP model.
        C : float
            Scaling coefficient for propagation probability.
        T : int
            Time step for feature extraction.
    
    Returns:
        tau_list : list of float
            Kendall’s tau correlation for each SIR run.
    """
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TPP = TPP.to(device)

        nodes = list(G.nodes())
        data_DD = Embeddings.mainTPP_T(G, L, C, T)
        DD = to_torchLSTM(data_DD).to(device)

        # Sort nodes by predicted score from TPP
        lstm_pred = [i for i, j in sorted(dict(zip(nodes, TPP(DD).to('cpu'))).items(), key=lambda x: x[1], reverse=True)]
        self_rank = np.array(nodesRank(lstm_pred), dtype=float)

        tau_list = []

        # Compute Kendall's tau between TPP and each SIR simulation
        for sir in sir_list:
            sir_sort = [i for i, j in sorted(sir.items(), key=lambda x: x[1], reverse=True)]
            sir_rank = np.array(nodesRank(sir_sort), dtype=float)
            tau_value, _ = stats.kendalltau(self_rank, sir_rank)
            tau_list.append(tau_value)

        return tau_list


def mi_TPP(G, L, TPP, C, T):
    """
    Compute mutual information (MI) for TPP-predicted node scores.
    
    Parameters:
        G : networkx.Graph
            Input network.
        L : int
            Sequence length.
        TPP : torch.nn.Module
            Trained TPP model.
        C : float
            Coefficient for propagation rate.
        T : int
            Time step for feature extraction.
    
    Returns:
        TPP_mi : float
            Mutual information score computed from TPP outputs.
    """
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TPP = TPP.to(device)

        nodes = list(G.nodes())
        data_DD = Embeddings.mainTPP_T(G, L, C, T)
        DD = to_torchLSTM(data_DD).to(device)

        TPP_outputs = dict(zip(nodes, TPP(DD).to('cpu')))
        TPP_mi = Utils.compute_mi(TPP_outputs)

        return TPP_mi


def jac_TPP(G, L, sir_list, TPP, top_percentage, C, T):
    """
    Compute average Jaccard similarity between TPP-predicted nodes 
    and SIR-infected nodes across multiple simulations.
    
    Parameters:
        G : networkx.Graph
            Input network.
        L : int
            Sequence length.
        sir_list : list of dict
            Infection results from SIR simulations.
        TPP : torch.nn.Module
            Trained TPP model.
        top_percentage : list[float]
            Top percentages of nodes to evaluate (e.g., [0.01, 0.05, 0.1]).
        C : float
            Coefficient for propagation rate.
        T : int
            Time step for feature extraction.
    
    Returns:
        jaccard_scores : list[float]
            Average Jaccard similarity for each top percentage.
    """
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TPP = TPP.to(device)

        nodes = list(G.nodes())
        data_DD = Embeddings.mainTPP_T(G, L, C, T)
        DD = to_torchLSTM(data_DD).to(device)

        # Sort nodes by predicted TPP score
        lstm_pred = [i for i, j in sorted(dict(zip(nodes, TPP(DD).to('cpu'))).items(), key=lambda x: x[1], reverse=True)]
        jaccard_scores = []

        # Compute average Jaccard similarity for each top threshold
        for top in top_percentage:
            jaccard_mean_list = []
            for sir in sir_list:
                sir_sort = [i for i, j in sorted(sir.items(), key=lambda x: x[1], reverse=True)]
                jaccard_value = Utils.Jaccard(lstm_pred, sir_sort, top)
                jaccard_mean_list.append(jaccard_value)

            jaccard_scores.append(np.mean(jaccard_mean_list))

        return jaccard_scores
