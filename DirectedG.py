import Utils
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import os 
import Models
import torch
import Test
from scipy import stats

def save_sir_dict(dic, path):
    """
    Save SIR results as a CSV file.
    Parameters:
        dic: dict, results of SIR simulation {node: influence value}
        path: str, target file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    node = list(dic.keys())
    sir = list(dic.values())
    Sir = pd.DataFrame({'Node': node, 'SIR': sir})
    Sir.to_csv(path, index=False)

def SIR_directed(G, infected, beta=0.1, miu=1):
    """
    Directed SIR model.
    Propagation only follows outgoing edges.
    Input:
        G: original directed network
        infected: initially infected node(s)
        beta: infection probability
        miu: recovery probability
    Return:
        re: average infection size after N simulations
    """
    N = 1000
    re = 0

    while N > 0:
        inf = set(infected)  # initially infected nodes
        R = set()  # recovered nodes
        while len(inf) != 0:
            newInf = []
            for i in inf:
                # spread only along outgoing edges
                for j in G.successors(i):
                    k = random.uniform(0, 1)
                    if (k < beta) and (j not in inf) and (j not in R):
                        newInf.append(j)
                k2 = random.uniform(0, 1)
                if k2 > miu:
                    newInf.append(i)
                else:
                    R.add(i)
            inf = set(newInf)

        re += len(R) + len(inf)
        N -= 1

    return re / 1000.0

def SIR_dict_directed(G, beta=0.1, miu=1, real_beta=None, a=1.5):
    """
    Compute SIR results for all nodes in a directed network.
    Input:
        G: target directed network (nx.DiGraph)
        beta: infection probability (only used when real_beta=False)
        miu: recovery probability
        real_beta: whether to calculate beta using directed threshold formula
        a: scaling factor for infection threshold
    Return:
        SIR_dic: dict, {node: influence value}
    """

    node_list = list(G.nodes())
    SIR_dic = {}

    if real_beta:
        k_in = np.array([G.in_degree(n) for n in G.nodes()], dtype=float)
        k_out = np.array([G.out_degree(n) for n in G.nodes()], dtype=float)

        mean_k_out = np.mean(k_out)
        mean_k_in_k_out = np.mean(k_in * k_out)

        # avoid division by zero
        if mean_k_in_k_out == 0:
            mean_k_in_k_out = 1e-9

        beta = a * (mean_k_out / mean_k_in_k_out)

    print('β (directed threshold) =', beta)

    for node in tqdm(node_list, desc="Simulating SIR"):
        sir = SIR_directed(G, infected=[node], beta=beta, miu=miu)
        SIR_dic[node] = sir

    return SIR_dic

def SIR_betas_directed(G, a_list, root_path, miu=1):
    """
    Perform SIR simulations for different infection thresholds on a directed graph.
    Parameters:
        G: target directed network
        a_list: list of scaling factors for infection probability
        root_path: path prefix for saving results
        miu: recovery probability
    """
    sir_list = []
    for inx, a in enumerate(a_list):
        sir_dict = SIR_dict_directed(G, real_beta=True, a=a, miu=miu)
        sir_list.append(sir_dict)
        path = root_path + str(inx) + '.csv'
        save_sir_dict(sir_dict, path)
    return sir_list



def mainTPP_T4_directed(G, L, C):
    """
    Generate T4 temporal sequence features for each node (directed version).
    Parameters:
        G: nx.DiGraph
        L: sequence length
        C: coefficient used for calculating beta
    Returns:
        data_D: dict, {node: 4xL feature matrix}
    """
    data_D = {}

    node_list = list(G.nodes())
    out_deg_list = np.array(list(dict(G.out_degree()).values()))
    K = float(out_deg_list.mean())
    beta = C * (K / (float((out_deg_list**2).mean()) - K))
    K = 1 / K

    DF = getfutue_T4_directed(G, beta, K)

    for node in node_list:
        matrix = []
        # get outgoing neighbors
        one_order = list(G.successors(node))
        sorted_neighbors = sorted(one_order, key=lambda n: G.out_degree[n], reverse=True)

        # features of current node
        DDF = DF.get(node)
        matrix.append([DDF[0], DDF[1], DDF[2], DDF[3]])

        # features of neighbors
        count = 0
        for neighbor in sorted_neighbors:
            if count >= L - 1:
                break
            DDF = DF.get(neighbor)
            matrix.append([DDF[0], DDF[1], DDF[2], DDF[3]])
            count += 1

        # pad with zeros if insufficient length
        if len(matrix) < L:
            matrix.extend([[0, 0, 0, 0]] * (L - len(matrix)))

        # transpose
        matrix = list(zip(*matrix))
        data_D[node] = matrix

    return data_D

def getfutue_T4_directed(G, beta, K):
    """
    Precompute T4 features for all nodes (directed version).
    """
    d_dict = {}
    node_list = list(G.nodes())
    degreeList = precompute_degrees_directed(G)
    neiList = precompute_neighbor_directed(G)

    for node in node_list:
        d_dict[node] = getTFature_T4_directed(node, beta, neiList, degreeList, K)

    return d_dict

def precompute_degrees_directed(G):
    """Precompute out-degree for each node."""
    return dict(G.out_degree())

def precompute_neighbor_directed(G):
    """Precompute outgoing neighbors for each node."""
    neighbor = {}
    for node in G.nodes():
        neighbor[node] = set(G.successors(node))
    return neighbor

def getTFature_T4_directed(node, beta, neighbor_cache, degree_cache, K):
    """
    Compute T1–T4 features for a node (directed version).
    """
    T1 = degree_cache[node]

    first_order_neighbors = neighbor_cache[node]
    T2 = 0
    B2 = {}
    T3 = 0
    second_order_neighbors = set()

    # second-order neighbors
    for neighbor in first_order_neighbors:
        T2 += degree_cache[neighbor]
        for x in neighbor_cache.get(neighbor, set()):
            if x != node:
                second_order_neighbors.add(x)
                B2[x] = B2.get(x, 1) * (1 - beta * beta)

    T2 *= beta

    B3 = {}
    T4 = 0
    third_order_neighbors = set()
    for neighbor in second_order_neighbors:
        T3 += degree_cache.get(neighbor, 0) * (1 - B2.get(neighbor, 0))
        for x in neighbor_cache.get(neighbor, set()):
            if x not in first_order_neighbors:
                third_order_neighbors.add(x)
                B3[x] = B3.get(x, 1) * (1 - (1 - B2.get(neighbor, 0)) * beta)

    for neighbor in third_order_neighbors:
        T4 += degree_cache.get(neighbor, 0) * (1 - B3.get(neighbor, 0))

    matrix_d = [T1 * K, T2 * K, T3 * K, T4 * K]

    return matrix_d

def compare_TPP_nb(G, L3, sir_list, LSTM, C):
    """
    Compare the ranking correlation (Kendall's τ) between TPP predictions and SIR simulations.
    """
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TPP = LSTM.to(device)

        nodes = list(G.nodes())
        data_DD = mainTPP_T4_directed(G, L3, C)
        DD = Test.to_torchLSTM(data_DD).to(device)

        lstm_pred = [i for i, j in sorted(dict(zip(nodes, TPP(DD).to('cpu'))).items(), key=lambda x: x[1], reverse=True)]
        self_rank = np.array(Test.nodesRank(lstm_pred), dtype=float)

        tau_list = []
        for sir in sir_list:
            sir_sort = [i for i, j in sorted(sir.items(), key=lambda x: x[1], reverse=True)]
            sir_rank = np.array(Test.nodesRank(sir_sort), dtype=float)
            tau0, _ = stats.kendalltau(self_rank, sir_rank)
            tau_list.append(tau0)

        return tau_list

def compare_jaccard_TPP(G, L3, sir_list, LSTM, top_percentage, C=1):
    """
    Compare Jaccard similarity between top-ranked nodes from TPP and SIR results.
    """
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        TPP = LSTM.to(device)

        nodes = list(G.nodes())
        data_DD = mainTPP_T4_directed(G, L3, C)
        DD = Test.to_torchLSTM(data_DD).to(device)

        lstm_pred = [i for i, j in sorted(dict(zip(nodes, TPP(DD).to('cpu'))).items(), key=lambda x: x[1], reverse=True)]
        map = []

        for top in top_percentage:
            mean = []
            for sir in sir_list:
                sir_sort = [i for i, j in sorted(sir.items(), key=lambda x: x[1], reverse=True)]
                mean.append(Utils.Jaccard(lstm_pred, sir_sort, top))
            map.append(np.mean(mean))
        
        return map


def graph_basic_features(G, name="Graph"):
    """
    Print basic statistics of a directed graph.
    """
    print(f"=== {name} ===")
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    out_degrees = [deg for node, deg in G.out_degree()]
    avg_out_degree = sum(out_degrees) / num_nodes

    k_in = np.array([G.in_degree(n) for n in G.nodes()], dtype=float)
    k_out = np.array([G.out_degree(n) for n in G.nodes()], dtype=float)

    mean_k_out = np.mean(k_out)
    mean_k_in_k_out = np.mean(k_in * k_out)

    if mean_k_in_k_out == 0:
        mean_k_in_k_out = 1e-9

    beta = (mean_k_out / mean_k_in_k_out)

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Average out-degree: {avg_out_degree:.4f}\n")
    print(f"beta: {beta:.4f}\n")


if __name__ == "__main__":
    Utils.setup_seed(5)
    FAA = nx.read_edgelist('./Networks/real/faa.txt', create_using=nx.DiGraph())
    a_list = np.arange(1.0, 2.0, 0.1)
    FAA = SIR_betas_directed(FAA, a_list, './SIR results/directedG/FAA_')

    P2P = nx.read_edgelist('./Networks/real/p2p-Gnutella05.txt', create_using=nx.DiGraph())
    a_list = np.arange(1.0, 2.0, 0.1)
    FAA = SIR_betas_directed(P2P, a_list, './SIR results/directedG/P2P_')

    FAA = nx.read_edgelist('./Networks/real/faa.txt', create_using=nx.DiGraph())
    FAA_SIR = Utils.load_sir_list('./SIR results/directedG/FAA_')

    P2P = nx.read_edgelist('./Networks/real/p2p-Gnutella05.txt', create_using=nx.DiGraph())
    P2P_SIR = Utils.load_sir_list('./SIR results/directedG/P2P_')

    # # Example: generate a directed BA training network
    # N = 1000
    # m_out = 4  # number of outgoing edges per new node
    # G_directed = nx.generators.directed.random_k_out_graph(
    #     n=N, k=m_out, alpha=1.0, self_loops=False
    # )
    # out_degrees = [d for n, d in G_directed.out_degree()]
    # print("Average out-degree:", sum(out_degrees)/N)
    # save_path = './directed_BA_network/'
    # os.makedirs(save_path, exist_ok=True)
    # nx.write_edgelist(G_directed, os.path.join(save_path, 'BA_directed.edgelist'), data=False)

    file_path = './directed_BA_network/BA_directed.edgelist'
    BA_1000_4 = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)

    BA_1000_4_label = SIR_dict_directed(BA_1000_4, real_beta=True)
    BA_1000_4_pd = pd.DataFrame({'Nodes': list(BA_1000_4_label.keys()), 'SIR': list(BA_1000_4_label.values())})

    save_path = './directed_BA_network'
    os.makedirs(save_path, exist_ok=True)

    file_name = os.path.join(save_path, 'BA_1000_4_SIR.csv')
    BA_1000_4_pd.to_csv(file_name, index=False)

    for i in range(1, 6):
        Utils.setup_seed(i)
        L = 8
        C = 1
        T = 4 
        batch_size = 64
        hidden = 64
        num_epochs = 100
        lr = 0.001

        BA_1000_4_sir = pd.read_csv('./directed_BA_network/BA_1000_4_SIR.csv')
        BA_1000_4_sir['SIR_log'] = np.log1p(BA_1000_4_sir['SIR'])
        BA_1000_4_label = dict(zip(np.array(BA_1000_4_sir['Nodes'], dtype=str), BA_1000_4_sir['SIR_log']))

        BA_1000_d = mainTPP_T4_directed(BA_1000_4, L, C)
        lstm_loader = Utils.Get_TPP_DataLoader(BA_1000_d, BA_1000_4_label, batch_size, seq_len=T)

        lstm = Models.LSTM(input_size=L, hidden_size=hidden, output_size=1)
        TPP, TPP_loss = Utils.train_model(lstm_loader, lstm, num_epochs, lr) 

        print(compare_TPP_nb(FAA, L, FAA_SIR, TPP, C))
        top = [5, 10, 15, 20]
        print(compare_jaccard_TPP(FAA, L, FAA_SIR, TPP, top, C))
