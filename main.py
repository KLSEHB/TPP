import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import torch

import Models
import Utils
import Test
import Embeddings

# ============================
# Configuration & Setup
# ============================

warnings.filterwarnings('ignore')
sns.set_style('ticks')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if __name__ == '__main__':
    # Set random seed for reproducibility (1–5 tested; here using seed 4)
    Utils.setup_seed(5)

    # Load synthetic training network: Barabási–Albert with 1000 nodes, m=4
    BA_1000_4 = Utils.load_graph('./Networks/training/Train_1000_4.txt')

    # Load normalized SIR spreading scores for training graph
    BA_1000_4_norm_sir = pd.read_csv('./SIR results/Train_1000_4/BA_1000_norm.csv')
    BA_1000_4_norm_label = dict(zip(
        np.array(BA_1000_4_norm_sir['Node'], dtype=str),
        BA_1000_4_norm_sir['SIR_log']
    ))

    # ============================
    # Load Real-World Networks
    # ============================
    Facebook      = Utils.load_graph('./Networks/real/facebook_combined.txt')
    LastFM        = Utils.load_graph('./Networks/real/LastFM.txt')
    Faa           = Utils.load_graph('./Networks/real/faa.txt')
    NetScience    = Utils.load_graph('./Networks/real/NetScience.txt')
    Hep           = Utils.load_graph('./Networks/real/CA-HepTh.txt')
    Vidal         = Utils.load_graph('./Networks/real/vidal.txt')
    GrQC          = Utils.load_graph('./Networks/real/CA-GrQc.txt')
    Politician    = Utils.load_graph('./Networks/real/Politician.txt')
    PowerGrid     = Utils.load_graph('./Networks/real/powergrid.txt')

    # Load precomputed SIR rankings for real-world networks
    Facebook_SIR   = Utils.load_sir_list('./SIR results/Facebook/Facebook_')
    LastFM_SIR     = Utils.load_sir_list('./SIR results/LastFM/LastFM_')
    Faa_SIR        = Utils.load_sir_list('./SIR results/Faa/Faa_')
    NetScience_SIR = Utils.load_sir_list('./SIR results/NetScience/NetScience_')
    Hep_SIR        = Utils.load_sir_list('./SIR results/Hep/Hep_')
    Vidal_SIR      = Utils.load_sir_list('./SIR results/vidal/vidal_')
    GrQC_SIR       = Utils.load_sir_list('./SIR results/GrQ/GrQ_')
    Politician_SIR = Utils.load_sir_list('./SIR results/Politician/Politician_')
    PowerGrid_SIR  = Utils.load_sir_list('./SIR results/powergrid/powergrid_')

    # Datasets to evaluate on
    datasets = ["Facebook", "LastFM", "Faa", "NetScience", "Hep",
                "Vidal", "GrQC", "Politician", "PowerGrid"]

    # ============================
    # Hyperparameters
    # ============================

    L = 8          # Embedding dimension (number of neighbors sampled)
    T = 4          # Temporal order (sequence length)
    C = 1          # Number of TPP components
    hidden = 64
    batch_size = 64
    num_epochs = 100
    lr = 0.001

    # ============================
    # Train TPP-LSTM Model
    # ============================

    # Generate TPP embeddings for training graph
    BA_1000_d = Embeddings.mainTPP_T(BA_1000_4, L, C, T)

    # Create DataLoader for LSTM training
    lstm_loader = Utils.Get_TPP_DataLoader(
        BA_1000_d, BA_1000_4_norm_label, batch_size, seq_len=T
    )

    # Define LSTM model
    lstm = Models.LSTM(input_size=L, hidden_size=hidden, output_size=1)

    # Train model
    TPP_model, TPP_loss = Utils.train_model(lstm_loader, lstm, num_epochs, lr)

    # ============================
    # Evaluation: Kendall's Tau
    # ============================
    TPP_tau_results = {dataset: [] for dataset in datasets}
    for data in datasets:
        G = globals()[data]
        sir_list = globals()[f"{data}_SIR"]
        tau_values = Test.tau_TPP(G, L, sir_list, TPP_model, C, T)
        TPP_tau_results[data].extend(tau_values)

    print("Kendall's Tau Results:")
    print(TPP_tau_results)

    # ============================
    # Evaluation: Mutual Information (MI)
    # ============================

    TPP_mi_results = {dataset: [] for dataset in datasets}
    for data in datasets:
        G = globals()[data]
        sir_list = globals()[f"{data}_SIR"]
        mi_values = Test.mi_TPP(G, L, TPP_model, C, T)
        TPP_mi_results[data].append(mi_values)

    print("\nMutual Information Results:")
    print(TPP_mi_results)

    # ============================
    # Evaluation: Jaccard Similarity
    # ============================

    top_k = [5, 10, 15, 20]
    TPP_jac_results = {dataset: [] for dataset in datasets}

    for data in datasets:
        G = globals()[data]
        sir_list = globals()[f"{data}_SIR"]
        jac_values = Test.jac_TPP(G, L, sir_list, TPP_model, top_k, C, T)
        TPP_jac_results[data].extend(jac_values)

    print("\nJaccard Similarity Results:")
    print(TPP_jac_results)