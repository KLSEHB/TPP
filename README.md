# TPP: A Topology–Propagation–Prediction Framework for Influence Identification

This repository contains the official implementation of our paper:

> **"TPP: Integrating Propagation-based Methods with Deep Learning for Influence Identification in Complex Networks"**  
> by KLSEHB et al., 2025.

---

## 📘 Overview
TPP integrates topology-based and propagation-based learning to identify influential nodes in complex networks.  
It combines diffusion simulations with deep representation learning to capture both **structural** and **dynamic** features of networks.

While traditional methods rely only on topology or handcrafted propagation models,  
TPP leverages both — the propagation model provides prior knowledge,  
while deep learning automatically discovers latent relationships between network structure and diffusion dynamics.

---

## 📂 Project Structure
TPP-main/
├── DirectedG.py # Directed graph generation and processing
├── Embeddings.py # Embedding learning for node features
├── getSirLable.py # Label generation using SIR model
├── main.py # Main training and execution script
├── Models.py # Deep learning model definitions
├── Test.py # Evaluation and testing script
├── Utils.py # Helper functions
├── Networks/ # Network data
├── directed_BA_network/ # Example directed BA networks
├── results/ # Experimental results
├── SIR results/ # SIR diffusion outcomes
└── time_Networks/ # Temporal network data
