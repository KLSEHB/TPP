 **TPP: A Temporal-Enhanced Propagation Probability Model for Identifying Influential Nodes in Complex Networks**

.
├─ Networks/          # Datasets
│   ├─ real/          # Real-world networks (facebook, vidal, powergrid…)
│   └─ training/      # Training graphs (BA)
│
├─ SIR_results/       # SIR simulation labels
│
├─ results/           # Experimental Results (json)
│
├─ main.py            # Main script – train / test
├─ getSirLabel.py     # Run SIR → save infection-curve labels
├─ Test.py            # Link-prediction & node-classification metrics
└─ DirectedG.py       # Experimental SIR on directed graphs
