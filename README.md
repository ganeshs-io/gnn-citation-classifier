<div align="center">

# 🧠 GNN Citation Network Classifier

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-orange?style=for-the-badge&logo=pytorch)](https://pyg.org)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Classifying research papers by training Graph Neural Networks on the Cora citation graph*

</div>

---

## 📌 Overview

Academic papers don't exist in isolation — they cite each other, forming a graph. 
This project trains two Graph Neural Networks to classify 2,708 research papers 
into 7 categories by learning from both paper content **and** citation structure.

The key question: **how much does the graph structure itself contribute?**
Answer: **+26% accuracy improvement** over using text features alone.

---

## 🏆 Results

<div align="center">

| Model | Test Accuracy | vs Baseline |
|:------|:------------:|:-----------:|
| 📊 Logistic Regression *(no graph)* | 57.0% | — |
| 🔵 GCN — Graph Convolutional Network | 81.0% | +24% |
| 🔴 GAT — Graph Attention Network | **83.0%** | **+26%** |

</div>

> **Why GAT beats GCN:** Academic citation graphs have heterogeneous neighborhoods —
> a reinforcement learning paper may cite both foundational RL papers and unrelated
> methodology papers. GAT's attention mechanism down-weights irrelevant citations,
> which uniform GCN averaging cannot do.

---

## 📊 Visualizations

<div align="center">

### Cora Citation Network — 200 nodes, 7 classes
![Graph](cora_graph.png)

### t-SNE of Learned Node Embeddings
*7 geometrically separated clusters = the model learned meaningful representations*
![t-SNE](embeddings_tsne.png)

### Training Curves — GCN vs GAT
![Training](training_curves.png)

</div>

---

## 🧩 Architecture
