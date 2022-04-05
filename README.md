# [SIGIR 22] AHP: Learning to Negative Sample for Hyperedge Prediction

## Overview
- Observation:We show that heuristic sampling schemes limit the generalization ability of deep learning-based hyperedge-prediction.

- Solution: AHP learns to sample negative examples by adversarial training for better generalization. In terms of AUROC, AHP is up to 28.2% better than best existing methods and up to 5.5% better than variants with sampling schemes tailored to test sets.

- Experiments: We compare AHP with three sampling schemes and three recent hyperedge-prediction methods on six real hypergraphs

## Main Paper
The main paper is available at [Here](./paper.pdf). If a preview does not appear properly, please download this file.

## Online Appendix
The online appendix is available at [Here](./OnlineAppendix.pdf). If a preview does not appear properly, please download this file.

## Code
The algorithms used in the paper is available at ```./AHP/```.

## Datasets
|Name|#Nodes|#Edges|Domain|
|:---:|:---:|:---:|:---:|:---:|
|Citeseer|1,457|1,078|Co-citation|
|Cora|1,434|1,579|Co-citation|
|Cora-A|2,388|1,072|Authorship|
|Pubmed|3,840|7,962|Co-citation|
|DBLP-A|39,283|16,483|Authorship|
|DBLP|15,639|22,964|Collaboration|
All data available at [Here](https://drive.google.com/drive/folders/1KKwkrZ2mMcc098pqwtpQrByWmTEigwzC?usp=sharing).

### Dataset format
Each dataset file contains following keys: 'N_edges', 'N_nodes', 'NodeEdgePair', 'EdgeNodePair', 'nodewt', 'edgewt', 'node_feat'.

We also provide preprocessed splits, which each contains train, validation, and a test set (both positive and negative).
This can be found in ```split/``` in the provided link above.
----

# AHP (Code Implementation)

## Execution
```
python hyperedge_prediction.py --dataset_name cora --model hnhn --epochs 200 --train_DG 1:1
```

### argument

## Output Format







