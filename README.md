# PRISM

Official implementation of **PRISM: Perspective-based Reliable Imitation with Semantic Multiplicity** for few-label node classification on text-attributed graphs.

## Overview

Text-attributed graph (TAG) learning aims to predict node labels by jointly leveraging textual attributes and graph structure.
However, in few-label settings, GNNs often suffer from limited supervision, while LLM-generated knowledge can also be unreliable when it is generated only from the textual attribute of a target node.

PRISM addresses this problem by generating multiple graph-context perspectives for each target node and estimating the reliability of LLM-generated knowledge across these perspectives.
The estimated reliability is then used as a weight for knowledge distillation, enabling GNNs to learn more from reliable LLM supervision while reducing the influence of unreliable responses.

## Key Ideas

PRISM consists of three main components:

* **Multi-Perspective Knowledge Generation**
  Generates multiple LLM responses for each target node by considering different graph-context perspectives.

* **Reliability Weight Estimation**
  Estimates the reliability of LLM-generated knowledge based on label agreement and rationale consistency across perspectives.

* **Reliable Knowledge Distillation**
  Distills LLM-generated labels and rationales into a GNN using reliability-aware weights.

## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

The implementation is based on Python and PyTorch.
Please make sure that PyTorch Geometric and other graph learning dependencies are properly installed according to your CUDA environment.

## Datasets

We evaluate PRISM on widely used text-attributed graph benchmark datasets, including:

* Cora
* Citeseer
* PubMed
* DBLP
* ogbn-arxiv

Please place the datasets under the following directory:

```text
dataset/
```

The expected structure is:

```text
dataset/
├── Cora/
├── Citeseer/
├── PubMed/
├── DBLP/
└── ogbn-arxiv/
```

## Running PRISM

An example command for running PRISM is:

```bash
python main.py \
  --dataset Cora \
  --init_n_per_class 3 \
  --perspective 3 \
  --use_neighbor 1 \
  --train_methods N+S_OPM
```

You can modify the dataset, number of labeled nodes per class, number of perspectives, and training options depending on the experimental setting.

For questions or discussions, please open an issue in this repository.
