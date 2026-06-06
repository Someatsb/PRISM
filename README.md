# PRISM

Official implementation of **PRISM: Reliable LLM Knowledge Distillation from Graph-Context Perspectives for Text-Attributed Graph Learning**.

PRISM is a reliability-aware knowledge distillation framework for few-label node classification on text-attributed graphs (TAGs). It improves the reliability of LLM-generated supervision by constructing multiple graph-context perspectives for each target node and estimating knowledge reliability based on cross-perspective consistency.

## Overview

Text-attributed graph (TAG) learning aims to leverage both textual attributes and graph structures for node classification. In few-label settings, however, GNNs often struggle to learn reliable decision boundaries due to the limited availability of labeled nodes.

Recent LLM-assisted TAG learning methods use Large Language Models (LLMs) to generate pseudo-labels and rationales as additional supervision for GNN training. However, these methods typically rely only on the textual attribute of each target node. When node texts are short, ambiguous, or incomplete, the generated pseudo-labels and rationales can become unreliable and may act as noisy supervision.

PRISM addresses this limitation by exploiting graph-context information. For each target node, PRISM constructs multiple structural-context perspectives using different sets of neighboring nodes. The LLM generates pseudo-labels and rationales from each perspective. PRISM then estimates the reliability of generated knowledge based on label agreement and rationale consistency across perspectives, and uses the estimated reliability as a distillation weight during GNN training.

## Key Features

* Multi-perspective LLM knowledge generation from graph-context perspectives
* Reliability estimation based on cross-perspective consistency
* Reliability-aware class-level and rationale-level knowledge distillation
* Active node selection under a limited LLM query budget
* Evaluation on five text-attributed graph benchmark datasets



## Requirements

Install the required packages with:

```bash
pip install -r requirements.txt
```

The implementation is based on Python and PyTorch. Please install PyTorch Geometric and other graph learning dependencies according to your CUDA environment.

## Datasets

PRISM is evaluated on the following text-attributed graph benchmark datasets:

* Cora
* Citeseer
* PubMed
* DBLP
* ogbn-arxiv

Please download the datasets from [Dataset](https://drive.google.com/drive/folders/1ZLF8ge2uxHDgiLtnYiofayWddBnr0UQ0?usp=sharing) and place the datasets under:

```text
dataset/
```

Expected directory structure:

```text
dataset/
├── Cora/
├── Citeseer/
├── PubMed/
├── DBLP/
└── ogbn-arxiv/
```

## Running PRISM

Example command:

```bash
python main.py \
  --dataset Cora \
  --init_n_per_class 3 \
  --perspective 3 \
  --use_neighbor 1 \
  --train_methods N+S_OPM
```


## Contact

For questions or discussions, please open an issue in this repository.
