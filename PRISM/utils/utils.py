from collections import defaultdict
from torch_geometric.utils import to_networkx
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import random
import numpy as np
from sentence_transformers import SentenceTransformer

def Encoding_sbert(LLM_feature, args , model_name='all-MiniLM-L6-v2'):
    #print(f"-- Generate embedding by S-BERT --")
    
    model = SentenceTransformer(model_name, device=args.device)
    
    
    texts = [item.get('explanation', "") for item in LLM_feature]
    
    if not texts:
        return LLM_feature
    
def load_data(data_path, device):
    data = torch.load(data_path, map_location='cpu')
    return data.to(device)

def build_global_networkx(data):
    
    G = to_networkx(data, to_undirected=True)
    return G

def seed_everything(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) 
    random.seed(seed)
    
def weight_setting(cur_LLM_feature, data, args, current_round=0):
    device = args.device

    for item in cur_LLM_feature:
        if item.get("use_neighbor", 0) == 0:
            item["weight"] = 1.0

    node_to_structural_items = defaultdict(list)

    for idx, item in enumerate(cur_LLM_feature):
        if item.get("use_neighbor", 0) != 0:
            node_to_structural_items[item["node_id"]].append((idx, item))

    for _, indexed_items in node_to_structural_items.items():
        num_perspectives = len(indexed_items)

        if num_perspectives <= 1:
            for _, item in indexed_items:
                item["weight"] = 1.0
            continue

        rationale_embs = [
            _to_normalized_tensor(item["rationale_emb"], device)
            for _, item in indexed_items
        ]

        labels = [
            item["category"]
            for _, item in indexed_items
        ]

        for k, (_, item) in enumerate(indexed_items):
            label_agreement_sum = 0.0
            rationale_consistency_sum = 0.0

            for m in range(num_perspectives):
                if m == k:
                    continue

                if labels[k] == labels[m]:
                    label_agreement_sum += 1.0

                cosine_value = torch.dot(rationale_embs[k], rationale_embs[m]).item()
                rationale_consistency_sum += (1.0 + cosine_value) / 2.0

            denominator = num_perspectives - 1
            label_agreement = label_agreement_sum / denominator
            rationale_consistency = rationale_consistency_sum / denominator

            item["weight"] = float(label_agreement * rationale_consistency)

    return cur_LLM_feature


def _to_normalized_tensor(embedding, device, eps=1e-8):
    tensor = torch.as_tensor(
        embedding,
        dtype=torch.float,
        device=device,
    ).view(-1)

    return F.normalize(
        tensor,
        p=2,
        dim=0,
        eps=eps,
    )


def Select_Additional_Nodes(data, gnn_logits, train_mask, unlabeled_mask, args):
    device = gnn_logits.device
    num_nodes = data.num_nodes
    num_classes = gnn_logits.size(1)

    unlabeled_mask = unlabeled_mask.to(device).bool()

    if unlabeled_mask.sum() == 0:
        return [], torch.zeros(num_nodes, device=device)

    edge_index = data.edge_index.to(device)
    row = edge_index[0]

    gnn_probs = torch.softmax(gnn_logits, dim=-1)
    gnn_confidence, pseudo_labels = torch.max(gnn_probs, dim=-1)

    node_degree = degree(
        row,
        num_nodes,
        dtype=torch.float,
    ).to(device)

    confidence_score = _rank_score(
        values=gnn_confidence,
        mask=unlabeled_mask,
        high_value_is_better=False,
    )

    degree_score = _rank_score(
        values=node_degree,
        mask=unlabeled_mask,
        high_value_is_better=True,
    )

    final_score = confidence_score + degree_score
    final_score[~unlabeled_mask] = 0.0

    k_per_class = getattr(args, "iter_n_per_class", 1)

    selected_nodes = _select_topk_per_pseudo_class(
        final_score=final_score,
        pseudo_labels=pseudo_labels,
        unlabeled_mask=unlabeled_mask,
        num_classes=num_classes,
        k_per_class=k_per_class,
    )

    return selected_nodes, final_score


def _rank_score(values, mask, high_value_is_better):
    device = values.device
    score = torch.zeros_like(values, dtype=torch.float, device=device)

    candidate_idx = mask.nonzero(as_tuple=False).view(-1)

    if candidate_idx.numel() == 0:
        return score

    candidate_values = values[candidate_idx]

    sorted_order = torch.argsort(
        candidate_values,
        descending=high_value_is_better,
    )

    ranked_idx = candidate_idx[sorted_order]

    if ranked_idx.numel() == 1:
        score[ranked_idx] = 1.0
        return score

    rank_values = torch.linspace(
        1.0,
        0.0,
        steps=ranked_idx.numel(),
        device=device,
    )

    score[ranked_idx] = rank_values

    return score


def _select_topk_per_pseudo_class(
    final_score,
    pseudo_labels,
    unlabeled_mask,
    num_classes,
    k_per_class,
):
    selected_nodes = []

    for class_id in range(num_classes):
        class_mask = (pseudo_labels == class_id) & unlabeled_mask
        class_idx = class_mask.nonzero(as_tuple=False).view(-1)

        if class_idx.numel() == 0:
            continue

        k = min(k_per_class, class_idx.numel())
        class_scores = final_score[class_idx]
        topk_local_idx = torch.topk(class_scores, k=k).indices
        selected_nodes.extend(class_idx[topk_local_idx].tolist())

    return selected_nodes

import torch
import torch.nn.functional as F
import networkx as nx


def get_clustered_neighbors(target_idx, x, graph_nx, args):
    target_idx = to_python_int(target_idx)

    candidate_neighbors = get_two_hop_neighbors(
        graph_nx=graph_nx,
        target_idx=target_idx,
    )

    clusters = {-1: []}

    if len(candidate_neighbors) == 0:
        return clusters

    num_perspectives = getattr(args, "perspective", 3)
    num_nodes_per_view = getattr(args, "num_nodes_per_view", 3)
    mmr_lambda = getattr(args, "mmr_lambda", 0.5)

    target_emb, neighbor_embs, neighbor_norm = prepare_neighbor_embeddings(
        target_idx=target_idx,
        candidate_neighbors=candidate_neighbors,
        x=x,
    )

    sim_to_target = compute_target_similarity(
        target_emb=target_emb,
        neighbor_norm=neighbor_norm,
    )

    available_mask = torch.ones(
        len(candidate_neighbors),
        dtype=torch.bool,
        device=x.device,
    )

    for perspective_idx in range(num_perspectives):
        selected_local_indices = select_mmr_neighbors_for_perspective(
            sim_to_target=sim_to_target,
            neighbor_norm=neighbor_norm,
            available_mask=available_mask,
            num_nodes_per_view=num_nodes_per_view,
            mmr_lambda=mmr_lambda,
        )

        if len(selected_local_indices) == 0:
            break

        clusters[perspective_idx + 1] = [
            candidate_neighbors[local_idx]
            for local_idx in selected_local_indices
        ]

    return clusters


def to_python_int(value):
    return value.item() if isinstance(value, torch.Tensor) else value


def get_two_hop_neighbors(graph_nx, target_idx):
    hop_lengths = nx.single_source_shortest_path_length(
        graph_nx,
        target_idx,
        cutoff=2,
    )

    return [
        node
        for node in hop_lengths.keys()
        if node != target_idx
    ]


def prepare_neighbor_embeddings(target_idx, candidate_neighbors, x):
    target_emb = x[target_idx].view(1, -1)
    target_emb = F.normalize(target_emb, p=2, dim=1)

    neighbor_embs = x[candidate_neighbors]
    neighbor_norm = F.normalize(neighbor_embs, p=2, dim=1)

    return target_emb, neighbor_embs, neighbor_norm


def compute_target_similarity(target_emb, neighbor_norm):
    sim_to_target = torch.mm(
        neighbor_norm,
        target_emb.t(),
    ).squeeze()

    if sim_to_target.dim() == 0:
        sim_to_target = sim_to_target.unsqueeze(0)

    return sim_to_target


def select_mmr_neighbors_for_perspective(
    sim_to_target,
    neighbor_norm,
    available_mask,
    num_nodes_per_view,
    mmr_lambda,
):
    selected_local_indices = []

    for _ in range(num_nodes_per_view):
        next_idx = pick_next_mmr_neighbor(
            sim_to_target=sim_to_target,
            neighbor_norm=neighbor_norm,
            available_mask=available_mask,
            selected_local_indices=selected_local_indices,
            mmr_lambda=mmr_lambda,
        )

        if next_idx is None:
            break

        selected_local_indices.append(next_idx)
        available_mask[next_idx] = False

    return selected_local_indices


def pick_next_mmr_neighbor(
    sim_to_target,
    neighbor_norm,
    available_mask,
    selected_local_indices,
    mmr_lambda,
):
    candidate_idx = available_mask.nonzero(as_tuple=True)[0]

    if candidate_idx.numel() == 0:
        return None

    if len(selected_local_indices) == 0:
        candidate_scores = sim_to_target[candidate_idx]
        return candidate_idx[torch.argmax(candidate_scores)].item()

    selected_embs = neighbor_norm[selected_local_indices]
    candidate_embs = neighbor_norm[candidate_idx]

    sim_to_selected = torch.mm(
        candidate_embs,
        selected_embs.t(),
    )

    max_redundancy, _ = torch.max(
        sim_to_selected,
        dim=1,
    )

    relevance = sim_to_target[candidate_idx]

    mmr_scores = (
        mmr_lambda * relevance
        - (1.0 - mmr_lambda) * max_redundancy
    )

    return candidate_idx[torch.argmax(mmr_scores)].item()