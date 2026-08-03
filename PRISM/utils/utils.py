from collections import defaultdict
from torch_geometric.utils import to_networkx
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
import random
import numpy as np
from sentence_transformers import SentenceTransformer

def encode_with_sbert(LLM_feature, args , model_name='all-MiniLM-L6-v2'):
    #print(f"-- Generate embedding by S-BERT --")

    model = SentenceTransformer(model_name, device=args.device)


    texts = [item.get('explanation', "") for item in LLM_feature]

    if not texts:
        return LLM_feature


    embeddings = model.encode(texts, convert_to_tensor=True, device=args.device)

    for i, item in enumerate(LLM_feature):
        item['rationale_emb'] = embeddings[i]

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

def weight_setting(cur_LLM_feature, data, args):
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


def select_additional_nodes(data, gnn_logits, train_mask, unlabeled_mask, args):
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
