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
