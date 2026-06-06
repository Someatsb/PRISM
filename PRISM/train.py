import torch
import torch.nn.functional as F
from copy import deepcopy


def test_pipeline(model, align_mlp, data, mask):
    model.eval()
    align_mlp.eval()

    with torch.no_grad():
        logits, embeddings = model(data)
        preds = logits.argmax(dim=-1)
        correct = preds[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())

    return acc, logits, embeddings


def apply_mask(indices, mask):
    return mask[indices]


def get_valid_features(LLM_feature, label_map):
    return [
        item for item in LLM_feature
        if label_map.get(item["category"], -1) != -1
    ]


def split_features_by_context(valid_feature):
    feat_alone = [
        item for item in valid_feature
        if item.get("use_neighbor") == 0
    ]

    feat_cluster = [
        item for item in valid_feature
        if item.get("use_neighbor") == 1
    ]

    return feat_alone, feat_cluster


def get_feature_weights(features, device):
    if not features:
        return None

    return torch.tensor(
        [item.get("weight", 1.0) for item in features],
        dtype=torch.float,
        device=device,
    )


def get_rationale_tensors(features, args):
    if not features:
        return None, None

    rationale = torch.stack(
        [item["rationale_emb"] for item in features]
    ).to(args.device)

    node_ids = torch.tensor(
        [item["node_id"] for item in features],
        device=args.device,
    )

    return rationale, node_ids


def build_targets(features, data, label_map, labeled_mask, device):
    if not features:
        return None

    targets = []

    for item in features:
        node_id = item["node_id"]

        if labeled_mask[node_id]:
            target = int(data.y[node_id].item())
        else:
            target = label_map[item["category"]]

        targets.append(target)

    return torch.tensor(targets, dtype=torch.long, device=device)


def compute_weighted_label_loss(logits, node_ids, targets, weights, train_mask, args):
    if node_ids is None or targets is None:
        return logits.new_tensor(0.0)

    mask = apply_mask(node_ids, train_mask)

    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    ce_loss = F.cross_entropy(
        logits[node_ids],
        targets,
        label_smoothing=args.label_smoothing,
        reduction="none",
    )

    if weights is None:
        weights = torch.ones_like(ce_loss)

    masked_loss = ce_loss[mask]
    masked_weight = weights[mask]

    return (masked_loss * masked_weight).sum() / (masked_weight.sum() + 1e-8)


def project_gnn_embedding(align_mlp, gnn_emb):
    try:
        return align_mlp(gnn_emb, input_tensor=True)
    except TypeError:
        return align_mlp(gnn_emb)


def compute_weighted_rationale_loss(gnn_emb, node_ids, rationale, weights, train_mask, align_mlp):
    if node_ids is None or rationale is None:
        return gnn_emb.new_tensor(0.0)

    mask = apply_mask(node_ids, train_mask)

    if mask.sum() == 0:
        return gnn_emb.new_tensor(0.0)

    student_feat = project_gnn_embedding(
        align_mlp=align_mlp,
        gnn_emb=gnn_emb[node_ids],
    )

    teacher_feat = rationale

    feat_loss = 1.0 - F.cosine_similarity(
        student_feat,
        teacher_feat,
        dim=1,
    )

    if weights is None:
        weights = torch.ones_like(feat_loss)

    masked_loss = feat_loss[mask]
    masked_weight = weights[mask]

    return (masked_loss * masked_weight).sum() / (masked_weight.sum() + 1e-8)


def train_pipeline(
    model,
    align_mlp,
    data,
    train_mask,
    val_mask,
    test_mask,
    LLM_feature,
    args,
    labeled_mask=None,
):
    if labeled_mask is None:
        labeled_mask = train_mask.clone()

    labeled_mask = labeled_mask.to(args.device).bool()
    train_mask = train_mask.to(args.device).bool()
    val_mask = val_mask.to(args.device).bool()
    test_mask = test_mask.to(args.device).bool()

    label_map = {
        name: i for i, name in enumerate(data.label_names)
    }

    valid_feature = get_valid_features(LLM_feature, label_map)
    feat_alone, feat_cluster = split_features_by_context(valid_feature)

    w_a = get_feature_weights(feat_alone, args.device)
    w_c = get_feature_weights(feat_cluster, args.device)

    rationale_a, node_ids_a = get_rationale_tensors(feat_alone, args)
    rationale_c, node_ids_c = get_rationale_tensors(feat_cluster, args)

    targets_a = build_targets(
        features=feat_alone,
        data=data,
        label_map=label_map,
        labeled_mask=labeled_mask,
        device=args.device,
    )

    targets_c = build_targets(
        features=feat_cluster,
        data=data,
        label_map=label_map,
        labeled_mask=labeled_mask,
        device=args.device,
    )

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(align_mlp.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val = 0.0
    best_model = None
    best_align_mlp = None
    early_stop_accum = 0

    for epoch in range(args.epochs):
        model.train()
        align_mlp.train()

        optimizer.zero_grad()

        logits, gnn_emb = model(data)

        loss_cls_node = compute_weighted_label_loss(
            logits=logits,
            node_ids=node_ids_a,
            targets=targets_a,
            weights=w_a,
            train_mask=train_mask,
            args=args,
        )

        loss_cls_struct = compute_weighted_label_loss(
            logits=logits,
            node_ids=node_ids_c,
            targets=targets_c,
            weights=w_c,
            train_mask=train_mask,
            args=args,
        )

        loss_rat_node = compute_weighted_rationale_loss(
            gnn_emb=gnn_emb,
            node_ids=node_ids_a,
            rationale=rationale_a,
            weights=w_a,
            train_mask=train_mask,
            align_mlp=align_mlp,
        )

        loss_rat_struct = compute_weighted_rationale_loss(
            gnn_emb=gnn_emb,
            node_ids=node_ids_c,
            rationale=rationale_c,
            weights=w_c,
            train_mask=train_mask,
            align_mlp=align_mlp,
        )

        loss_cls = args.alpha * loss_cls_node + (1.0 - args.alpha) * loss_cls_struct
        loss_rat = args.alpha * loss_rat_node + (1.0 - args.alpha) * loss_rat_struct

        total_loss = args.beta * loss_cls + (1.0 - args.beta) * loss_rat

        total_loss.backward()
        optimizer.step()

        val_acc, _, _ = test_pipeline(
            model=model,
            align_mlp=align_mlp,
            data=data,
            mask=val_mask,
        )

        if epoch > args.warm_up:
            if val_acc > best_val:
                best_val = val_acc
                best_model = deepcopy(model)
                best_align_mlp = deepcopy(align_mlp)
                early_stop_accum = 0
            else:
                early_stop_accum += 1
                if early_stop_accum > args.early_stopping:
                    break

    if best_model is None:
        best_model = model
        best_align_mlp = align_mlp

    test_acc, test_logits, test_embeddings = test_pipeline(
        model=best_model,
        align_mlp=best_align_mlp,
        data=data,
        mask=test_mask,
    )

    return test_acc, test_logits, test_embeddings