import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import numpy as np
import torch
import wandb

from args import get_command_line_args
from utils.utils import *
from train import *
from LLM import *
from models.nn import *


def init_wandb(args):
    wandb.init(
        project="PRISM",
        config={
            "dataset": args.dataset,
            "init_n_per_class": args.init_n_per_class,
            "perspective": args.perspective,
            "round": args.round,
            "alpha": args.alpha,
            "beta": args.beta,
            "lr": args.lr,
            "hidden_dimension": args.hidden_dimension,
            "train_method": "PRISM",
            "llm_call": True,
        },
    )


def sample_initial_nodes(pool_indices, labels, num_classes, init_n_per_class):
    init_node_indices = []

    for class_id in range(num_classes):
        class_nodes = pool_indices[labels[pool_indices] == class_id]
        sample_size = min(init_n_per_class, len(class_nodes))
        sampled_nodes = class_nodes[:sample_size]
        init_node_indices.extend(sampled_nodes.tolist())

    return init_node_indices


def build_split_masks(data, args):
    device = args.device
    num_nodes = args.num_node

    indices = torch.randperm(num_nodes, device=device)

    pool_size = int(num_nodes * 0.6)
    val_size = int(num_nodes * 0.2)

    pool_indices = indices[:pool_size]
    val_indices = indices[pool_size:pool_size + val_size]
    test_indices = indices[pool_size + val_size:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    pool_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    val_mask[val_indices] = True
    test_mask[test_indices] = True
    pool_mask[pool_indices] = True

    init_node_indices = sample_initial_nodes(
        pool_indices=pool_indices,
        labels=data.y,
        num_classes=args.num_classes,
        init_n_per_class=args.init_n_per_class,
    )

    train_mask[init_node_indices] = True
    pool_mask[init_node_indices] = False

    labeled_mask = train_mask.clone()

    return train_mask, val_mask, test_mask, pool_mask, labeled_mask, init_node_indices





def generate_or_load_llm_features(
    target_nodes,
    data,
    graph_nx,
    args,
    seed_idx,
    cur_round,
    llm,
    tokenizer,
):
    use_cache = getattr(args, "use_llm_cache", True)

    

    target_nodes = torch.tensor(
        target_nodes,
        dtype=torch.long,
        device=args.device,
    )

    features = Generate_feature_by_LLM(
        node_indices=target_nodes,
        data=data,
        graph_nx=graph_nx,
        args=args,
        cur_round=cur_round,
        llm=llm,
        tokenizer=tokenizer,
    )

    return features


def prepare_llm_features(
    target_nodes,
    data,
    graph_nx,
    args,
    seed_idx,
    cur_round,
    llm,
    tokenizer,
):
    cur_features = generate_or_load_llm_features(
        target_nodes=target_nodes,
        data=data,
        graph_nx=graph_nx,
        args=args,
        seed_idx=seed_idx,
        cur_round=cur_round,
        llm=llm,
        tokenizer=tokenizer,
    )

    cur_features = [
        item for item in cur_features
        if item.get("category") != "Unknown"
    ]

    cur_features = Encoding_sbert(cur_features, args)
    cur_features = weight_setting(cur_features, data, args, cur_round)

    return cur_features


def train_prism_one_round(
    data,
    train_mask,
    val_mask,
    test_mask,
    llm_features,
    args,
    labeled_mask,
):
    model = get_model(args).to(args.device)

    align_mlp = MLP(
        args.hidden_dimension,
        data.x.shape[1],
        args.dropout,
    ).to(args.device)

    test_acc, logits, embeddings = train_pipeline(
        model=model,
        align_mlp=align_mlp,
        data=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        LLM_feature=llm_features,
        args=args,
        labeled_mask=labeled_mask,
    )

    return round(test_acc * 100, 2), logits, embeddings


def update_train_pool_masks(train_mask, pool_mask, selected_nodes, device):
    if len(selected_nodes) == 0:
        return train_mask, pool_mask

    selected_nodes = torch.tensor(
        selected_nodes,
        dtype=torch.long,
        device=device,
    )

    train_mask[selected_nodes] = True
    pool_mask[selected_nodes] = False

    return train_mask, pool_mask


def print_experiment_header(args, llm_cache_enabled):
    print("\n========== PRISM Experiment ==========")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Nodes: {args.num_node}")
    print(f"Classes: {args.num_classes}")
    print("LLM call: enabled")
    print(f"LLM cache: {llm_cache_enabled}")
    print("=====================================")


def print_seed_summary(seed_idx, train_mask, val_mask, test_mask, pool_mask):
    print(f"\n[Seed {seed_idx}]")
    print(f"Train nodes: {train_mask.sum().item()}")
    print(f"Validation nodes: {val_mask.sum().item()}")
    print(f"Test nodes: {test_mask.sum().item()}")
    print(f"Active pool nodes: {pool_mask.sum().item()}")


def print_round_summary(cur_round, llm_count, test_acc, selected_nodes=None):
    message = (
        f"[Round {cur_round}] "
        f"LLM features: {llm_count} | "

        f"Test acc: {test_acc:.2f}"
    )

    if selected_nodes is not None:
        message += f" | Selected: {len(selected_nodes)}"

    print(message)


def log_round_result(seed_idx, cur_round, test_acc,  llm_count, selected_count):
    wandb.log({
        "seed": seed_idx,
        "round": cur_round,
        f"round_{cur_round}/test_acc": test_acc,
        f"round_{cur_round}/llm_feature_count": llm_count,
        f"round_{cur_round}/selected_count": selected_count,
    })


def run_single_seed(seed_idx, data, graph_nx, args, llm, tokenizer):
    seed_everything(42 + seed_idx)

    train_mask, val_mask, test_mask, pool_mask, labeled_mask, init_node_indices = build_split_masks(
        data=data,
        args=args,
    )

    print_seed_summary(
        seed_idx=seed_idx,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        pool_mask=pool_mask,
    )

    all_llm_features = []
    current_target_nodes = init_node_indices

    seed_round_acc = []
    best_acc = -1.0
    best_round = -1

    for cur_round in range(args.round):
        cur_features = prepare_llm_features(
            target_nodes=current_target_nodes,
            data=data,
            graph_nx=graph_nx,
            args=args,
            seed_idx=seed_idx,
            cur_round=cur_round,
            llm=llm,
            tokenizer=tokenizer,
        )

        if cur_round == 0:
            all_llm_features = cur_features
        else:
            all_llm_features.extend(cur_features)

        test_acc, logits, embeddings = train_prism_one_round(
            data=data,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            llm_features=all_llm_features,
            args=args,
            labeled_mask=labeled_mask,
        )

        seed_round_acc.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_round = cur_round

        if cur_round == args.round - 1:
            print_round_summary(
                cur_round=cur_round,
                llm_count=len(all_llm_features),
               
                test_acc=test_acc,
            )

            log_round_result(
                seed_idx=seed_idx,
                cur_round=cur_round,
                test_acc=test_acc,
             
                llm_count=len(all_llm_features),
                selected_count=0,
            )
            break

        selected_nodes, final_score = Select_Additional_Nodes(
            data=data,
            gnn_logits=logits.to(args.device),
            train_mask=train_mask,
            unlabeled_mask=pool_mask,
            args=args,
        )

        train_mask, pool_mask = update_train_pool_masks(
            train_mask=train_mask,
            pool_mask=pool_mask,
            selected_nodes=selected_nodes,
            device=args.device,
        )

        current_target_nodes = selected_nodes

        print_round_summary(
            cur_round=cur_round,
            llm_count=len(all_llm_features),
           
            test_acc=test_acc,
            selected_nodes=selected_nodes,
        )

        log_round_result(
            seed_idx=seed_idx,
            cur_round=cur_round,
            test_acc=test_acc,
            llm_count=len(all_llm_features),
            selected_count=len(selected_nodes),
        )

    return {
        "final_acc": seed_round_acc[-1],
        "best_acc": best_acc,
        "best_round": best_round,
        "round_acc": seed_round_acc,
    }


def summarize_results(seed_results, args):
    final_accs = [result["final_acc"] for result in seed_results]
    best_accs = [result["best_acc"] for result in seed_results]
    best_rounds = [result["best_round"] for result in seed_results]

    final_mean = round(np.mean(final_accs), 2)
    final_std = round(np.std(final_accs), 2)
    best_mean = round(np.mean(best_accs), 2)
    best_std = round(np.std(best_accs), 2)
    best_round_mean = round(np.mean(best_rounds), 2)

    print("\n========== Final Summary ==========")
    print(f"Dataset: {args.dataset}")
    print(f"Shot: {args.init_n_per_class}")
    print(f"Perspective: {args.perspective}")
    print(f"Rounds: {args.round}")
    print(f"Final Acc: {final_mean:.2f} ± {final_std:.2f}")
    print(f"Best Acc: {best_mean:.2f} ± {best_std:.2f}")
    print(f"Best Round: {best_round_mean:.2f}")

    wandb.log({
        "final_acc_mean": final_mean,
        "final_acc_std": final_std,
        "final_acc_writing": f"{final_mean}±{final_std}",
        "best_acc_mean": best_mean,
        "best_acc_std": best_std,
        "best_acc_writing": f"{best_mean}±{best_std}",
        "best_round_mean": best_round_mean,
    })


def main(args):
    init_wandb(args)

    data = load_data(args.data_path, args.device)
    args.input_dim = data.x.shape[1]
    args.num_node = data.x.shape[0]
    args.num_classes = data.y.max().item() + 1

    graph_nx = build_global_networkx(data)

    llm, tokenizer = load_llm(
        getattr(args, "llm_model_id", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    )

    use_cache = getattr(args, "use_llm_cache", True)
    print_experiment_header(args, use_cache)

    seed_results = []

    for seed_idx in range(args.seed):
        result = run_single_seed(
            seed_idx=seed_idx,
            data=data,
            graph_nx=graph_nx,
            args=args,
            llm=llm,
            tokenizer=tokenizer,
        )
        seed_results.append(result)

    summarize_results(seed_results, args)
    wandb.finish()


if __name__ == "__main__":
    args = get_command_line_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.abspath(os.path.join(base_dir, ".."))

    args.main_path = main_path
    args.data_path = f"{main_path}/dataset/{args.dataset}_fixed_sbert.pt"

    if not hasattr(args, "use_llm_cache"):
        args.use_llm_cache = True

    main(args)