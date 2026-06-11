import argparse


def replace_args_with_dict_values(args, dictionary):
    for key, value in dictionary.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def get_command_line_args():
    parser = argparse.ArgumentParser(description='PRISM')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--dataset', default='pubmed', type=str)
    parser.add_argument('--round', default=5, type=int)
    parser.add_argument('--seed', type=int, default=3)

    parser.add_argument('--model_name', type=str, default='GCN')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dimension', type=int, default=256)
    parser.add_argument('--norm', type=str, default=None)
    parser.add_argument('--theta', type=float, default=.5, help='theta for gcn2')
    parser.add_argument('--gcn2_alpha', type=float, default=0.1)
    parser.add_argument('--num_of_heads', type=int, default=8)
    parser.add_argument('--shared_weights', type=bool, default=True)
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument('--perspective', type=int, default=3)
    parser.add_argument('--init_n_per_class', type=int, default=3)
    parser.add_argument('--iter_n_per_class', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument("--num_nodes_per_view", type=int, default=3)
    parser.add_argument("--mmr_lambda", type=float, default=0.5)
    args = parser.parse_args()
    parser.add_argument(
        "--llm_model_id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.set_defaults(use_llm_cache=False)
    return args