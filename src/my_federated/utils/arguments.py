import argparse

from my_federated.NeVe.utils import add_neve_arguments


def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def _get_default_arguments(parser):
    # General
    parser.add_argument("--seed", type=int, default=0,
                        help="Reproducibility seed.")
    parser.add_argument("--amp", type=int2bool, choices=[0, 1], default=True,
                        help="If True use torch.cuda.amp.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda",
                        help="Device type.")
    parser.add_argument("--model-name", type=str, default="tiny_vit_5m_224",
                        choices=["resnet18", "efficientnet_b0", "deit_tiny_patch16_224", "tiny_vit_5m_224"],
                        help="Name of the model to train.")
    parser.add_argument("--use-pretrain", type=int2bool, choices=[0, 1], default=True,
                        help="True to use a pretrained model, False to train from scratch. Default is False.")
    parser.add_argument("--model-use-groupnorm", type=int2bool, choices=[0, 1], default=True,
                        help="Use groupnorm rather than layernorm/batchnorm in the model")
    parser.add_argument("--model-groupnorm-groups", type=int, default=2,
                        help="Number of Groups to use in the GroupNorm. Default 2, select -1 to use all.")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw"], default="sgd",
                        help="Optimizer name.")
    parser.add_argument("--scheduler", type=str, choices=["constant", "multistep"], default="multistep",
                        help="Scheduler name.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size, an higher value requires more memory.")
    parser.add_argument("--val-percentage", type=int, default=10,
                        help="Percentage of train used as validation.")
    parser.add_argument("--lda-concentration", type=float, default=0.1,
                        help="If not IID data is used, this parameter is used to control the lda partitioning."
                             "Higher value generates uniform partitions, lower value generates imbalanced partitions."
                             "Es: 0.0 Generates one class per client, +inf generates uniform partitions over classes.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Optimizer starting learning rate.")
    parser.add_argument("--min-lr", type=float, default=1e-5,
                        help="Optimizer minimum learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Optimizer weight decay.")

    # Dataset
    parser.add_argument("--print-clients-distribution", type=int2bool, choices=[0, 1], default=False,
                        help="Print client data distribution. I suggest to keep it off for high amount of clients.")
    parser.add_argument("--dataset-root", type=str, default="../datasets",
                        help="Dataset root folder.")
    parser.add_argument("--dataset-name", type=str, default="cifar10",
                        choices=["mnist", "fashionmnist", "cifar10", "cifar100", "caltech256", "eurosat", "imagenette",
                                 "medmnist_pathmnist", "medmnist_chestmnist", "medmnist_dermamnist",
                                 "medmnist_octmnist",
                                 "medmnist_pneumoniamnist", "medmnist_retinamnist", "medmnist_breastmnist",
                                 "medmnist_bloodmnist",
                                 "medmnist_tissuemnist", "medmnist_organamnist", "medmnist_organcmnist",
                                 "medmnist_organsmnist"],
                        help="Dataset folder name.")
    parser.add_argument("--medmnist-size", type=int, default=224,
                        choices=[28, 64, 128, 224],
                        help="Size of medmnist dataset samples. Default 224.")
    parser.add_argument("--dataset-iid", type=int2bool, choices=[0, 1], default=True,
                        help="Use a IID split for the dataset")
    # NeVe
    parser = add_neve_arguments(parser)

    # Wandb
    parser.add_argument("--wandb-project-name", type=str, default="NeVe-Federated-ClientSelection")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=[], nargs="+",
                        help="Tags associated to the wandb run")


def _get_federated_arguments(parser):
    parser.add_argument("--server-address", type=str, default="127.0.0.1:6789",
                        help="Server address:port.")
    parser.add_argument("--inner-epochs", type=int, default=1,
                        help="Number of inner training epochs.")
    parser.add_argument("--num-clients", type=int, default=10,
                        help="Number of clients in the federated learning.")
    parser.add_argument("--min-fit-clients", type=int, default=5,
                        help="Minimum number of clients ready to start a fit operation.")
    parser.add_argument("--min-evaluate-clients", type=int, default=5,
                        help="Minimum number of clients ready to start an evaluate operation.")
    parser.add_argument("--strategy-name", type=str,
                        choices=["fedavg", "neve"],
                        default="fedavg",
                        help="What strategy to use in the server. (Default: FedAvg)")


def _get_client_arguments(parser):
    _get_default_arguments(parser)
    _get_federated_arguments(parser)
    parser.add_argument("--current-client", type=int, default=0,
                        help="Client index of this process.")


def _get_server_arguments(parser):
    _get_default_arguments(parser)
    _get_federated_arguments(parser)


def _get_federated_script_arguments(parser):
    _get_default_arguments(parser)
    _get_federated_arguments(parser)
    parser.add_argument("--max-core-count", type=int, default=12,
                        help="Max amount cores that can be used for the simulation.")


def _get_classical_script_arguments(parser):
    _get_default_arguments(parser)
    parser.add_argument("--num-clients", type=int, default=10,
                        help="Number of clients in the federated learning.")
    parser.add_argument("--current-client", type=int, default=0,
                        help="Client index of this process.")
    parser.add_argument("--strategy-name", type=str,
                        choices=["fedavg", "neve"],
                        default="fedavg",
                        help="What strategy to use in the server. (Default: FedAvg)")


def get_args(script="federated"):
    assert script in ["federated", "classical"]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    match script:
        case "federated":
            _get_federated_script_arguments(parser)
        case "classical":
            _get_classical_script_arguments(parser)
        case _:
            raise Exception(f"Error script type: {script} is not managed yet.")
    args = parser.parse_args()
    args.min_fit_clients = args.num_clients
    args.min_evaluate_clients = args.num_clients
    print("Arguments:", args)
    return args
