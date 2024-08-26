import argparse

from src.NeVe.utils import add_neve_arguments


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
    parser.add_argument("--model-name", type=str, default="resnet18",
                        choices=["resnet18", "efficientnet_b0"],
                        help="Name of the model to train.")
    parser.add_argument("--use-pretrain", type=int2bool, choices=[0, 1], default=False,
                        help="True to use a pretrained model, False to train from scratch. Default is False.")
    parser.add_argument("--model-use-groupnorm", type=int2bool, choices=[0, 1], default=True,
                        help="Use groupnorm rather than layernorm/batchnorm in the model")
    parser.add_argument("--model-groupnorm-groups", type=int, default=2,
                        help="Number of Groups to use in the GroupNorm. Default 2, select -1 to use all.")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="sgd",
                        help="Optimizer name.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size, an higher value requires more memory.")
    parser.add_argument("--lda-concentration", type=float, default=0.5,
                        help="If not IID data is used, this parameter is used to control the lda partitioning."
                             "Higher value generates uniform partitions, lower value generates imbalanced partitions."
                             "Es: 0.0 Generates one class per client, +inf generates uniform partitions over classes.")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Optimizer starting learning rate.")
    parser.add_argument("--min-lr", type=float, default=0.00001,
                        help="Optimizer minimum learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Optimizer momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Optimizer weight decay.")

    # Dataset
    parser.add_argument("--dataset-root", type=str, default="../datasets",
                        help="Dataset root folder.")
    parser.add_argument("--dataset-name", type=str, default="cifar10",
                        choices=["emnist", "cifar10", "cifar100", "imagenet100"],
                        help="Dataset folder name.")
    parser.add_argument("--dataset-iid", type=int2bool, choices=[0, 1], default=True,
                        help="Use a IID split for the dataset")
    parser.add_argument("--leaf-input-dim", type=int, default=10,
                        help="Input dimension of certain leaf datasets (e.g. synthetic)")
    # NeVe
    parser = add_neve_arguments(parser)
    parser.add_argument("--scheduler-name", type=str, choices=["neve", "baseline", "neq"], default="baseline",
                        help="'neve' use NeVe scheduler. 'baseline' use use MultiStepLR.")

    # Wandb
    parser.add_argument("--wandb-project-name", type=str, default="NeVe-Federated")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=[], nargs="+",
                        help="Tags associated to the wandb run")


def _get_federated_arguments(parser):
    parser.add_argument("--server-address", type=str, default="127.0.0.1:6789",
                        help="Server address:port.")
    parser.add_argument("--num-clients", type=int, default=10,
                        help="Number of clients in the federated learning.")
    parser.add_argument("--min-fit-clients", type=int, default=5,
                        help="Minimum number of clients ready to start a fit operation.")
    parser.add_argument("--min-evaluate-clients", type=int, default=5,
                        help="Minimum number of clients ready to start an evaluate operation.")
    parser.add_argument("--clients-sampling-method", type=str,
                        choices=["default", "percentage_random", "percentage_groups", "velocity"],
                        default="default",
                        help="How FedeNevAvg samples clients.")
    parser.add_argument("--clients-sampling-percentage", type=float, default=0.5,
                        help="Percentage of clients to sample.")
    parser.add_argument("--clients-sampling-velocity-aging", type=float, default=0.01,
                        help="Aging applied to velocity for not-sampled clients (Percentage [0.0, 1.0]).")
    parser.add_argument("--clients-sampling-highest-velocity", type=int2bool, choices=[0, 1], default=True,
                        help="1 if we sample clients with highest velocity, 0 to sample with the lowest velocity.")
    parser.add_argument("--clients-sampling-wait-epochs", type=int, default=10,
                        help="Number of epochs in which we select all available clients.")
    parser.add_argument("--clients-sampling-min-epochs", type=int, default=2,
                        help="Number of epochs a client is at least selected for in a row.")
    parser.add_argument("--clients-sampling-use-probability", type=int2bool, choices=[0, 1], default=True,
                        help="1 if we sample clients randomly based on velocity, false to always select the ones "
                             "with the highest velocity")


def _get_client_arguments(parser):
    _get_default_arguments(parser)
    _get_federated_arguments(parser)
    parser.add_argument("--current-client", type=int, default=0,
                        help="Client index of this process.")


def _get_server_arguments(parser):
    _get_default_arguments(parser)
    _get_federated_arguments(parser)


def _get_simulation_arguments(parser):
    _get_default_arguments(parser)
    _get_federated_arguments(parser)


def _get_classical_arguments(parser):
    _get_default_arguments(parser)


def get_args(script="client"):
    assert script in ["client", "server", "simulation", "classical"]
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    match script:
        case "client":
            _get_client_arguments(parser)
        case "server":
            _get_server_arguments(parser)
        case "simulation":
            _get_simulation_arguments(parser)
        case "classical":
            _get_classical_arguments(parser)
        case _:
            raise Exception(f"Error script type: {script} is not managed yet.")
    args = parser.parse_args()
    print("Arguments:", args)
    return args
