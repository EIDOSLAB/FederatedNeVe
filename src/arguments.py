import argparse
import sys

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
                        help="Name of the model to train.")
    parser.add_argument("--model-use-groupnorm", type=int2bool, choices=[0, 1], default=True,
                        help="Use groupnorm rather than layernorm/batchnorm in the model")
    parser.add_argument("--model-groupnorm-groups", type=int, default=2,
                        help="Number of Groups to use in the GroupNorm. Default 2, select -1 to use all.")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="sgd",
                        help="Optimizer name.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size, an higher value requires more memory.")
    parser.add_argument("--epochs", type=int, default=250,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Optimizer starting learning rate.")
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
    # NeVe
    parser = add_neve_arguments(parser)
    parser.add_argument("--scheduler-name", type=str, choices=["neve", "baseline"], default="baseline",
                        help="'neve' use NeVe scheduler. 'baseline' use use MultiStepLR.")

    # Wandb
    parser.add_argument("--wandb-project-name", type=str, default="NeVe-Federated")
    parser.add_argument("--wandb-run-name", type=str, default=None)


def _get_federated_arguments(parser):
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080",
                        help="Server address:port.")
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Number of clients in the federated learning.")


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
            print(f"Error script type: {script} is not managed yet.")
            sys.exit(-1)
    args = parser.parse_args()
    print("Arguments:", args)
    return args
