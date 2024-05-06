import argparse


def _int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def add_neve_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--neve-momentum", type=float, default=0.5,
                        help="NeVe - Velocity momentum.")
    parser.add_argument("--neve-epsilon", type=float, default=1e-3,
                        help="NeVe - Early-stop velocity threshold.")
    parser.add_argument("--neve-alpha", type=float, default=0.5,
                        help="NeVe - Scheduler rescaling factor.")
    parser.add_argument("--neve-delta", type=int, default=10,
                        help="NeVe - Scheduler epochs of patience.")
    parser.add_argument("--neve-only-ll", type=_int2bool, choices=[0, 1], default=False,
                        help="NeVe - Hooks only for the last layer or whole model.")
    parser.add_argument("--neve-use-lr-scheduler", type=_int2bool, choices=[0, 1], default=True,
                        help="NeVe - Use a scheduling for the LR. If False it will be constant.")

    return parser
