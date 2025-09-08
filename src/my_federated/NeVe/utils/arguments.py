import argparse


def _int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def add_neve_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--neve-momentum", type=float, default=0.5,
                        help="NeVe - Velocity momentum.")
    parser.add_argument("--neve-only-ll", type=_int2bool, choices=[0, 1], default=True,
                        help="NeVe - Hooks only for the last layer or whole model.")
    parser.add_argument("--neve-velocity-inverted", type=_int2bool, choices=[0, 1], default=False,
                        help="NeVe - Invert the velocity from lowest to highest.")
    parser.add_argument("--neve-velocity-aggregation-fn", type=str,
                        choices=["avg", "soft_exp"], default="avg",
                        help="NeVe - Aggregation function (avg, soft_exp, etc..).")
    parser.add_argument("--neve-use-distribution", type=_int2bool, choices=[0, 1], default=True,
                        help="NeVe - Normalize velocity value by the training class distribution.")
    parser.add_argument("--neve-softmax-temperature", type=float, default=1.0,
                        help="NeVe - Softmax temperature.")

    return parser
