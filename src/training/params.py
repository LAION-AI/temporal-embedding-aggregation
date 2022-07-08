import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Data location
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Tar files containing training data"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Tar files containing val data"
    )

    # Data shape
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Standard sequence length of all embedding sequences (by zero-pad or crop end)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size of embedding sequences - (batch_size, sequence_length, emb_dim)"
    )

    # Data speed
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of DataLoader workers"
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for",
    )

    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient clipping")



    args = parser.parse_args()
    return args
