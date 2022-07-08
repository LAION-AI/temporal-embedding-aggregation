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
        "--epoch",
        type=int,
        default=1,
        help="Number of epochs to train for",
    )

    args = parser.parse_args()
    return args
