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
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
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
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    # distributed stuff
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )

    # Model:
    parser.add_argument(
        "--model",
        default="aggregation/model_configs/self_attn_default.json",
        type=str,
        help="path to valid model config",
    )

    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=1000, help="Number of steps to warmup for."
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="gradient clipping")


    # Logging
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store logs. Use None to avoid storing logs.",
    ) 
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="",
        help="Log to 'tensorboard' or 'wandb'",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )

    args = parser.parse_args()
    return args
