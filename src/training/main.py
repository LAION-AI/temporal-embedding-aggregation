import logging
import os

from datetime import datetime

import torch
import torch.utils.tensorboard as tensorboard


# TODO: better way of getting models:
from aggregation.transformer import AttentionalPooler

from training.data import get_data
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate


# TODO: use this
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


DIM = 512 # for now this is constant


def main():
    args = parse_args()

    # TODO: model from params
    args.model = "attentionalpooling"

    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
        ])

    # Set up logging:
    log_base_path = os.path.join(args.logs, args.name)
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = "out.log"
    args.log_path = os.path.join(log_base_path, log_filename)
    if os.path.exists(args.log_path):
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard")
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    for dirname in [args.tensorboard_path, args.checkpoint_path]:
        os.makedirs(dirname, exist_ok=True)

    writer = tensorboard.SummaryWriter(args.tensorboard_path)

    # Resume from checkpoint
    #TODO: implement this.

    # Get data:
    data = get_data(args)

    # Device:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model:
    # TODO: make more systematic way of initializing model:
    # TODO: define some model config from yaml or json or whatever
    model = AttentionalPooler(
        dim=DIM,
        context_dim=DIM,
        seq_len=args.sequence_length,
        heads=8,
        dim_head=64,
        proj_dim=700, # kinetics700
    ).to(args.device)

    if args.train_data:
        # Create optimizer:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        # Create scheduler:
        total_steps = (args.train_num_samples // args.batch_size) * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)


    # TODO: implement some kind of experiment continuation like open_clip
    for epoch in range(args.epochs):
        logging.info(f'Start epoch {epoch}')
        train_one_epoch(model, data, epoch, optimizer, scheduler, args, writer)

        if 'val' in data:
            evaluate(model, data, epoch, args, writer)

        # Save checkpoint
        # TODO: implement this


if __name__ == "__main__":
    main()
