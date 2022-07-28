import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.utils.tensorboard as tensorboard


# TODO: better way of getting models:
# from aggregation.transformer import AttentionalPooler
from aggregation.transformer2 import AttentionalPooler

from training.data import get_data
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate


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
            f"depth_{args.depth}",
            f"dropout_{args.dropout}",
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

    # Get data:
    data = get_data(args)

    # Device:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model:
    random_seed(args.seed)
    # TODO: make more systematic way of initializing model:
    # TODO: define some model config from yaml or json or whatever
    model = AttentionalPooler(
        dim=DIM,
        context_dim=DIM,
        seq_len=args.sequence_length,
        heads=8,
        dim_head=64,
        depth=args.depth,
        proj_dim=700, # kinetics700
        dropout=args.dropout,
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

    # Resume from checkpoint
    start_epoch = 0
    if args.resume is not None:
        # NOTE: resuming doesn't work with torch >1.11.0 yet (https://github.com/pytorch/pytorch/issues/80809)
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=args.device)
            if 'epoch' in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")



    for epoch in range(start_epoch, args.epochs):
        logging.info(f'Start epoch {epoch}')
        train_one_epoch(model, data, epoch, optimizer, scheduler, args, writer)
        completed_epoch = epoch + 1

        if 'val' in data:
            evaluate(model, data, epoch, args, writer)

        # Save checkpoint
        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }               
        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )
        if args.save_most_recent:
            torch.save(
                checkpoint_dict,
                os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
            )



if __name__ == "__main__":
    main()
