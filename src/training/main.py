import logging
import os
import random
import wandb

from datetime import datetime

import open_clip
import numpy as np
import torch
import torch.utils.tensorboard as tensorboard

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from aggregation.factory import create_model
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    args = parse_args()

    conf_name = args.model.split("/")[-1][:-5]
    # get the name of the experiments
    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{conf_name}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
        ])

    # Device:
    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # Set up logging:
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)
        if not args.resume and os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

        args.log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = init_distributed_device(args)

    if is_master(args):
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard")
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    # Get data:
    preprocess = (torch.nn.Identity(), torch.nn.Identity())
    data = get_data(args, preprocess)

    # Create model:
    random_seed(args.seed)
    
    model_video, model_str = create_model(args.model)
    model_video.to(args.device)

    logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    # Make model distributed
    if args.distributed and not args.horovod:
        model_video = torch.nn.parallel.DistributedDataParallel(model_video, device_ids=[device])

    if args.train_data:
        # Create optimizer:
        optimizer = torch.optim.AdamW(
            model_video.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        # Create scheduler:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # Writing:
    writer = None
    if args.report_to == "tensorboard" and is_master(args):
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    elif args.report_to == "wandb" and is_master(args):
        logging.debug("Starting wandb.")
        wandb.init(
            project="video-clip",
            entity="iejmac", #TODO: do you need this?
            name=args.name,
            config=vars(args),
        )
        if args.debug: # TODO test this out
            wandb.watch(model_video, log='all')

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
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model_video.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model_video.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info(f"=> no checkpoint found at '{args.resume}'")


    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        train_one_epoch(model_video, data, epoch, optimizer, scheduler, args, writer)
        completed_epoch = epoch + 1

        if 'val' in data:
            evaluate(model_video, data, epoch, args, writer)

        # Save checkpoint
        if is_master(args):
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model_video.state_dict(),
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

    if args.report_to == "wandb" and is_master(args):
        wandb.finish()


if __name__ == "__main__":
    main()
