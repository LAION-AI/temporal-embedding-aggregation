"""training code"""
import logging

import torch
import numpy as np

from torch import nn
from training.loss import ClipLoss


def train_one_epoch(model_video, model_text, logit_scale, data, epoch, optimizer, scheduler, args, writer):

    num_batches_per_epoch = args.train_num_samples // args.batch_size
    model_video.train()
    loss_func = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=0,
        world_size=1,
        use_horovod=False,
    )
    dataloader = data["train"]

    running_loss = 0.0
    for  i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        embeddings = batch["embeddings"].to(args.device)
        zero_masks = batch["zero_mask"].to(args.device)
        toks = batch["text_tokens"].to(args.device)

        optimizer.zero_grad()

        video_embeddings = model_video(embeddings, zero_masks)
        with torch.no_grad():
            text_embeddings = model_text(toks).float()

        loss = loss_func(video_embeddings, text_embeddings, logit_scale.exp())
        running_loss += loss.item() # maybe this doesn't make sense

        loss.backward()
        nn.utils.clip_grad_norm_(model_video.parameters(), args.grad_clip) # clip grads
        optimizer.step()

        batch_count = i + 1
        if batch_count % 100 == 0 or batch == num_batches_per_epoch:
            logging.info(
                f"Train Epoch: {epoch} [{batch_count}/{num_batches_per_epoch} ({((batch_count/num_batches_per_epoch) * 100.0):.2f}%)] "
                f"Loss: {running_loss/100.0} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )

            log_data = {
                "loss": running_loss/100.0,
                "lr": optimizer.param_groups[0]["lr"],
            }
            for name, val in log_data.items():
                writer.add_scalar(name, val, step + 1)

            running_loss = 0.0


def evaluate(model_video, model_text, logit_scale, data, epoch, args, writer):
    dataloader = data["val"]
    model_video.eval()

    metrics = {
        "loss": 0.0
    }

    loss_func = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=0,
        world_size=1,
        use_horovod=False,
    )

    count = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            embeddings = batch["embeddings"].to(args.device)
            zero_masks = batch["zero_mask"].to(args.device)
            toks = batch["text_tokens"].to(args.device)

            video_embeddings = model_video(embeddings, zero_masks).cpu()
            text_embeddings = model_text(toks).float()
            loss = loss_func(video_embeddings, text_embeddings, logit_scale.exp())

            count += len(labs)
            metrics["loss"] += loss.item()

    for metric in metrics:
        metrics[metric] /= count

    for name, val in metrics.items():
        writer.add_scalar(name, val, epoch)

    logging.info(
        f"Eval epoch: {epoch} | "
        f"top1 accuracy: {metrics['top1']} "
        f"top5 accuracy: {metrics['top5']} "
    )
