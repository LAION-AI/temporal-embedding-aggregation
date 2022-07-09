"""training code"""
import logging

import torch

from torch import nn


def train_one_epoch(model, data, epoch, optimizer, scheduler, args, writer):

    num_batches_per_epoch = args.train_num_samples // args.batch_size
    model.train()
    loss_func = nn.CrossEntropyLoss() # right now only CLIP-Kinetics700

    dataloader = data["train"]
    # Get all kinetics700 lables
    with open("training/k700_labels.txt") as f:
        all_labels = f.read().splitlines()

    running_loss = 0.0
    for  i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        embeddings = batch["embeddings"].to(args.device)
        labs = torch.Tensor([all_labels.index(l) for l in batch["text"]]).long().to(args.device)

        optimizer.zero_grad()

        pred = model(embeddings)
        loss = loss_func(pred, labs)
        running_loss += loss.item() # maybe this doesn't make sense

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # clip grads
        optimizer.step()

        batch_count = i + 1
        if batch_count % 100 == 0 or batch == num_batches_per_epoch:
            logging.info(
                f"Train Epoch: {epoch} | {((batch_count/num_batches_per_epoch) * 100.0):.2f}% complete "
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


def evaluate(model, data, epoch, args, writer):
    dataloader = data["val"]
    # Get all kinetics700 lables
    with open("training/k700_labels.txt") as f:
        all_labels = f.read().splitlines()

    metrics = {
        "top1": 0.0,
        "top5": 0.0,
    }

    count = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            embeddings = batch["embeddings"].to(args.device)
            labs = torch.Tensor([all_labels.index(l) for l in batch["text"]])

            pred = model(embeddings).cpu()
            top5 = pred.topk(5, dim=-1).indices

            count += len(labs)
            metrics["top1"] += (top5[:, 0] == labs).sum()
            for i in range(len(labs)):
                metrics["top5"] += labs[i] in top5[i]

    for key in metrics: # turn into accuracy
        metrics[key] /= count

    for name, val in metrics.items():
        writer.add_scalar(name, val, epoch)

    logging.info(
        f"Eval epoch: {epoch} | "
        f"top1 accuracy: {metrics['top1']}"
        f"top5 accuracy: {metrics['top5']}"
    )
