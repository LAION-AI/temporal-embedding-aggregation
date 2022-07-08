"""training code"""
import torch

from torch import nn


def train_one_epoch(model, data, epoch, optimizer, scheduler, args):

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
            print(f"epoch {epoch} : step {step + 1} average loss = {running_loss/100}")
            running_loss = 0.0


def evaluate(model, data, epoch, args):

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

    for key in metrics:
        metrics[key] /= count
    print(metrics)
