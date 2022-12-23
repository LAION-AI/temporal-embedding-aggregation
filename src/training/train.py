"""training code"""
import logging
import wandb

import torch
import numpy as np

from torch import nn
from training.loss import ClipLoss
from .distributed import is_master

from embedding_reader import EmbeddingReader

def train_one_epoch(model_video, data, epoch, optimizer, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    model_video.train()
    loss_func = ClipLoss(
        local_loss=False,
        gather_with_grad=True,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=False,
    )
    
    dataloader = data["train"].dataloader
    
    if args.image_data:
        embeddings_images = EmbeddingReader(
            embeddings_folder=f'{args.image_data}/img_emb/',
            file_format='npy'
        )
        embeddings_txt = EmbeddingReader(
            embeddings_folder=f'{args.image_data}/text_emb/',
            file_format = 'npy'
        )

        img_iter = iter(embeddings_images(batch_size=args.image_batch_size, start=0, end=dataloader.num_batches*args.image_batch_size, show_progress=False))
        text_iter = iter(embeddings_txt(batch_size=args.image_batch_size, start=0, end=dataloader.num_batches*args.image_batch_size, show_progress=False))

    num_batches_per_epoch = dataloader.num_batches

    running_video_loss = 0.0
    running_image_loss = 0.0
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        embeddings, toks = batch

        embeddings = embeddings.to(device, non_blocking=True)
        toks = toks.to(device, non_blocking=True)

        optimizer.zero_grad()
        dims = embeddings.shape
        video_embeddings, text_embeddings, logit_scale = model_video(embeddings, toks, prenorm=True, postnorm=True)
        loss_video = loss_func(video_embeddings, text_embeddings, logit_scale)
        running_video_loss += loss_video.item() # maybe this doesn't make sense
        running_loss += loss_video.item()

        loss_video.backward()
        nn.utils.clip_grad_norm_(model_video.parameters(), args.grad_clip) # clip grads
        optimizer.step()
        embeddings = None
        video_embeddings = None
        text_embeddings = None
        if args.image_data:
            optimizer.zero_grad()
            img_embeddings, _ = next(img_iter)
            img_embeddings = torch.tensor(img_embeddings)
            
            vid_emb = torch.zeros(args.image_batch_size, dims[1], dims[2])
            
            vid_emb[:, 0, :] = img_embeddings
            vid_emb = vid_emb.to(device, non_blocking=True)

            txt_emb, _ = next(text_iter)
            txt_emb = torch.tensor(txt_emb)
            txt_emb = txt_emb.to(device, non_blocking=True)
            
            vid_emb, _, logit_scale = model_video(vid_emb, toks, prenorm=True, postnorm=True)
            loss_image = loss_func(vid_emb, txt_emb, logit_scale)
            running_image_loss += loss_image.item()
            running_loss += loss_image.item()

            loss_image.backward()
            nn.utils.clip_grad_norm_(model_video.parameters(), args.grad_clip) # clip grads
            optimizer.step()
            
            vid_emb = None
            txt_emb = None
            img_embeddings = None
        batch_count = i + 1
        if is_master(args) and (batch_count % 10 == 0 or batch == num_batches_per_epoch):
            logging.info(
                f"Train Epoch: {epoch} [{batch_count}/{num_batches_per_epoch} ({((batch_count/num_batches_per_epoch) * 100.0):.2f}%)] "
                f"Loss: {running_loss/10.0} "
                f"Video loss: {running_video_loss/10.0} "
                f"Image loss: {running_image_loss/10.0} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
            )

            log_data = {
                "train_loss": running_loss/10.0,
                "video_loss": running_video_loss/10.0,
                "image_loss": running_image_loss/10.0,
                "lr": optimizer.param_groups[0]["lr"],
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)
                if args.report_to == "wandb":
                    wandb.log({name: val, 'step': step}, step=step)

            running_loss = 0.0
            running_video_loss = 0.0
            running_image_loss = 0.0

def evaluate(model_video, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    dataloader = data["val"].dataloader
    model_video.eval()

    metrics["val_loss"] = 0.0
    loss_func = ClipLoss(
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        rank=0,
        world_size=1,
        use_horovod=False,
    )

    count = 0.0
    all_video_features, all_text_features = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            embeddings, toks = batch
            embeddings = embeddings.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            video_embeddings, text_embeddings, logit_scale = model_video(embeddings, toks, prenorm=True, postnorm=True)

            all_video_features.append(video_embeddings.cpu())
            all_text_features.append(text_embeddings.cpu())
            loss = loss_func(video_embeddings, text_embeddings, logit_scale.exp())

            count += 1
            metrics["val_loss"] += loss.item()

        val_metrics = get_metrics(
            video_features=torch.cat(all_video_features),
            text_features=torch.cat(all_text_features),
            logit_scale=logit_scale.cpu(),
        )
        metrics.update(**val_metrics)

    metrics["val_loss"] /= count

    for name, val in metrics.items():
        name = "val/" + name
        if tb_writer is not None:
            tb_writer.add_scalar(name, val, epoch)
        if args.report_to == "wandb":
            wandb.log({name: val, 'epoch': epoch}, step=epoch)

    if is_master(args):
        logging.info(
            f"Eval epoch: {epoch} | "
            f"loss : {metrics['val_loss']} "
        )

def get_metrics(video_features, text_features, logit_scale):
    metrics = {}
    logits_per_video = (logit_scale * video_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_video.t().detach().cpu()

    logits = {"video_to_text": logits_per_video, "text_to_video": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
