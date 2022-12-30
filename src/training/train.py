"""training code"""
import logging
import wandb

import torch
import numpy as np

from torch import nn
from training.loss import ClipLoss
from .distributed import is_master
import time

from torch.cuda.amp import autocast

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
        img_iter = iter(
            data["img_reader"](
                batch_size=args.image_batch_size,
                start=data["worker_start"],
                end=data["worker_end"],
                show_progress=False
            )
        )
        text_iter = iter(
            data["img_txt_reader"](
                batch_size=args.image_batch_size,
                start=data["worker_start"],
                end=data["worker_end"],
                show_progress=False
            )
        )

    num_batches_per_epoch = dataloader.num_batches

    running_video_loss = 0.0
    running_image_loss = 0.0
    running_loss = 0.0
    running_logit_scale = 0.0
    global_batch_size = args.world_size * args.batch_size
    print(global_batch_size)
    start = time.time()
    times = {}
    t = time.time()
    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        embeddings, toks = batch

        embeddings = embeddings.to(device, non_blocking=True)
        toks = toks.to(device, non_blocking=True)

        optimizer.zero_grad()
        dims = embeddings.shape
        times['dataloader_video'] = times.get('dataloader_video', 0) + time.time()-t
        t = time.time()
        with autocast():
            video_embeddings, text_embeddings, logit_scale = model_video(embeddings, toks, prenorm=True, postnorm=True)
            times['forward_video'] = times.get('forward_video', 0) + time.time()-t
            t = time.time()
            loss_video = loss_func(video_embeddings, text_embeddings, logit_scale)
            times['loss_video'] = times.get('loss_video', 0) + time.time()-t
        t = time.time()

        running_video_loss += loss_video.item() # maybe this doesn't make sense
        running_loss += loss_video.item()
        loss_video.backward()
        nn.utils.clip_grad_norm_(model_video.parameters(), args.grad_clip) # clip grads
        optimizer.step()
        times['backward_video'] = times.get('backward_video', 0) + time.time()-t
        t = time.time()
        embeddings = None
        video_embeddings = None
        text_embeddings = None
        if args.image_data:
            optimizer.zero_grad()
            try:
                img_embeddings, _ = next(img_iter)
            except:
                continue
            img_embeddings = torch.tensor(img_embeddings)

            batch_padded_img_embeddings = torch.zeros(dims[0], dims[2])
            batch_padded_img_embeddings[:len(img_embeddings), :] = img_embeddings

            vid_emb = torch.zeros(args.image_batch_size, dims[1], dims[2])

            vid_emb[:, 0, :] = batch_padded_img_embeddings
            vid_emb = vid_emb.to(device, non_blocking=True)
            try:
                 txt_emb, _ = next(text_iter)
            except:
                 continue
            txt_emb = torch.tensor(txt_emb)
            batch_padded_txt_emb = torch.zeros(dims[0], dims[2])
            batch_padded_txt_emb[:len(txt_emb), :] = txt_emb
            batch_padded_txt_emb = batch_padded_txt_emb.to(device, non_blocking=True)
            # print(img_embeddings.float().cuda() @ txt_emb.cuda().t().float())
            times['dataloader_image'] = times.get('dataloader_image', 0) + time.time()-t
            t = time.time()
            with autocast():
                vid_emb, _, logit_scale = model_video(vid_emb, toks, prenorm=True, postnorm=True, encode_text=False)
                times['forward_image'] = times.get('forward_image', 0) + time.time()-t
                t = time.time()
                loss_image = loss_func(vid_emb, batch_padded_txt_emb, logit_scale)
            times['loss_image'] = time.time()-t
            t = time.time()
            running_image_loss += loss_image.item()
            running_loss += loss_image.item()

            loss_image.backward()
            nn.utils.clip_grad_norm_(model_video.parameters(), args.grad_clip) # clip grads
            optimizer.step()
            times['backward_image'] = times.get('backward_image', 0) + time.time()-t
            t = time.time()
            vid_emb = None
            txt_emb = None
            img_embeddings = None
        batch_count = i+1
        if is_master(args) and (batch_count % 10 == 0 or batch == num_batches_per_epoch):
            time_for_batches = time.time()-start
            print(f'Time for batches: {time_for_batches}')
            print(f'Mean time for batches: {time_for_batches/10}')

            bs_times = {x:global_batch_size*10/(times[x]*args.world_size) for x in times}
            print(f'Raw times: {times}')
            print(f'Samples/s: {bs_times}')
            start = time.time()
            times = {}
            samples_per_second = global_batch_size*10/time_for_batches
            samples_per_second_per_gpu = samples_per_second/args.world_size
            if args.image_data:
                samples_per_second *= 2
                samples_per_second_per_gpu *= 2
            logging.info(
                f"Train Epoch: {epoch} [{batch_count}/{num_batches_per_epoch} ({((batch_count/num_batches_per_epoch) * 100.0):.2f}%)] "
                f"Loss: {running_loss/10.0} "
                f"Video loss: {running_video_loss/10.0} "
                f"Image loss: {running_image_loss/10.0} "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit scale: {logit_scale.item()} "
                f"Samples/s: {samples_per_second} "
                f"Samples/s/gpu: {samples_per_second_per_gpu} "
            )

            log_data = {
                "train_loss": running_loss/10.0,
                "video_loss": running_video_loss/10.0,
                "image_loss": running_image_loss/10.0,
                "lr": optimizer.param_groups[0]["lr"],
                "logit_scale":logit_scale.item(),
                "samples_per_s":samples_per_second,
                "samples_per_s_per_gpu":samples_per_second_per_gpu
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
        times['logging'] = times.get('logging', 0) + time.time()-t
        t = time.time()

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
