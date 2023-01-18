import torch
import os
import argparse
from aggregation.factory import create_model
from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation
import time
import wandb
from functools import partial
import multiprocessing
from multiprocessing import Pool

def evaluate(ckpt, epoch, device, args):
    assert isinstance(ckpt, VideoCLIP)
    val_reader = EmbeddingWebDatasetReader(
        args.val_data,
        standard_seq_len=args.seq_len,
        batch_size=1,
        num_prepro_workers=0,
        to_tensor=False,
        enable_text=True,
        enable_meta=True
    )
    ckpt = ckpt.to(device, non_blocking=True)
    metrics = retrieval_evaluation(ckpt, val_reader, args.multicaption)
    return {'metrics': metrics, 'epoch': epoch}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Dir with checkpoints",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Location of webdataset tar files with evaluation data"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=200,
        help="Sequence length for transformer aggregator"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Model config for eval"
    )
    parser.add_argument(
        "--multicaption",
        type=bool,
        default=True,
        help="Whether or not to do multicaption eval"
    )
    args = parser.parse_args()
    return args

def multiprocess_eval(ckpt, args):
    rank = multiprocessing.current_process()._identity[0]-1

    epoch = int(ckpt.split('.')[-2].split('_')[-1])
    device = f'cuda:{rank}'
    checkpoint = torch.load(ckpt, map_location=device)
    model_video, model_str = create_model(args.cfg)
    model_video = model_video.to(device, non_blocking=True)
    sd = checkpoint['state_dict']
    state_dict_real = {'.'.join(a.split('.')[1:]):sd[a] for a in sd}
    model_video.load_state_dict(state_dict_real)
    return evaluate(model_video, epoch, device, args)

def main():
    args = parse_args()
    seen_checkpoints = set()
    name = args.checkpoint_dir.split('/')[-3]
    wandb.init(
        project="laion-video-clip",
        name=name,
    )
    with Pool(processes=8) as pool:
        while True:
            checkpoints = os.listdir(args.checkpoint_dir)
            checkpoints = [x for x in checkpoints if 'latest' not in x]
            checkpoints = sorted(checkpoints, key=lambda i: int(i.split('_')[1].split('.')[0]))
            checkpoints = [args.checkpoint_dir + ckpt for ckpt in checkpoints]
            checkpoints = [x for x in checkpoints if x not in seen_checkpoints]
            for checkpoint in checkpoints:
                seen_checkpoints.add(checkpoint)

            for eval in pool.imap_unordered(partial(multiprocess_eval, args=args), checkpoints):
                metrics = eval['metrics']
                epoch = eval['epoch']
                for metric, value in metrics.items():
                    wandb.log({f'eval/{metric}': value, 'epoch': epoch})
            if len(checkpoints) == 0:
                time.sleep(300)

if __name__ == "__main__":
    main()
