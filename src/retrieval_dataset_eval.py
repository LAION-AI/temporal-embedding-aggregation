import torch
import os
import argparse
from aggregation.factory import create_model
from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation
import time
import wandb

def evaluate_datasets_and_ckpts(eval_data, args):
    for ckpt, tar, multicaption in eval_data:
        assert isinstance(ckpt, VideoCLIP)
        val_reader = EmbeddingWebDatasetReader(
            tar,
            standard_seq_len=args.seq_len,
            batch_size=1,
            num_prepro_workers=8,
            to_tensor=False,
            enable_text=True,
            enable_meta=True
        )

        metrics = retrieval_evaluation(ckpt, val_reader, multicaption)
        return metrics

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
    args = parser.parse_args()
    return args

def load_checkpoint(ckpt, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(ckpt, map_location=device)
    model_video, model_str = create_model(args.cfg)
    model_video = model_video.to(device)
    sd = checkpoint['state_dict']
    state_dict_real = {'.'.join(a.split('.')[1:]):sd[a] for a in sd}
    model_video.load_state_dict(state_dict_real)
    return model_video

def main():
    args = parse_args()
    seen_checkpoints = set()
    name = args.checkpoint_dir.split('/')[-3]
    wandb.init(
        project="laion-video-clip",
        name=name,
    )
    while True:
        checkpoints = os.listdir(args.checkpoint_dir)
        checkpoints = [x for x in checkpoints if 'latest' not in x]
        checkpoints = sorted(checkpoints, key=lambda i: int(i.split('_')[1].split('.')[0]))
        for checkpoint in checkpoints:
            if checkpoint not in seen_checkpoints:
                seen_checkpoints.add(checkpoint)
                epoch = checkpoint.split('_')[1].split('.')[0]
                full_ckpt_path = args.checkpoint_dir + checkpoint
                print(checkpoint, epoch)
                model_video = load_checkpoint(full_ckpt_path, args)
                metrics = evaluate_datasets_and_ckpts([(model_video, args.val_data, True)], args)
                for metric, value in metrics.items():
                     print(metric, value)
                     wandb.log({f'eval/{metric}': value, 'step': epoch})

        time.sleep(120)

if __name__ == "__main__":
    main()
