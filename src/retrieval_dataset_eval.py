import torch
import argparse
from aggregation.factory import create_model
from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation
import wandb

def evaluate_datasets_and_ckpts(eval_data):
    for ckpt, tar, multicaption, name in eval_data:
        print(f'Eval for {name}')
        assert isinstance(ckpt, VideoCLIP)
        val_reader = EmbeddingWebDatasetReader(
            tar,
            standard_seq_len=200,
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
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Location of webdataset tar files with evaluation data" 
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of videoclip run"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log metrics to wandb"
    )
    args = parser.parse_args()
    return args

def load_checkpoint(checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_video, model_str = create_model("aggregation/model_configs/self_attn_default.json")
    model_video = model_video.to(device)
    sd = checkpoint['state_dict']
    state_dict_real = {'.'.join(a.split('.')[1:]):sd[a] for a in sd}
    model_video.load_state_dict(state_dict_real)
    return model_video

def main():
    args = parse_args()
    print(args.checkpoint_path)
    epoch = int(args.checkpoint_path.split('/')[-1].split('_')[1].split('.')[0])
    model_video = load_checkpoint(args.checkpoint_path)
    eval_data = [(model_video, args.val_data, True, 'videoclp')]
    metrics = evaluate_datasets_and_ckpts(eval_data)
    if args.wandb:
        wandb.init(
             project="laion-video-clip",
             resume="auto",
             name=args.name
        )

        for metric, value in metrics.items():
             wandb.log({f'eval/{metric}': value, 'step': epoch})

if __name__ == "__main__":
    main()

'''
ckpt_dir = "/fsx/daniel_mend/videoclip_logs/logs/"
videos_dir = "2022_12_31-03_44_03-model_self_attn_default-lr_0.001-b_500-j_6/checkpoints/"
ckpt_dir = "logs/"
videos_dir = "2023_01_07-09_04_31-model_self_attn_default-lr_0.001-b_667-j_6/checkpoints/"
video_ckpts = []
eval_data = []
data_loc = 'pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/oc_h14/test_fix/{000000000..000000007}.tar -'
print("Loading checkpoints...")
for i in range(1, 2):
		video_ckpts.append(model_video)

for i in range(len(video_ckpts)):
	eval_data.append((video_ckpts[i], data_loc, True, f'video_{i+1}'))

evaluate_datasets_and_ckpts(eval_data)
'''
