import open_clip
import torch 
import os
from aggregation.factory import create_model

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.mean import Mean
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation

oc_h14, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
oc_h14.to(device)

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
        print(metrics)

ckpt_dir = "logs/"
videos_dir = "2023_01_07-09_04_31-model_self_attn_default-lr_0.001-b_667-j_6/checkpoints/"
video_ckpts = []
eval_data = []
data_loc = 'pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/oc_h14/test_fix/{000000000..000000007}.tar -'
print("Loading checkpoints...")
for i in range(1, 6):
	checkpoint = torch.load(f"{ckpt_dir}{videos_dir}epoch_{i}.pt", map_location=device)
	model_video, model_str = create_model("aggregation/model_configs/self_attn_default.json")
	model_video = model_video.to(device)
	sd = checkpoint['state_dict']
	state_dict_real = {'.'.join(a.split('.')[1:]):sd[a] for a in sd}
	model_video.load_state_dict(state_dict_real)
	video_ckpts.append(model_video)

for i in range(len(video_ckpts)):
	eval_data.append((video_ckpts[i], data_loc, True, f'video_{i+1}'))

evaluate_datasets_and_ckpts(eval_data)
