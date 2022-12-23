import open_clip
import torch 
import os

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.mean import Mean
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation

oc_h14, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
# oai_b32, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
oc_h14.to(device)
# oai_b32.to(device)

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

# vc_oc_h14 = VideoCLIP(Mean(), oc_h14)
# vc_oai_b32 = VideoCLIP(Mean(), oai_b32)
from aggregation.factory import create_model

#image_dir = "2022_12_16-06_54_17-model_self_attn_default-lr_0.001-b_32-j_3/checkpoints/"
ckpt_dir = "/fsx/daniel_mend/temporal-embedding-aggregation/src/logs/"
videos_dir = "the-official-run-30k-v3/checkpoints/"
#image_ckpts = []
video_ckpts = []

#for i in range(1, 11):
#	checkpoint = torch.load(f"{ckpt_dir}{image_dir}epoch_{i}.pt", map_location=device)
#	model_video, model_str = create_model("aggregation/model_configs/self_attn_default.json")
#	model_video = model_video.to(device)
#	sd = checkpoint['state_dict']
#	state_dict_real = {'.'.join(a.split('.')[1:]):sd[a] for a in sd}
#	model_video.load_state_dict(state_dict_real)
#	image_ckpts.append(model_video)
#
#
print("Loading checkpoints...")
for i in range(2, 3):
	checkpoint = torch.load(f"{ckpt_dir}{videos_dir}epoch_{i}.pt", map_location=device)
	model_video, model_str = create_model("aggregation/model_configs/self_attn_default_depth2.json")
	model_video = model_video.to(device)
	sd = checkpoint['state_dict']
	state_dict_real = {'.'.join(a.split('.')[1:]):sd[a] for a in sd}
	model_video.load_state_dict(state_dict_real)
	video_ckpts.append(model_video)
data_loc =  'pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/oc_h14/test_fix/{000000000..000000007}.tar -'
from torch import nn

class ModifiedMean(nn.Module):
    def __init__(self, max_frames=500):
        super().__init__()
        self.max_frames = max_frames
    def forward(self, x):
        if len(x) > self.max_frames:
            x = x[:self.max_frames]
        return torch.mean(x, axis=-2)
test_txt = "hi there!"
'''
checkpoint = torch.load(f"{ckpt_dir}{videos_dir}epoch_latest.pt", map_location=device)
model_video, model_str = create_model("aggregation/model_configs/self_attn_default.json")
model_video = model_video.to(device)
sd = checkpoint['state_dict']
state_dict_real = {'.'.join(a.split('.')[1:]):sd[a] for a in sd}
model_video.load_state_dict(state_dict_real)
'''
#toks = open_clip.tokenize(test_txt)
#toks = toks.to(device)

#emb1 = oc_h14.encode_text(toks)
#emb2 = model_video.encode_text(toks)

#print((emb1 - emb2))

#video_ckpts.append(model_video)
#vc = VideoCLIP(ModifiedMean(max_frames=500), oc_h14)
#eval_data = [(vc, data_loc, True, 'test')]
eval_data = []

#for i in range(len(image_ckpts)):
#	eval_data.append((image_ckpts[i], data_loc, True, f'image_{i}'))

for i in range(len(video_ckpts)):
	eval_data.append((video_ckpts[i], data_loc, True, f'video_{i+1}'))

evaluate_datasets_and_ckpts(eval_data)
