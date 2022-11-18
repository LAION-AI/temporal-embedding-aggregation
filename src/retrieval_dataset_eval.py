import open_clip
import torch 

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.mean import Mean
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation

oc_h14, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
oai_b32, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
oc_h14.to(device)
oai_b32.to(device)

def evaluate_datasets_and_ckpts(eval_data):
    for ckpt, tar, multicaption in eval_data:
        assert isinstance(ckpt, VideoCLIP)
        val_reader = EmbeddingWebDatasetReader(
            tar,
            standard_seq_len=-1,
            batch_size=1,
            num_prepro_workers=8,
            to_tensor=False,
            enable_text=True,
            enable_meta=True
        )
        
        metrics = retrieval_evaluation(ckpt, val_reader, multicaption)
        print(metrics)

vc_oc_h14 = VideoCLIP(Mean(), oc_h14)
vc_oai_b32 = VideoCLIP(Mean(), oai_b32)

eval_data = [
    (vc_oc_h14, 'CLIP-MSR-VTT/data/oc_h14/test/{000000000..000000007}.tar', True),
    (vc_oai_b32, 'CLIP-MSR-VTT/data/oai_b32/test_full_fps/{000000000..000000007}.tar', True),
    (vc_oc_h14, 'CLIP-MSVD/data/oc_h14/test/000000000.tar', True),
    (vc_oai_b32, 'CLIP-MSVD/data/oai_b32/test/000000000.tar', True),
]

evaluate_datasets_and_ckpts(eval_data)