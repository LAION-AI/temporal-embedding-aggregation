import open_clip

from clip_video_encode.dataset import EmbeddingWebDatasetReader
from aggregation.mean import Mean
from aggregation.aggregator_wrapper import VideoCLIP
from evaluation.retrieval import retrieval_evaluation

oc_h14, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
oai_b32, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

def evaluate_datasets_and_ckpts(eval_data):
    for ckpt, tar, multicaption in eval_data:
        assert isinstance(ckpt, VideoCLIP)
        val_reader = EmbeddingWebDatasetReader(
            tar,
            standard_seq_len=200,
            batch_size=1,
            num_prepro_workers=6,
            to_tensor=False,
            enable_text=True,
            enable_meta=True
        )
        
        metrics = retrieval_evaluation(ckpt, val_reader, multicaption)
        print(metrics)

vc_oc_h14 = VideoCLIP(Mean(), oc_h14)
vc_oai_b32 = VideoCLIP(Mean(), oai_b32)

eval_data = [
    (vc_oc_h14, 'pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/oc_h14/test/{000000000..000000007}.tar -', True),
    (vc_oai_b32, 'pipe:aws s3 cp s3://s-laion/msr_vtt/clip_msr_vtt/oai_b32/test/{000000000..000000007}.tar -', True),
    (vc_oc_h14, 'pipe:aws s3 cp s3://s-laion/msvd/clip_msvd/oc_h14/test/000000000.tar -', True),
    (vc_oai_b32, 'pipe:aws s3 cp s3://s-laion/msvd/clip_msvd/oai_b32/test/000000000.tar -', True),
]

evaluate_datasets_and_ckpts(eval_data)