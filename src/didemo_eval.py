from aggregation.mean import Mean
import sys
import os
import open_clip
import torch

sys.path.insert(1, '/Users/daniel/Desktop/LAION_Videoclip/clip-video-encode')
from clip_video_encode.dataset import EmbeddingWebDatasetReader
from evaluation.retrieval import retrieval_evaluation

eval_path = '/Users/daniel/Documents/GitHub/temporal-embedding-aggregation/CLIP-DiDeMo/data/oc_h14/test/'

def process_times(times, len_embeddings):
    '''
    Assumptions: 
    - times contains [start, end] in intervals of 5 seconds, 
    i.e. [0, 1] corresponds to [0*5, (1+1)*5]

    - there's 1 embedding per second, i.e. embeddings[10] is the sole embedding for the 10th second
    '''
    SEGMENT_INTERVAL = 5
    return [
        SEGMENT_INTERVAL * times[0], 
        min( SEGMENT_INTERVAL * (times[1] + 1), len_embeddings )
    ]

def zero_pad(e, seq_len, model_dim=1024):
    out = torch.zeros(size=(seq_len, model_dim))
    out[0:len(e)] = e

    return out

def process_didemo_segments(embeddings, segments, seq_len=200):
    times_frames = [
        process_times(caption_segments[0], seq_len)
        for caption_segments in segments
    ]

    out_embeddings = torch.stack([
        zero_pad(embeddings[:, start:end, :].squeeze(0), seq_len)
        for (start, end) in times_frames
    ])

    return out_embeddings


def process_didemo_batch(batch, caption_sep = ';', device='cuda'):
    SEGMENT_KEY = 'times'
    embeddings = batch['embeddings']
    
    seq_len = embeddings.shape[1]
    captions = [
        text.split(caption_sep) 
        for text in batch['text']
    ]

    times_frames = [
        process_times(caption_segments[0], seq_len)
        for caption_segments in batch['meta'][SEGMENT_KEY]
    ] # just take the first annotation for each caption

    toks = torch.stack([
        open_clip.tokenize(caption).to(device)
        for caption in captions
    ]).squeeze(0)

    out_embeddings = torch.stack([
        zero_pad(embeddings[:, start:end, :].squeeze(0), seq_len)
        for (start, end) in times_frames
    ])

    return out_embeddings, toks


if __name__ == "__main__":
    val_urls = eval_path + '{000000000..000000007}.tar'
    val_reader = EmbeddingWebDatasetReader(
        val_urls,
        standard_seq_len=200,
        batch_size=1,
        num_prepro_workers=0,
        to_tensor=False,
        enable_text=True,
        enable_meta=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')

    model_text = model.encode_text
    model_video = Mean().to(device)

    ret_mets = retrieval_evaluation(model_video, model_text, val_reader, multicaption=True, segment=True, segment_key='times', process_segments=process_didemo_segments)
    print(ret_mets)
