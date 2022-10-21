from aggregation.mean import Mean
import sys
import os
import open_clip
import torch

sys.path.insert(1, '/Users/daniel/Desktop/LAION_Videoclip/clip-video-encode')
from clip_video_encode.dataset import EmbeddingWebDatasetReader
from evaluation.multicaption_retrieval import multicaption_retrieval_evaluation

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

def zero_pad(e, seq_len, model_dim=512):
    out = torch.zeros(size=(seq_len, model_dim))
    out[0:len(e)] = e

    return out

def process_didemo_batch(batch, caption_sep = ';', device='cuda'):
    SEGMENT_KEY = 'times'
    embeddings = batch['embeddings']
    print(embeddings.shape)
    seq_len = embeddings.shape[1]

    #print(batch)

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

    #print(times_frames)
    #print(out_embeddings.shape)
    #print(toks.shape)
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
    #TODO: update dis to work with ViT-H/14
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion400m_e32')


    model_text = model.encode_text 
    model_video = Mean().to(device)

    ret_mets = multicaption_retrieval_evaluation(model_video, model_text, val_reader, segment=True, process_batch=process_didemo_batch)
    print(next(iter(val_reader)))
