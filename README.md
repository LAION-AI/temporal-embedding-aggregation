# temporal-embedding-aggregation
Experimental repo for testing various ways of aggregating information across video frame embeddings


#Temporary notes:

EmbeddingDatasetReader isn't in the current pip install version of clip_video_encode. Either copy from the notebook, or install with setup.py.

Huggingface load_dataset is not currently implemented for the CLIP-Kinetics700 dataset (https://github.com/iejMac/clip-video-encode/issues/14). Please use:

```bash
git lfs install
git clone https://huggingface.co/datasets/iejMac/CLIP-Kinetics700
cd CLIP-Kinetics700
git lfs pull
```
