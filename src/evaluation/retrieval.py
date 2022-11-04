import open_clip
import numpy as np
import torch
import torch.nn.functional as F


def retrieval_evaluation(model_video, model_text, data, multicaption=False):
    if type(data) == dict:
        dataloader = data["val"].dataloader
    else:
        dataloader = data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_video_features, all_text_features = [], []
    ground_truth = []
    samp = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            embeddings = batch["embeddings"]
            embeddings = embeddings.type(torch.float32)
            embeddings = F.normalize(embeddings, dim=-1)
            toks = []
            # TODO: does this require batch_size = 1 ??
            for cap in batch["text"]:
                if multicaption:
                    for c in cap.split(";"): # multiple captions separated by ;
                        toks.append(open_clip.tokenize(c))
                        ground_truth.append(samp)
                else:
                    toks.append(open_clip.tokenize(cap))
                    ground_truth.append(samp)
                samp += 1
            toks = torch.cat(toks)
            embeddings = embeddings.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            video_embeddings = model_video(embeddings)
            text_embeddings = model_text(toks).float()

            # TODO: trying
            # video_embeddings /= video_embeddings.norm(dim=-1, keepdim=True)
            # text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            video_embeddings = F.normalize(video_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

            all_video_features.append(video_embeddings.cpu())
            all_text_features.append(text_embeddings.cpu())

        val_metrics = get_metrics(
            video_features=torch.cat(all_video_features),
            text_features=torch.cat(all_text_features),
            ground_truth=ground_truth,
            logit_scale=100.0,
        )
    return val_metrics


def normalize_matrix(A):
  assert len(A.shape) == 2
  A_norm = torch.linalg.norm(A, dim = -1, keepdim = True)
  return A / A_norm


def get_metrics(video_features, text_features, ground_truth, logit_scale):
    metrics = {}

    video_features = video_features.float()
    logits_per_video = (logit_scale * video_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_video.t().detach().cpu()

    # logits_per_text = normalize_matrix(logits_per_text)


    # TODO: let's to text2video correctly and then figure out how to do video2text
    # maybe video2text is average logits over multiple captions
    '''
    # Average out over every 20 texts
    avg_per_20 = torch.zeros((video_features.shape[0], video_features.shape[0]))
    for i in range(video_features.shape[0]):
        avg_per_20[i, :] = torch.mean(logits_per_video[:, i*20:(i+1)*20], axis=-1)
    logits_per_video = avg_per_20
    '''

    # logits = {"video_to_text": logits_per_video, "text_to_video": logits_per_text}
    logits = {"text_to_video": logits_per_text}
    ground_truth = torch.tensor(ground_truth).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
