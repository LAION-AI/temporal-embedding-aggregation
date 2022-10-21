import numpy as np
import torch


def retrieval_evaluation(model_video, model_text, data):
    if type(data) == dict:
        dataloader = data["val"].dataloader
    else:
        dataloader = data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = data
    all_video_features, all_text_features = [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # embeddings, toks = batch
            embeddings = batch["embeddings"]
            toks = batch["text_tokens"]
            embeddings = embeddings.to(device, non_blocking=True)
            toks = toks.to(device, non_blocking=True)

            # video_embeddings = model_video(embeddings, None)
            video_embeddings = model_video(embeddings)
            text_embeddings = model_text(toks).float()

            all_video_features.append(video_embeddings.cpu())
            all_text_features.append(text_embeddings.cpu())

        val_metrics = get_metrics(
            video_features=torch.cat(all_video_features),
            text_features=torch.cat(all_text_features),
            logit_scale=100.0,
        )
    return val_metrics


def get_metrics(video_features, text_features, logit_scale):
    metrics = {}

    video_features = video_features.float()
    logits_per_video = (logit_scale * video_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_video.t().detach().cpu()

    logits = {"video_to_text": logits_per_video, "text_to_video": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
