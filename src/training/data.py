from .dataset_reader import EmbeddingWebDatasetReader

def get_embedding_webdataset(urls, args):
    dl = EmbeddingWebDatasetReader(
        urls=urls,
        standard_seq_len=args.sequence_length,
        batch_size=args.batch_size,
        num_prepro_workers=args.workers,
        to_tensor=True,
        enable_text=True,
        enable_meta=False,
        embedding_transform=lambda emb: emb,
    )
    return dl


def get_data(args):
    data = {}

    if args.train_data:
        data["train"] = get_embedding_webdataset(args.train_data, args)
    if args.val_data:
        data["val"] = get_embedding_webdataset(args.val_data, args)
   
    return data
