
import torch

# TODO: better way of getting models:
from aggregation.transformer import AttentionalPooler

from training.data import get_data
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate


# TODO: use this
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


DIM = 512 # for now this is constant


def main():
    args = parse_args()

    # Get data:
    data = get_data(args)

    # Device:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


    # Create model:
    # TODO: make more systematic way of initializing model:
    # TODO: define some model config from yaml or json or whatever
    model = AttentionalPooler(
        dim=DIM,
        context_dim=DIM,
        seq_len=args.sequence_length,
        heads=8,
        dim_head=64,
        proj_dim=700, # kinetics700
    ).to(args.device)

    if args.train_data:
        # Create optimizer:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        # Create scheduler:
        total_steps = (args.train_num_samples // args.batch_size) * args.epochs
        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)


    # TODO: implement some kind of experiment continuation like open_clip
    for epoch in range(args.epochs):

        train_one_epoch(model, data, epoch, optimizer, scheduler, args)

        if 'val' in data:
            evaluate(model, data, epoch, args)


        # Save checkpoint
        # TODO: implement this


if __name__ == "__main__":
    main()
