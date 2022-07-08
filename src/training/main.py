from training.data import get_data
from training.params import parse_args


def main():
    args = parse_args()

    data = get_data(args)

    # TODO: implement some kind of experiment continuation like open_clip

    for epoch in range(args.epochs):
        pass

if __name__ == "__main__":
    main()
