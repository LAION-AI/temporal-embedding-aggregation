from training.data import get_data
from training.params import parse_args


def main():
    args = parse_args()

    data = get_data(args)

if __name__ == "__main__":
    main()
