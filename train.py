import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="News summary NLP training entrypoint placeholder."
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Dataset directory for future summarization training workflows.",
    )
    return parser


def main() -> None:
    build_parser().parse_args()
    print("Training notebooks and saved models exist, but the end-to-end training pipeline is not packaged as a CLI yet.")


if __name__ == "__main__":
    main()
