import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the news summarization demo application."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the Flask app.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Flask app.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from src.news_summary_nlp.web import app

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
