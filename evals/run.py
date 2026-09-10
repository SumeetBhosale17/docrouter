"""CLI: python -m evals.run [--answers] [-k N]"""

import argparse

from evals.harness import format_report, format_table, run


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals")
    parser.add_argument("-k", type=int, default=3, help="chunks from the dense arm")
    parser.add_argument(
        "--answers",
        action="store_true",
        help="also generate answers and score them (calls Gemini)",
    )
    args = parser.parse_args()

    before = run("dense", k=args.k, backfill=0, with_answers=args.answers)
    after = run("hybrid", k=args.k, backfill=2, with_answers=args.answers)

    print(format_report(before), "\n")
    print(format_report(after), "\n")
    print(format_table([before, after]))


if __name__ == "__main__":
    main()
