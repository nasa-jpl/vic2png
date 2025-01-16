"""CLI frontend for converting images."""

import argparse
from pathlib import Path
from vic2png import vic2png


def main() -> None:
    """
    Main function for vic2png to be run as a script. Set up as a console script
    in the pyproject.toml
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        type=Path,
        metavar="FILE",
        help="Vicar or PDS .VIC/.IMG format file to be converted",
    )
    parser.add_argument(
        "-o", "--out", type=Path, help="Output directory or whole filename"
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default=".png",
        help="Output format, default is .png but can provide jpg or tif",
    )
    parser.add_argument(
        "-dnmax",
        type=int,
        help="Max. DN value to clip the upper bound of data in the input image.",
    )
    parser.add_argument(
        "-dnmin",
        type=int,
        help="Min. DN value to clip the lower bound of data in the input image.",
    )
    parser.add_argument(
        "--silent",
        action="store_false",
        help="If used, no output will be printed during execution.",
    )
    args: argparse.Namespace = parser.parse_args()

    source: Path = Path(args.source).resolve()
    outpath: Path | None = None
    if args.out:
        outpath = Path(args.out).resolve()

    vic2png(
        source,
        out=outpath,
        fmt=args.format,
        dnmin=args.dnmin,
        dnmax=args.dnmax,
        verbose=args.silent,
    )
