#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2024, 2025 Jacqueline Ryan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import numpy as np
from pathlib import Path, PurePath
from PIL import Image
import numpy.typing as npt
from typing import Tuple

from . import reader


INTMAX = {1: 255, 2: 4095, 4: 65535}
MAXDEFAULT = 4095

SUPPORTED_FORMATS = [".png", ".jpg", ".tif"]


def validate_dn_range(
    raw_dnmin: int | None,
    raw_dnmax: int | None,
    arr_min: int,
    arr_max: int,
    dtype: npt.DTypeLike,
) -> Tuple[int, int]:
    """
    Handle the dnmin and dnmax parameters
    Control flow:
       -> Use the provided dnmin if it exists. If it is negative, set it to 0.
          -> if no dnmin option was provided through the cli (or by the caller), use the
             image data's min value.
       -> Use the provided dnmax if it exists. If it is greater than the max allowed
          for the data type, truncate to the max. If the dnmax is less than the dnmin,
          raise it to be equal to the dnmin
          -> if no dnmax option was provided through the cli (or by the caller), use the
             image data's max value.

    :param raw_dnmin:  Max. DN value to clip the upper bound of data in the input image.
    :param raw_dnmax:  Min. DN value to clip the upper bound of data in the input image.
    :param arr_min: Min pixel value observed in the image. Used as a default if dnmin is None.
    :param arr_max: Max pixel value observed in the image. Used as a default if dnmax is None.
    :param dtype:  A string representing the data type reported by Vicar. Used for finding intmax
    :returns: A tuple of ints containing the (dnmin, dnmax) values after validation.
    """
    # min
    if raw_dnmin is None:
        dnmin: int = arr_min
    elif dtype.kind in ("i", "u"):
        rdnmin = max(raw_dnmin, 0)
        # Subtract 1 to avoid dividing by 0
        dnmin = min(rdnmin, INTMAX.get(dtype.itemsize, MAXDEFAULT - 1) - 1)
    else:
        # floating point images can have negative dnmin
        dnmin = rdnmin

    # max
    if raw_dnmax is None:
        dnmax = arr_max
    elif dtype.kind in ("i", "u"):
        # has to be +1 or it will cause a divide by 0
        rdnmax = max(raw_dnmax, 1)
        dnmax = min(INTMAX.get(dtype.itemsize, MAXDEFAULT), rdnmax)
    else:
        dnmax = rdnmax

    print(f"dnmin = {dnmin}, dnmax = {dnmax}")
    if dnmin > dnmax:
        raise ValueError("dn min is greater than dn max")

    return (dnmin, dnmax)


def quantize_vimg(vimg: npt.NDArray, dnmin: int, dnmax: int) -> npt.NDArray:
    """
    Private function for quantizing a raw raster to 8-bit color

    :param vimg:   A numpy NDArray containing the raster in BIP format.
    :param dnmin:  Max. DN value to clip the upper bound of data in the input image.
    :param dnmax:  Min. DN value to clip the upper bound of data in the input image.
    :returns:       A numpy.ndarray object containing 0-255 8-bit data that can be used
                     to create a png.
    """
    # Convert to the range 0-1 (Normalize the data)
    arr_nml = (vimg - dnmin) / (dnmax - dnmin)
    # Convert to the range 0-255 used in pngs and type uint8
    arr_bytes = (arr_nml * 255).astype(np.uint8)
    print(f"Image dimensions: {arr_bytes.shape}")
    return arr_bytes


def get_mode(nbands: int) -> str:
    """
    Private function for determining PIL Image mode based on number of bands in
    the image.
    """
    if nbands == 1:
        print("Image type: black-and-white image")
        return "L"
    elif nbands == 3:
        print("Image type: color image")
        return "RGB"


def get_outpath(out: Path, source: Path, fmt: str) -> Path:
    """Determine the output filepath."""
    if out is not None:
        if out.is_dir():
            # Create an output file with the same name as the input but put it in
            # the specified output directory
            outpath = PurePath(out).joinpath(PurePath(source).stem + fmt)
        elif PurePath(out).suffix != fmt:
            outpath = out.with_suffix(fmt)
        else:
            outpath = out
    else:
        outpath = source.with_suffix(fmt)
    return outpath


def vic2png(
    source: Path,
    out: Path | None = None,
    fmt: str = ".png",
    dnmin: int | None = None,
    dnmax: int | None = None,
) -> Path:
    """
    Entry point function for converting a Vicar format image to png.
    The source image is opened and read using the pyvicar module, its
    data is normalized and transposed into a format expected by the png
    format, and then it is written out to disk using the PIL module.

    If no value is provided for the "out" keyword, the location of the output
    file will default to the same directory as the input (source) file.

    :param source: A Path pointing to the location of the .VIC or .IMG file to be converted.
    :param out:    An optional Path pointing to either a directory to output the png or the
                     full name of the png to be output. If a file extension other than png is
                     chosen, that will be over-ridden with the .png extension.
    :param fmt:    Output format for conversion. The default as per the function name is vic
                     but jpg and tif are also possible.
    :param dnmin:  Max. DN value to clip the upper bound of data in the input image. Used in
                     format_vimg
    :param dnmax:  Min. DN value to clip the upper bound of data in the input image. Used in
                     format_vimg
    :return:       A string containing the path to the output image. The png file is written
                     as a side effect of this function. Output is printed to stdout
                     unconditionally.
    """
    if not fmt.startswith("."):
        fmt = "." + fmt
    if out is not None and not out.is_dir() and out.suffix != fmt:
        fmt = out.suffix
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format: {fmt}")
    print(f"Converting {source} to {fmt.lstrip('.')}...")

    _, vimg = reader.read_vic(source)
    dnmin, dnmax = validate_dn_range(dnmin, dnmax, vimg.min(), vimg.max(), vimg.dtype)
    png_data = quantize_vimg(vimg, dnmin, dnmax)

    # Determine PIL mode based on number of bands
    mode = get_mode(vimg.shape[2])
    if mode == "L":
        png_data = np.squeeze(png_data)
    img = Image.fromarray(png_data, mode)

    outpath = get_outpath(out, source, fmt)

    img.save(str(outpath))
    print(f"Wrote {str(outpath)} to disk.")
    return outpath


def cli() -> None:
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
    )


if __name__ == "__main__":
    cli()
