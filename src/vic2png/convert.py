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

import numpy as np
from pathlib import Path, PurePath
from PIL import Image
import numpy.typing as npt
from typing import Dict, FrozenSet, Optional, Tuple, Union

from . import reader


INTMAX: Dict[int, int] = {1: 255, 2: 4095, 4: 65535}
MAXDEFAULT: int = 4095
SUPPORTED_FORMATS: FrozenSet = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})


def validate_dn_range(
    raw_dnmin: Optional[int],
    raw_dnmax: Optional[int],
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
    :param arr_min: Min pixel value observed in the image. Used if dnmin is None.
    :param arr_max: Max pixel value observed in the image. Used if dnmax is None.
    :param dtype:  A string representing the data type reported by Vicar. Used for
        finding intmax
    :returns: A tuple of ints containing the (dnmin, dnmax) values after validation.
    """
    # min
    if raw_dnmin is None:
        dnmin = arr_min
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
    return arr_bytes


def get_mode(nbands: int) -> str:
    """
    Private function for determining PIL Image mode based on number of bands in
    the image.
    """
    if nbands == 1:
        return "L"
    elif nbands == 3:
        return "RGB"
    else:
        raise ValueError("unsupported band count")


def get_outpath(out: Path, source: Path, fmt: str) -> Path:
    """Determine the output filepath."""
    if out is not None:
        if out.is_dir():
            # Create an output file with the same name as the input but put it in
            # the specified output directory
            outpath = out.joinpath(source.stem + fmt)
        elif out.suffix != fmt:
            outpath = out.with_suffix(fmt)
        else:
            outpath = out
    else:
        outpath = source.with_suffix(fmt)
    return outpath


def vic2png(
    source: Union[Path, str],
    out: Optional[Union[Path, str]] = None,
    fmt: str = ".png",
    dnmin: Optional[int] = None,
    dnmax: Optional[int] = None,
    verbose: bool = False,
) -> Path:
    """
    Function for converting a Vicar or PDS3 format raster to various image formats.
    The source image is opened and read using the pvl module, quantizes and transposes
    the data into the expected format, then writes it to disk using PIL.

    :param source:  Path to the .VIC or .IMG file to be converted.
    :param out:     Optional path for output. If None, uses the source directory.
        If a directory, outputs a file with the same name as the source in that
        directory. If a file, uses the format in that file extension.
    :param fmt:     Output format extension (".png", ".jpg"/".jpeg", ".tif"/".tiff")
    :param dnmin:   Optional minimum DN value for clipping.
    :param dnmax:   Optional maximum DN value for clipping.
    :param verbose: If True, print parsed information about the image.
    :return:        Location of the output image file.
    """
    if type(source) != Path:
        source = Path(source)
    if out is not None and type(out) != Path:
        out = Path(out)
    if not fmt.startswith("."):
        fmt = "." + fmt
    # If the outpath specifies a file format, prioritize that over the format keyword
    if out is not None and not out.is_dir() and out.suffix != fmt:
        fmt = out.suffix
    if fmt.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format: {fmt}")
    if verbose:
        print(f"Converting {source} to {fmt.lstrip('.')}...")

    _, vimg = reader.read_vic(source)
    dnmin, dnmax = validate_dn_range(dnmin, dnmax, vimg.min(), vimg.max(), vimg.dtype)
    if verbose:
        print(f"dnmin = {dnmin}, dnmax = {dnmax}")
    png_data = quantize_vimg(vimg, dnmin, dnmax)
    if verbose:
        print(f"Image dimensions: {png_data.shape}")

    # Determine PIL mode based on number of bands
    mode = get_mode(vimg.shape[2])
    if mode == "L":
        if verbose:
            print("Image type: black-and-white image")
        png_data = np.squeeze(png_data)
    elif verbose:
        print("Image type: color image")
    img = Image.fromarray(png_data, mode)

    outpath = get_outpath(out, source, fmt)

    img.save(str(outpath))
    if verbose:
        print(f"Wrote {str(outpath)} to disk.")
    return outpath
