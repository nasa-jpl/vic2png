from dataclasses import dataclass
import numpy as np
from pathlib import Path
import pvl
import numpy.typing as npt
from typing import Tuple


ODL_DTYPE_MAPPING = {
    "IEEE_REAL": "float",
    "MSB_INTEGER": ">i",
    "LSB_INTEGER": "<i",
    "UNSIGNED_INTEGER": ">u",
}


VICAR_DTYPE_MAPPING = {
    "BYTE": "u1",
    "HALF": "i2",
    "FULL": "i4",
    "REAL": "f4",
    "DOUB": "f8",
    "COMP": "c8",
    "WORD": "i2",
    "LONG": "i4",
}


class UnsupportedFileTypeError(Exception):
    pass


@dataclass
class ImageParms:
    lblsize: int
    dtype: npt.DTypeLike
    shape: Tuple[int, int, int]
    org: str


def get_ODL_imageparms(label: pvl.PVLModule) -> ImageParms:
    """Determine the parameters needed to read a raster from a PDS3 image."""

    """lblsize"""
    # If the ODL label is the only label in the file, this should be accurate
    lblsize = label["RECORD_BYTES"] * label["LABEL_RECORDS"]
    if label.get("IMAGE_HEADER") is not None:
        # If there is also a Vicar label, we need to tack that on to the total
        try:
            vicar_bytes = label.get("IMAGE_HEADER").get("BYTES")
            lblsize += int(vicar_bytes)
        except (AttributeError, TypeError):
            print(
                "Warning: unable to read Vicar label information despite IMAGE_HEADER existing. Raster may be inaccurate."
            )

    image_label = label.get("IMAGE")

    """dtype"""
    samp_type = image_label["SAMPLE_TYPE"]
    samp_bits = int(image_label["SAMPLE_BITS"])
    dtype_prefix = ODL_DTYPE_MAPPING.get(samp_type)
    try:
        dtype = np.dtype(f"{dtype_prefix}{samp_bits // 8}")
    except TypeError:
        raise UnsupportedFileTypeError(
            f"file has unknown data type: SAMPLE_TYPE = {samp_type}, SAMPLE_BITS = {samp_bits}"
        )

    """shape"""
    nlines = image_label["LINES"]
    nsamps = image_label["LINE_SAMPLES"]
    nbands = image_label["BANDS"]
    org = image_label.get("BAND_STORAGE_TYPE", "BAND_SEQUENTIAL")
    if org == "BAND_SEQUENTIAL":
        shape = (nbands, nlines, nsamps)
        org = "BSQ"
    elif org == "BAND_INTERLEAVED_PIXEL":
        shape = (nlines, nsamps, nbands)
        org = "BIP"
    elif org == "BAND_INTERLEAVED_LINE":
        shape = (nlines, nbands, nsamps)
        org = "BIL"
    else:
        raise UnsupportedFileTypeError(f"file has unknown band organization: {org}")

    return ImageParms(lblsize, dtype, shape, org)


def get_Vicar_imageparms(label: pvl.PVLModule) -> ImageParms:
    """Determine the parameters needed to read a raster from a Vicar image."""

    """lblsize"""
    lblsize = int(label["LBLSIZE"])

    """dtype"""
    dtype_name = VICAR_DTYPE_MAPPING[label["FORMAT"]]
    kind = np.dtype(dtype_name).kind
    if kind in ("i", "u"):
        intfmt = label.get("INTFMT", "LOW")
        itemsize = np.dtype(dtype_name).itemsize
        if intfmt == "LOW" and itemsize > 1:
            dtype = np.dtype(f"<{dtype_name}")
        elif itemsize > 1:
            dtype = np.dtype(f">{dtype_name}")
        else:
            dtype = np.dtype(dtype_name)
    else:
        realfmt = label.get("REALFMT", "IEEE")
        if realfmt == "IEEE":
            dtype = np.dtype(f">{dtype_name}")
        elif realfmt == "RIEEE":
            dtype = np.dtype(f"<{dtype_name}")
        else:
            raise UnsupportedFileTypeError(
                f"VAX floating point is not supported: REALFMT = {realfmt}"
            )

    """shape"""
    nlines = label["N1"]
    nsamps = label["N2"]
    nbands = label["N3"]
    org = label["ORG"]
    if org == "BSQ":
        shape = (nbands, nsamps, nlines)
    elif org == "BIP":
        shape = (nsamps, nlines, nbands)
    elif org == "BIL":
        shape = (nlines, nbands, nsamps)
    else:
        raise UnsupportedFileTypeError(f"file has unknown band organization: {org}")
    org = org

    # Check some additional label items to make sure the file is supported
    # if int(label.get("EOL", 0)) == 1:
    #    raise UnsupportedFileTypeError(
    #        "Vicar file contains an EOL label, this is not currently supported."
    #    )
    if int(label.get("NLB", 0)) != 0 or int(label.get("NBB", 0)) != 0:
        raise UnsupportedFileTypeError(
            "Vicar file contains a binary header, this is not currently supported."
        )
    if label.get("TYPE") != "IMAGE":
        raise UnsupportedFileTypeError(
            "Vicar file is not an image, this is not currently supported."
        )

    return ImageParms(lblsize, dtype, shape, org)


def read_vic(filepath: Path) -> Tuple[pvl.PVLModule, np.ndarray]:
    # Read the label using PVL, this will parse the ODL label, if available,
    # otherwise it will parse the Vicar label
    # Read 40 bytes to detect if this is a Vicar label only
    is_vicar = False
    with open(filepath, "rb") as f:
        header = f.read(40)
        if header[0:8] == b"LBLSIZE=":
            iblank = header.index(b" ", 8)
            lblsize = int(header[8:iblank])
            f.seek(0)
            vicar_header = f.read(lblsize).rstrip(b"\0")
            is_vicar = True

    if is_vicar:
        # Using this method ensures that PVL will not run into trouble tokenizing
        # the Vicar label
        label = pvl.loads(vicar_header)
        parms = get_Vicar_imageparms(label)
    else:
        with open(filepath, "r") as f:
            label = pvl.load(f)
        if label.get("ODL_VERSION_ID") is not None:
            parms = get_ODL_imageparms(label)
        else:
            raise UnsupportedFileTypeError(
                "unsupported file type encountered, only VIC or IMG are allowed."
            )

    with open(filepath, "rb") as f:
        f.seek(parms.lblsize)
        # Handles EOL label edge case
        raster_bytes = f.read(
            parms.dtype.itemsize * parms.shape[0] * parms.shape[1] * parms.shape[2]
        )
        pixel_data = np.frombuffer(raster_bytes, dtype=parms.dtype)

    # shape pixel stream into 3d array according to the label
    raw_data = pixel_data.reshape(parms.shape)
    # Transpose the data to (line, sample, band) organization [modern standard]
    # if it isn't already
    if parms.org == "BSQ":
        image_data = np.transpose(raw_data, (1, 2, 0))
    elif parms.org == "BIL":
        image_data = np.transpose(raw_data, (0, 2, 1))
    else:
        image_data = raw_data

    return (label, image_data)
