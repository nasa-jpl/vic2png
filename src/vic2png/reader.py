"""
APACHE LICENSE, VERSION 2.0
Copyright (c) 2023-24 California Institute of Technology (“Caltech”).
U.S. Government sponsorship acknowledged.

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:
• Redistributions of source code must retain the above copyright notice, this list of
  conditions and the following disclaimer.
• Redistributions in binary form must reproduce the above copyright notice, this list
  of conditions and the following disclaimer in the documentation and/or other materials
  provided with the distribution.
• Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory,
  nor the names of its contributors may be used to endorse or promote products derived 
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import pvl
import numpy.typing as npt
from typing import Dict, Tuple


ODL_DTYPE_MAPPING: Dict[str, str] = {
    "IEEE_REAL": "float",
    "MSB_INTEGER": ">i",
    "LSB_INTEGER": "<i",
    "UNSIGNED_INTEGER": ">u",
}


VICAR_DTYPE_MAPPING: Dict[str, str] = {
    "BYTE": "u1",
    "HALF": "i2",
    "FULL": "i4",
    "REAL": "f4",
    "DOUB": "f8",
    "COMP": "c8",
    "WORD": "i2",
    "LONG": "i4",
}

VICAR_HEADER_SIZE = 40
VICAR_LABEL_PREFIX = b"LBLSIZE="


class UnsupportedFileTypeError(Exception):
    pass


class BandOrg(str, Enum):
    BSQ = "BSQ"
    BIP = "BIP"
    BIL = "BIL"

    @classmethod
    def from_pds3(cls, org: str) -> "BandOrg":
        """Convert PDS3 organization string to Organization enum.

        :param org: PDS3 organization string
        :return: BandOrg enum value
        """
        mapping = {
            "BAND_SEQUENTIAL": cls.BSQ,
            "BAND_INTERLEAVED_PIXEL": cls.BIP,
            "BAND_INTERLEAVED_LINE": cls.BIL,
        }
        if org not in mapping:
            raise UnsupportedFileTypeError(f"File has unknown band organization: {org}")
        return mapping[org]

    def get_shape_order(
        self, nlines: int, nsamps: int, nbands: int
    ) -> Tuple[int, int, int]:
        """Get the shape tuple for this organization."""
        if self == BandOrg.BSQ:
            return (nbands, nlines, nsamps)
        elif self == BandOrg.BIP:
            return (nlines, nsamps, nbands)
        else:  # BIL
            return (nlines, nbands, nsamps)


@dataclass(frozen=True)
class ImageParms:
    """Parameters for image data reading.

    lblsize: Size of the label in bytes
    dtype: numpy dtype for the image data
    shape: (lines, samples, bands) tuple
    org: Band organization of the data (BSQ, BIP, or BIL)
    """

    lblsize: int
    dtype: npt.DTypeLike
    shape: Tuple[int, int, int]
    org: BandOrg


def get_odl_imageparms(label: pvl.PVLModule) -> ImageParms:
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
                "Warning: unable to read Vicar label information despite IMAGE_HEADER"
                + "existing. Raster may be inaccurate."
            )

    image_label = label.get("IMAGE")

    """dtype"""
    samp_type = image_label["SAMPLE_TYPE"]
    samp_bits = int(image_label["SAMPLE_BITS"])
    dtype_prefix = ODL_DTYPE_MAPPING.get(samp_type)
    try:
        dtype = np.dtype(f"{dtype_prefix}{samp_bits // 8}")
    except TypeError as e:
        raise UnsupportedFileTypeError(
            f"file has unknown data type: SAMPLE_TYPE = {samp_type}, "
            + f"SAMPLE_BITS = {samp_bits}"
        ) from e

    """shape"""
    nlines = image_label["LINES"]
    nsamps = image_label["LINE_SAMPLES"]
    nbands = image_label["BANDS"]
    pds3_org = image_label.get("BAND_STORAGE_TYPE", "BAND_SEQUENTIAL")
    try:
        org = BandOrg.from_pds3(pds3_org)
    except ValueError as e:
        raise UnsupportedFileTypeError(
            f"file has unknown band organization: {label['ORG']}"
        ) from e
    shape = org.get_shape_order(nlines, nsamps, nbands)

    return ImageParms(lblsize, dtype, shape, org)


def get_vicar_imageparms(label: pvl.PVLModule) -> ImageParms:
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
    nlines = label["N2"]
    nsamps = label["N1"]
    nbands = label["N3"]
    try:
        org = BandOrg(label["ORG"])
    except ValueError as e:
        raise UnsupportedFileTypeError(
            f"File has unknown band organization: {label['ORG']}"
        ) from e
    shape = org.get_shape_order(nlines, nsamps, nbands)

    # Check some additional label items to make sure the file is supported
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
    """
    Read a Vicar of PDS3 format raster file.

    :params filepath: Path to the image raster.
    :return: A tuple[label, image_data] where label is a PVLModule containing the
        parsed metadata label and image_data is a numpy array of
        shape (lines, samples, bands)
    """
    # Read the label using PVL, this will parse the ODL label, if available,
    # otherwise it will parse the Vicar label
    # Read 40 bytes to detect if this is a Vicar label only
    is_vicar = False
    with open(filepath, "rb") as f:
        header = f.read(VICAR_HEADER_SIZE)
        if header[0:8] == VICAR_LABEL_PREFIX:
            iblank = header.index(b" ", 8)
            lblsize = int(header[8:iblank])
            f.seek(0)
            vicar_header = f.read(lblsize).rstrip(b"\0")
            is_vicar = True

    if is_vicar:
        # Using this method ensures that PVL will not run into trouble tokenizing
        # the Vicar label
        label = pvl.loads(vicar_header)
        parms = get_vicar_imageparms(label)
    else:
        with open(filepath, "r") as f:
            try:
                label = pvl.load(f)
            except:
                raise UnsupportedFileTypeError(
                    "unsupported file type encountered, only VIC or IMG are allowed."
                )
        if label.get("ODL_VERSION_ID") is not None:
            parms = get_odl_imageparms(label)
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
    # Transpose the data to (line, sample, band) organization if it isn't already
    if parms.org == BandOrg.BSQ:
        image_data = np.transpose(raw_data, (1, 2, 0))
    elif parms.org == BandOrg.BIL:
        image_data = np.transpose(raw_data, (0, 2, 1))
    else:
        image_data = raw_data

    return (label, image_data)
