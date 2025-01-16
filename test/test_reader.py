# tests/test_reader.py
import numpy as np
import os
import pvl
import pytest
from vic2png.reader import (
    read_vic,
    BandOrg,
    UnsupportedFileTypeError,
)


def test_read_vic_valid_file(vic_file):
    """Test reading a valid VICAR file."""
    label, data = read_vic(vic_file)
    assert isinstance(label, pvl.PVLModule)
    assert isinstance(data, np.ndarray)
    assert data.shape == (60, 80, 3)


def test_read_vic_invalid_file(tmp_path):
    """Test reading an invalid file raises appropriate error."""
    invalid_file = tmp_path / "invalid.vic"
    invalid_file.write_text("Not a VICAR file")

    with pytest.raises(UnsupportedFileTypeError):
        read_vic(invalid_file)
    os.remove(invalid_file)


def test_organization_enum():
    """Test Organization enum functionality."""
    assert BandOrg.BSQ == "BSQ"
    assert BandOrg.from_pds3("BAND_SEQUENTIAL") == BandOrg.BSQ

    with pytest.raises(UnsupportedFileTypeError):
        BandOrg.from_pds3("INVALID")
