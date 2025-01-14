import numpy as np
import os
from pathlib import Path
from PIL import Image
import pytest
from img2png.vic2png import vic2png

"""
These tests are not intended to determine whether the png looks as expected,
rather they are just meant to verify that the utility runs without error and
outputs a file to the disk.

Output files are deleted after the test run.
"""


def compare_images(img1_path: Path, img2_path: Path, tolerance: float = 0.01) -> bool:
    """
    Compare two images using mean squared error. img here refers to .png, .jpg, or .tif,
    not the literal .IMG extension.
    Returns True if images are similar within tolerance.
    """
    img1 = np.array(Image.open(img1_path))
    img2 = np.array(Image.open(img2_path))

    if img1.shape != img2.shape:
        return False

    mse = np.mean((img1 - img2) ** 2)
    max_possible_mse = 255**2  # Maximum possible pixel value difference
    normalized_mse = mse / max_possible_mse

    return normalized_mse <= tolerance


def test_vic2png(vic_file, reference_images):
    out_png = vic2png(vic_file)
    assert out_png.exists()
    assert out_png.suffix == ".png"
    assert compare_images(out_png, reference_images("test_vic"))
    os.remove(out_png)


def test_vic2png_bw(vic_file_bw, reference_images):
    out_png = vic2png(vic_file_bw)
    assert out_png.exists()
    assert out_png.suffix == ".png"
    assert compare_images(out_png, reference_images("test_bw"))
    os.remove(out_png)


def test_vic2png_dnrange(vic_file, reference_images):
    out_png = vic2png(vic_file, dnmin=512, dnmax=2048)
    assert out_png.exists()
    assert out_png.suffix == ".png"
    assert compare_images(out_png, reference_images("test_dnrange"))
    os.remove(out_png)


def test_vic2jpg(vic_file, reference_images):
    out_jpg = vic2png(vic_file, fmt=".jpg")
    assert out_jpg.exists()
    assert out_jpg.suffix == ".jpg"
    assert compare_images(out_jpg, reference_images("test_vic", ".jpg"))
    os.remove(out_jpg)


def test_vic2tif(vic_file, reference_images):
    out_tif = vic2png(vic_file, fmt=".tif")
    assert out_tif.exists()
    assert out_tif.suffix == ".tif"
    assert compare_images(out_tif, reference_images("test_vic", ".tif"))
    os.remove(out_tif)


def test_img2png(img_file, reference_images):
    out_png = vic2png(img_file)
    assert out_png.exists()
    assert out_png.suffix == ".png"
    assert compare_images(out_png, reference_images("test_img"))
    os.remove(out_png)


def test_invalid_input_file():
    """Test handling of non-existent input file"""
    with pytest.raises(FileNotFoundError):
        vic2png(Path("nonexistent.vic"))


def test_invalid_format(vic_file):
    """Test handling of invalid output format"""
    with pytest.raises(ValueError, match="unsupported format"):
        vic2png(vic_file, fmt=".invalid")


def test_invalid_dnrange(vic_file):
    """Test handling of invalid DN range"""
    with pytest.raises(ValueError, match="dn min is greater than dn max"):
        vic2png(vic_file, dnmin=100, dnmax=50)
