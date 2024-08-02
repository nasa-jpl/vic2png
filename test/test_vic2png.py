import os
from img2png.vic2png import vic2png

'''
These tests are not intended to determine whether the png looks as expected,
rather they are just meant to verify that the utility runs without error and
outputs a file to the disk.

Output files are deleted after the test run.
'''

def test_vic2png(vic_file):
    out_png = vic2png(vic_file)
    assert os.path.exists(out_png)
    assert out_png.suffix == ".png"
    os.remove(out_png)

def test_vic2png_dnrange(vic_file):
    out_png = vic2png(vic_file,
                      dnmin=0,
                      dnmax=4095)
    assert os.path.exists(out_png)
    assert out_png.suffix == ".png"
    os.remove(out_png)

def test_vic2jpg(vic_file):
    out_jpg = vic2png(vic_file,
                      fmt=".jpg")
    assert os.path.exists(out_jpg)
    assert out_jpg.suffix == ".jpg"
    os.remove(out_jpg)

def test_vic2tif(vic_file):
    out_tif = vic2png(vic_file,
                      fmt=".tif")
    assert os.path.exists(out_tif)
    assert out_tif.suffix == ".tif"
    os.remove(out_tif)

def test_img2png(img_file):
    out_png = vic2png(img_file)
    assert os.path.exists(out_png)
    assert out_png.suffix == ".png"
    os.remove(out_png)

def test_img2jpg(img_file):
    out_jpg = vic2png(img_file,
                      fmt=".jpg")
    assert os.path.exists(out_jpg)
    assert out_jpg.suffix == ".jpg"
    os.remove(out_jpg)

def test_img2tif(img_file):
    out_tif = vic2png(img_file,
                      fmt=".tif")
    assert os.path.exists(out_tif)
    assert out_tif.suffix == ".tif"
    os.remove(out_tif)