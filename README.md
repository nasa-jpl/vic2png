# vic2png
Utility for converting .VIC/.IMG images to compressed image formats.

## Installation

### From PyPI
```bash
pip install vic2png
```

### From Source
```bash
git clone https://github.jpl.nasa.gov/MIPL/vic2png
cd vic2png/
python3 -m venv venv # Optional
source venv/bin/activate # Optional
pip install .
```

## Usage

### Command Line Interface

```bash
usage: vic2png [-h] [-o OUT] [-f FORMAT] [-dnmax DNMAX] [-dnmin DNMIN] [--silent] FILE

positional arguments:
  FILE                  Vicar or PDS .VIC/.IMG format file to be converted

options:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     Output directory or whole filename
  -f FORMAT, --format FORMAT
                        Output format, default is .png but can provide jpg or tif
  -dnmax DNMAX          Max. DN value to clip the upper bound of data in the input image.
  -dnmin DNMIN          Min. DN value to clip the lower bound of data in the input image.
  --silent              If used, no output will be printed during execution.
```

### Example CLI Usage

```bash
# Basic conversion to PNG
vic2png image.vic

# Convert to JPEG with custom output path
vic2png image.vic -o output/converted.jpg

# Convert with DN value clipping (and tif format output)
vic2png image.vic -dnmin 0 -dnmax 255 -f .tif
```

## Python API

This package can be used directly in Python scripts:

```python
from vic2png import vic2png

# Basic conversion
out_png = vic2png("image.vic")

# Advanced usage with all options
out_path = vic2png(
    source=Path("image.vic"),
    out=Path("output/converted.jpg"),
    fmt=".jpg",
    dnmin=0,
    dnmax=255,
    verbose=True
)
```

## Author

[Jacqueline Ryan](mailto:Jacqueline.Ryan@jpl.caltech.edu), Jet Propulsion Laboratory
