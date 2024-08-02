# vic2png
Utility for converting .VIC/.IMG images to compressed image formats.

## Installation

```
git clone https://github.jpl.nasa.gov/jryan/vic2png
cd vic2png/
python3 -m venv venv # Optional
source venv/bin/activate # Optional
pip install .
```

## Usage

```
usage: vic2png [-h] [-o OUT] [-f FORMAT] [-dnmax DNMAX] [-dnmin DNMIN] source

positional arguments:
  source                Vicar or PDS .VIC/.IMG format file to be converted

options:
  -h, --help            show this help message and exit
  -o OUT, --out OUT     Output directory or whole filename
  -f FORMAT, --format FORMAT
                        Output format, default is .png but can provide jpg or tif
  -dnmax DNMAX          Max. DN value to clip the upper bound of data in the input image.
  -dnmin DNMIN          Min. DN value to clip the lower bound of data in the input image.
```