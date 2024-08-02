import json
from pathlib import Path, PurePath
import pytest

def get_test_resource(resource_name):
    test_directory = PurePath(__file__).parent
    test_file = test_directory.joinpath("test_resources", resource_name)
    return Path(test_file).absolute()

@pytest.fixture
def vic_file():
    return get_test_resource("NLF_0074_0673513257_993EDR_T0032430NCAM00190_01_600J01.VIC")

@pytest.fixture
def img_file():
    return get_test_resource("NLF_0074_0673513257_993EDR_T0032430NCAM00190_01_600J03.IMG")