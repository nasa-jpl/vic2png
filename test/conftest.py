from pathlib import Path, PurePath
import pytest


def get_test_resource(resource_name):
    test_directory = PurePath(__file__).parent
    test_file = test_directory.joinpath("test_resources", resource_name)
    return Path(test_file).absolute()


@pytest.fixture
def vic_file():
    return get_test_resource(
        "NLF_0074_0673513257_993EDR_T0032430NCAM00190_01_600J01.VIC"
    )


@pytest.fixture
def vic_file_bw():
    return get_test_resource(
        "NLG_0900_0746838848_005FDR_N0440898NCAM00500_0A02I4J01.IMG"
    )


@pytest.fixture
def img_file():
    return get_test_resource(
        "NLF_0074_0673513257_993EDR_T0032430NCAM00190_01_600J03.IMG"
    )


@pytest.fixture
def reference_images():
    """Fixture to provide paths to reference images"""

    def get_reference(base_name: str, fmt: str = ".png") -> Path:
        return get_test_resource(f"reference/{base_name}{fmt}")

    return get_reference


@pytest.fixture
def tmp_path():
    return Path.cwd()
