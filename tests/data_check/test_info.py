import pandas as pd
import pytest
from genomic_benchmarks.data_check import info


def test_info():
    mouse_info = info("dummy_mouse_enhancers_ensembl", 0)

    assert mouse_info is not None
    assert isinstance(mouse_info, pd.DataFrame)
    assert mouse_info.shape == (2, 2)
    assert mouse_info.train.sum() == 968


def test_info_fails_for_missing_dataset():
    with pytest.raises(FileNotFoundError):
        info("donotexist")
