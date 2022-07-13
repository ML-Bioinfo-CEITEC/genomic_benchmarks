from pathlib import Path

import pandas as pd
import pytest
from genomic_benchmarks.data_check import info, is_downloaded, list_datasets
from genomic_benchmarks.utils.paths import CACHE_PATH, DATASET_DIR_PATH

# dataset_name, version, train samples, test samples, num of classes
datasets_test_data = [
        ("human_nontata_promoters", 0, 27097, 9034, 2),
        ("demo_coding_vs_intergenomic_seqs", 0, 75000, 25000, 2),
        ("demo_human_or_worm", 0, 75000, 25000, 2),
        ("drosophila_enhancers_stark", 0, 5184, 1730, 2),
        ("dummy_mouse_enhancers_ensembl", 0, 968, 242, 2),
        ("human_enhancers_cohn", 0, 20843, 6948, 2),
        ("human_enhancers_ensembl", 0, 123872, 30970, 2),
        ("human_ocr_ensembl", 0, 139804, 34952, 2),
        ("human_ensembl_regulatory", 0, 231348, 57713, 3),
    ]


@pytest.mark.parametrize("dataset_name, version, train_samples, test_samples, classes", datasets_test_data)
def test_info(dataset_name, version, train_samples, test_samples, classes):
    dataset_info = info(dataset_name, version)

    assert dataset_info is not None
    assert isinstance(dataset_info, pd.DataFrame)
    assert dataset_info.shape == (classes, 2)
    assert dataset_info.train.sum() == train_samples
    assert dataset_info.test.sum() == test_samples


def test_info_fails_for_missing_dataset():
    with pytest.raises(FileNotFoundError):
        info("donotexist")


# @pytest.mark.parametrize("dataset_name, version, train_samples, test_samples, classes", datasets_test_data)
# def test_list_datasets_returns_all_expected_datasets(dataset_name, version, train_samples, test_samples, classes):
#     assert dataset_name in list_datasets()


def test_list_datasets_returns_exactly_expected_datasets():
    expected = [dataset[0] for dataset in datasets_test_data]

    actual = list_datasets()
    lacks = set(expected) - set(actual)
    extra = set(actual) - set(expected)

    message = f"list_datasets() lacks elements {lacks} " if lacks else ''
    message += f"list_datasets() has extra elements {extra}" if extra else ''
    assert not message


# ToDo mock path and directories instead of repeating functions functionality?
def test_list_datasets_returns_exactly_expected_in_local_repo():
    actual = [x.name for x in Path(DATASET_DIR_PATH).iterdir() if x.is_dir()]

    expected = list_datasets(local_repo=True)
    lacks = set(expected) - set(actual)
    extra = set(actual) - set(expected)

    message = f"list_datasets(local_repo=True) lacks elements {lacks} " if lacks else ''
    message += f"list_datasets(local_repo=True) has extra elements {extra}" if extra else ''
    assert not message


def test_is_downloaded_returns_false_not_existing():
    assert not is_downloaded('not_existing')


@pytest.mark.parametrize("dataset_name, version, train_samples, test_samples, classes", datasets_test_data)
def test_is_downloaded_returns_true_for_downloaded_datasets(dataset_name, version, train_samples, test_samples, classes):
    base_path = CACHE_PATH / dataset_name

    if(Path(base_path).exists()):
        assert is_downloaded(dataset_name)


@pytest.mark.parametrize("dataset_name, version, train_samples, test_samples, classes", datasets_test_data)
def test_is_downloaded_returns_false_for_not_downloaded_datasets(dataset_name, version, train_samples, test_samples, classes):
    base_path = CACHE_PATH / dataset_name

    if(not Path(base_path).exists()):
        assert not is_downloaded(dataset_name)


