from genomic_benchmarks.data_check import info, is_downloaded, list_datasets

def test_info():
    assert info("dummy_mouse_enhancers_ensembl", 0) is not None
