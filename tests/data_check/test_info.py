from genomic_benchmarks.data_check import info, is_downloaded, list_datasets

def test_info():
    assert info("demo_mouse_enhancers", 0) is not None
