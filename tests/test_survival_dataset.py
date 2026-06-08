from data.survival_dataset import Stage2Paths, make_toy_dataset, prepare_stage2_dataset


def test_toy_survival_dataset_and_stratified_splits(tmp_path):
    result = prepare_stage2_dataset(tmp_path, small_test=True)
    manifest = result["manifest"]
    assert manifest["dataset_mode"] == "toy_small_test"
    assert manifest["os_time_unit"] == "days"
    assert manifest["os_event_encoding"] == "death=1,censored=0"
    assert manifest["prepared_patient_count"] == 96
    assert set(make_toy_dataset()["os_event"].unique()) == {0, 1}
    paths = Stage2Paths.from_root(tmp_path)
    for seed in (42, 3407, 2026):
        assert paths.split_file(seed).is_file()
