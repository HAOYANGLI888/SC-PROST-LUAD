import json

from data.survival_dataset import prepare_stage2_dataset
from models.stage2_training import train_seed


def test_stage2_preprocessors_fit_training_patients_only(tmp_path):
    prepare_stage2_dataset(tmp_path, small_test=True)
    train_seed(tmp_path, seed=42, small_test=True)
    manifest = json.loads(
        (tmp_path / "outputs" / "logs" / "stage2_training_manifest_seed42.json").read_text()
    )
    fitted = set(manifest["rna_preprocessor_fit_patient_ids"])
    validation = set(manifest["validation_patient_ids"])
    test = set(manifest["test_patient_ids"])
    assert fitted
    assert fitted.isdisjoint(validation)
    assert fitted.isdisjoint(test)
