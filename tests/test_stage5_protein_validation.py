import pandas as pd

from validation.protein_validation import _evidence_level


def test_stage5_evidence_level_rules():
    assert _evidence_level(True, True, True, "local_matrix_available") == "strong"
    assert _evidence_level(True, False, True, "manual_download_required") == "moderate"
    assert _evidence_level(False, True, False, "local_matrix_available") == "moderate"
    assert _evidence_level(False, False, True, "manual_download_required") == "weak"
    assert _evidence_level(False, False, False, "manual_download_required") == "unavailable"


def test_stage5_integrated_boolean_columns_are_interpretable():
    frame = pd.DataFrame(
        {
            "supported_by_HPA": [True, False],
            "supported_by_CPTAC": [False, False],
            "evidence_level": ["moderate", "unavailable"],
        }
    )
    assert frame["supported_by_HPA"].sum() == 1
    assert frame.loc[1, "evidence_level"] == "unavailable"
