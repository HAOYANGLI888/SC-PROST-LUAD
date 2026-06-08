from pathlib import Path

import pandas as pd
import torch

from data.wsi_manifest import flatten_patient_slides
from models.clinical_pathology_fusion import ClinicalPathologyFusionSurvival
from models.pathology_attention_mil import AttentionMILSurvival
from models.pathology_gated_mil import GatedAttentionMILSurvival
from pathology.patch_extraction import extract_slide_patch_index
from pathology.patch_feature_extraction import extract_slide_features
from pathology.wsi_io import create_synthetic_slide
from training.pathology_train import _load_patient_bags


def test_wsi_patient_slide_map_deterministically_prefers_primary_slide():
    hits = [
        {
            "file_id": "file-b",
            "file_name": "B.svs",
            "cases": [{"case_id": "case", "submitter_id": "TCGA-05-4244", "samples": [{"submitter_id": "sample-b", "sample_type": "Recurrent Tumor"}]}],
        },
        {
            "file_id": "file-a",
            "file_name": "A.svs",
            "cases": [{"case_id": "case", "submitter_id": "TCGA-05-4244", "samples": [{"submitter_id": "sample-a", "sample_type": "Primary Tumor"}]}],
        },
    ]
    frame = flatten_patient_slides(hits)
    assert frame.iloc[0]["file_id"] == "file-a"
    assert bool(frame.iloc[0]["preferred_slide_for_smallset"])
    assert frame["patient_slide_count"].tolist() == [2, 2]


def test_synthetic_patch_and_feature_extraction(tmp_path):
    slide = create_synthetic_slide(tmp_path / "toy.tif", size=768)
    patch_dir = tmp_path / "patches"
    result = extract_slide_patch_index(slide, patch_dir, max_patches=10)
    assert result["selected_patch_count"] > 0
    assert (patch_dir / "patch_index.csv").is_file()
    assert (patch_dir / "qc_tissue_mask.png").is_file()
    feature_path = tmp_path / "features.pt"
    summary = extract_slide_features(slide, patch_dir / "patch_index.csv", feature_path, backend="handcrafted")
    assert summary["feature_dim"] > 0
    artifact = torch.load(feature_path, map_location="cpu", weights_only=False)
    assert artifact["features"].shape[0] == result["selected_patch_count"]


def test_pathology_models_forward_patient_bags():
    bags = [torch.randn(7, 16), torch.randn(9, 16), torch.randn(5, 16)]
    clinical = torch.randn(3, 3)
    attention_scores, attention = AttentionMILSurvival(16)(bags)
    gated_scores, _ = GatedAttentionMILSurvival(16)(bags)
    fusion_scores, _ = ClinicalPathologyFusionSurvival(16)(bags, clinical)
    assert attention_scores.shape == gated_scores.shape == fusion_scores.shape == (3,)
    assert len(attention) == len(bags)
    assert torch.isclose(attention[0].sum(), torch.tensor(1.0))


def test_patient_bag_loader_concatenates_multiple_slides(tmp_path):
    rows = []
    for index, patches in enumerate((3, 4)):
        path = tmp_path / f"slide_{index}.pt"
        torch.save(
            {
                "features": torch.randn(patches, 8),
                "coordinates": torch.zeros(patches, 2, dtype=torch.int64),
            },
            path,
        )
        rows.append(
            {
                "patient_id": "PATIENT-1",
                "file_id": f"SLIDE-{index}",
                "feature_path": str(path),
                "feature_status": "extracted",
            }
        )
    summary = tmp_path / "summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    frame, bags, coordinates = _load_patient_bags(summary)
    assert len(frame) == len(bags) == len(coordinates) == 1
    assert frame.iloc[0]["aggregated_slide_count"] == 2
    assert bags[0].shape == (7, 8)
