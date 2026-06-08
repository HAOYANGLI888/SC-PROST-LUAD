"""Conservative malignant epithelial annotation helpers."""

from __future__ import annotations

import pandas as pd


def malignant_annotation_from_metadata(obs: pd.DataFrame) -> pd.Series:
    """Use only explicit source annotations to label malignant cells."""

    subtype = obs.get("Cell_subtype", pd.Series("", index=obs.index)).astype(str)
    explicit = subtype.str.contains("malignant", case=False, na=False)
    result = pd.Series("not_explicitly_malignant", index=obs.index, dtype=object)
    result.loc[explicit] = "malignant_author_annotation"
    return result


def epithelial_confidence_label(obs: pd.DataFrame) -> pd.Series:
    """Return malignant only when the public dataset explicitly says so."""

    refined = obs.get(
        "Cell_type.refined", obs.get("Cell_type", pd.Series("", index=obs.index))
    ).astype(str)
    malignant = malignant_annotation_from_metadata(obs).eq(
        "malignant_author_annotation"
    )
    labels = pd.Series("non_epithelial", index=obs.index, dtype=object)
    epithelial = refined.str.contains("epithelial", case=False, na=False)
    labels.loc[epithelial] = "epithelial"
    labels.loc[malignant] = "malignant_epithelial_author_annotated"
    return labels

