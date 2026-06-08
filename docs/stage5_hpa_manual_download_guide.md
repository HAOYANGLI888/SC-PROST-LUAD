# Stage 5 HPA Manual Download Guide

This guide is for manual Human Protein Atlas review when automatic JSON parsing
does not expose lung-cancer IHC image-level fields.

## Scope

- Use HPA only as qualitative orthogonal protein/IHC support.
- Do not treat HPA IHC staining as quantitative survival validation.
- Do not modify Stage 4 signatures based on HPA results.
- Do not claim causal confirmation.

## Candidate Table

Run:

```powershell
python scripts/stage5_select_candidate_genes.py --config configs/base.yaml
python scripts/stage5_hpa_validation.py --config configs/base.yaml
```

Then open:

- `outputs/tables/stage5_candidate_genes.csv`
- `outputs/tables/stage5_hpa_ihc_links.csv`

## Manual Review Fields

For each candidate gene, review the HPA gene and pathology links and record:

- gene symbol
- HPA URL
- pathology/lung cancer URL
- antibody ID
- lung cancer or lung adenocarcinoma IHC availability
- staining level
- staining intensity
- staining quantity
- tumor versus normal qualitative pattern
- reliability category
- representative image links if needed

## Optional Manual Output Template

Save optional manual review as:

`data/metadata/stage5_hpa_manual_review.csv`

Recommended columns:

```text
gene_symbol,hpa_url,hpa_pathology_url,antibody_id,lung_cancer_ihc_available,staining_level,staining_intensity,staining_quantity,tumor_vs_normal_pattern,reliability,representative_image_url,reviewer_note
```

Do not download bulk IHC images unless a small representative set is explicitly
needed for a figure panel.

