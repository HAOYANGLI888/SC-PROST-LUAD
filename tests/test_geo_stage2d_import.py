import gzip

from data.geo_download import GEO_COHORTS
from data.geo_expression_import import read_geo_series_metadata
from data.geo_platform_annotation import parse_geo_platform_annotation
from data.geo_survival_preprocess import prepare_geo_os_from_series_metadata


def test_geo_platform_annotation_accepts_genesymbol_alias(tmp_path):
    annotation = tmp_path / "GPLTEST.txt"
    annotation.write_text(
        "!platform_table_begin\n"
        "ID\tGeneSymbol\n"
        "probe_1\tTP53\n"
        "probe_2\tEGFR /// OTHER\n"
        "!platform_table_end\n",
        encoding="utf-8",
    )
    parsed = parse_geo_platform_annotation(annotation)
    assert parsed.to_dict("records") == [
        {"probe_id": "probe_1", "gene_symbol": "TP53"},
        {"probe_id": "probe_2", "gene_symbol": "EGFR"},
    ]


def test_geo_series_metadata_and_survival_are_parsed_to_days(tmp_path):
    matrix = tmp_path / "GSE72094_series_matrix.txt.gz"
    content = (
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_platform_id\t"GPL15048"\t"GPL15048"\n'
        '!Sample_characteristics_ch1\t"vital_status: Dead"\t"vital_status: Alive"\n'
        '!Sample_characteristics_ch1\t"survival_time_in_days: 300"\t"survival_time_in_days: 900"\n'
        '!Sample_characteristics_ch1\t"age_at_diagnosis: 65"\t"age_at_diagnosis: 70"\n'
        '!Sample_characteristics_ch1\t"gender: Male"\t"gender: Female"\n'
        '!Sample_characteristics_ch1\t"Stage: Stage II"\t"Stage: Stage I"\n'
        "!series_matrix_table_begin\n"
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"probe_1"\t1\t2\n'
        "!series_matrix_table_end\n"
    )
    with gzip.open(matrix, "wt", encoding="utf-8") as handle:
        handle.write(content)
    metadata, summary = read_geo_series_metadata(matrix)
    survival = prepare_geo_os_from_series_metadata(metadata, GEO_COHORTS["GSE72094"])
    assert summary["sample_count"] == 2
    assert survival["os_time_days"].tolist() == [300.0, 900.0]
    assert survival["os_event"].tolist() == [1, 0]
    assert survival["stage_numeric"].tolist() == [2.0, 1.0]


def test_geo_characteristics_are_aligned_by_each_cells_key(tmp_path):
    matrix = tmp_path / "mixed_series_matrix.txt.gz"
    content = (
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_characteristics_ch1\t"status: Dead"\t"time: 900"\n'
        '!Sample_characteristics_ch1\t"time: 300"\t"status: Alive"\n'
        "!series_matrix_table_begin\n"
        '"ID_REF"\t"GSM1"\t"GSM2"\n'
        '"probe_1"\t1\t2\n'
        "!series_matrix_table_end\n"
    )
    with gzip.open(matrix, "wt", encoding="utf-8") as handle:
        handle.write(content)
    metadata, _ = read_geo_series_metadata(matrix)
    assert metadata["status"].tolist() == ["Dead", "Alive"]
    assert metadata["time"].tolist() == ["300", "900"]


def test_gse50081_survival_time_is_converted_from_years():
    from pandas import DataFrame

    metadata = DataFrame(
        {
            "sample_id": ["GSM1", "GSM2"],
            "survival_time": ["1.0", "2.0"],
            "status": ["dead", "alive"],
            "age": ["60", "70"],
            "sex": ["male", "female"],
            "stage": ["1A", "2B"],
        }
    )
    survival = prepare_geo_os_from_series_metadata(metadata, GEO_COHORTS["GSE50081"])
    assert survival["os_time_days"].tolist() == [365.25, 730.5]
