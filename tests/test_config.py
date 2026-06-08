from pathlib import Path

from scprost_luad.config import load_config_set


def test_load_config_set_has_required_sections():
    root = Path(__file__).resolve().parents[1]
    config = load_config_set(root)
    assert config["project"]["name"] == "SC-PROST-LUAD"
    assert "data" in config
    assert "model" in config
    assert "paths" in config
