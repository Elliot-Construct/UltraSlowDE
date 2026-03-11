import pytest

from ultra_slow_de.data_sources import acquire_dataset, built_in_sources


def test_built_in_sources_have_required_fields():
    sources = built_in_sources()
    assert len(sources) >= 4
    for src in sources.values():
        assert src.name
        assert src.kind
        assert src.source_url
        assert src.version
        assert src.license
        assert src.provenance


def test_acquire_dataset_offline_error_contains_metadata(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        acquire_dataset("pantheon_plus", tmp_path)
    msg = str(exc.value)
    assert "url=" in msg
    assert "version=" in msg
    assert "license=" in msg
    assert "provenance=" in msg