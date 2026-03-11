import pytest

from ultra_slow_de.evidence_contract import (
    build_likelihood_metadata,
    check_lnz_upper_bound,
    compare_likelihood_metadata,
    validate_nested_vs_production,
)


class _DummyDS:
    def __init__(self, name: str, n: int):
        self.name = name
        self.y_obs = [0.0] * n


def _meta(obs: str = "mb", rd: float = 147.09):
    ds = [(_DummyDS("pantheon_plus", 1624), obs)]
    return build_likelihood_metadata(
        datasets=ds,
        rd_mpc=rd,
        include_planck=True,
        prior_bounds_by_model={"lcdm": [[60.0, 80.0], [0.1, 0.5]]},
        param_names_by_model={"lcdm": ["h0", "omega_m"]},
        growth_likelihood_mode=None,
        growth_backend_requested=None,
    )


def test_compare_likelihood_metadata_detects_mismatch():
    lhs = _meta(obs="mb")
    rhs = _meta(obs="dm_rd")
    diffs = compare_likelihood_metadata(lhs, rhs)
    assert diffs


def test_check_lnz_upper_bound_passes_when_consistent():
    ok, _ = check_lnz_upper_bound(lnz=10.0, lnz_err=0.2, max_loglike=10.5)
    assert ok


def test_check_lnz_upper_bound_fails_when_violated():
    ok, reason = check_lnz_upper_bound(lnz=12.0, lnz_err=0.1, max_loglike=10.0)
    assert not ok
    assert "bound" in reason


def test_validate_nested_vs_production_errors_on_bound_violation():
    nested = {
        "likelihood_metadata": _meta(),
        "nested": {"lcdm": {"lnZ": 12.0, "lnZ_err": 0.1}},
    }
    prod = {
        "likelihood_metadata": _meta(),
        "production": {"lcdm": {"max_loglike_all_samples": 10.0}},
    }
    with pytest.raises(ValueError):
        validate_nested_vs_production(nested, prod, strict=True)


def test_validate_nested_vs_production_reports_metadata_mismatch_nonstrict():
    nested = {
        "likelihood_metadata": _meta(obs="mb"),
        "nested": {"lcdm": {"lnZ": 8.0, "lnZ_err": 0.1}},
    }
    prod = {
        "likelihood_metadata": _meta(obs="dm_rd"),
        "production": {"lcdm": {"max_loglike_all_samples": 9.0}},
    }
    report = validate_nested_vs_production(nested, prod, strict=False)
    assert report["status"] == "error"
    assert report["metadata_mismatches"]


def test_growth_metadata_optional_without_fsig8():
    nested = {
        "likelihood_metadata": _meta(obs="mb"),
        "nested": {"lcdm": {"lnZ": 8.0, "lnZ_err": 0.1}},
    }
    pm = _meta(obs="mb")
    pm["growth_likelihood_mode"] = "production"
    pm["growth_backend_requested"] = "auto"
    prod = {
        "likelihood_metadata": pm,
        "production": {"lcdm": {"max_loglike_all_samples": 9.0}},
    }
    report = validate_nested_vs_production(nested, prod, strict=False)
    assert report["metadata_mismatches"] == []
