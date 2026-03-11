from __future__ import annotations

import hashlib
import json
from typing import Any


def _canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _dataset_signature(datasets: list[tuple[Any, str]]) -> list[dict[str, Any]]:
    sig: list[dict[str, Any]] = []
    for ds, obs in datasets:
        sig.append(
            {
                "name": str(getattr(ds, "name", "unknown")),
                "observable": str(obs).lower(),
                "n_points": int(len(getattr(ds, "y_obs"))),
            }
        )
    return sig


def build_likelihood_metadata(
    *,
    datasets: list[tuple[Any, str]],
    rd_mpc: float,
    include_planck: bool,
    prior_bounds_by_model: dict[str, list[list[float]]],
    param_names_by_model: dict[str, list[str]],
    growth_likelihood_mode: str | None = None,
    growth_backend_requested: str | None = None,
) -> dict[str, Any]:
    payload = {
        "rd_mpc": float(rd_mpc),
        "include_planck": bool(include_planck),
        "datasets": _dataset_signature(datasets),
        "prior_bounds_by_model": prior_bounds_by_model,
        "param_names_by_model": param_names_by_model,
        "growth_likelihood_mode": growth_likelihood_mode,
        "growth_backend_requested": growth_backend_requested,
    }
    payload["metadata_hash"] = hashlib.sha256(_canon_json(payload).encode("utf-8")).hexdigest()
    return payload


def compare_likelihood_metadata(lhs: dict[str, Any], rhs: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    base_keys = (
        "rd_mpc",
        "include_planck",
        "datasets",
        "prior_bounds_by_model",
        "param_names_by_model",
    )
    for k in base_keys:
        if lhs.get(k) != rhs.get(k):
            diffs.append(f"metadata mismatch: {k}")

    # Growth metadata is optional unless fsig8 datasets are present in both.
    lhs_has_fsig8 = any(d.get("observable") == "fsig8" for d in lhs.get("datasets", []))
    rhs_has_fsig8 = any(d.get("observable") == "fsig8" for d in rhs.get("datasets", []))
    if lhs_has_fsig8 and rhs_has_fsig8:
        for k in ("growth_likelihood_mode", "growth_backend_requested"):
            lv = lhs.get(k)
            rv = rhs.get(k)
            if lv is not None and rv is not None and lv != rv:
                diffs.append(f"metadata mismatch: {k}")
    return diffs


def check_lnz_upper_bound(
    lnz: float,
    lnz_err: float,
    max_loglike: float,
    *,
    n_sigma: float = 5.0,
    min_slack: float = 0.5,
) -> tuple[bool, str]:
    slack = max(float(min_slack), float(n_sigma) * float(max(0.0, lnz_err)))
    bound = float(max_loglike) + slack
    ok = float(lnz) <= bound
    reason = (
        f"lnZ={lnz:.6g}, max_loglike={max_loglike:.6g}, slack={slack:.6g}, "
        f"bound={bound:.6g}"
    )
    return ok, reason


def validate_nested_vs_production(
    nested_payload: dict[str, Any],
    production_payload: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "ok",
        "metadata_mismatches": [],
        "bound_violations": [],
    }

    nmeta = nested_payload.get("likelihood_metadata")
    pmeta = production_payload.get("likelihood_metadata")
    if nmeta is None or pmeta is None:
        report["metadata_mismatches"].append("missing likelihood_metadata in one or both payloads")
    else:
        report["metadata_mismatches"] = compare_likelihood_metadata(nmeta, pmeta)

    nested_models = nested_payload.get("nested", {})
    production_models = production_payload.get("production", {})
    shared_models = sorted(set(nested_models.keys()) & set(production_models.keys()))

    for model in shared_models:
        lnz = float(nested_models[model].get("lnZ", float("nan")))
        lnz_err = float(nested_models[model].get("lnZ_err", 0.0))
        max_ll = float(production_models[model].get("max_loglike_all_samples", float("nan")))
        if not (lnz == lnz and max_ll == max_ll):
            continue
        ok, reason = check_lnz_upper_bound(lnz, lnz_err, max_ll)
        if not ok:
            report["bound_violations"].append({"model": model, "reason": reason})

    if report["metadata_mismatches"] or report["bound_violations"]:
        report["status"] = "error"

    if strict and report["status"] == "error":
        raise ValueError(
            "Nested/production compatibility check failed: "
            f"metadata={len(report['metadata_mismatches'])}, "
            f"bounds={len(report['bound_violations'])}"
        )

    return report
