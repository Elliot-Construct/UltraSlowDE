"""Nested sampling for Bayesian evidence (ln Z) using dynesty.

Runs dynesty's dynamic nested sampler for ΛCDM, Model A, and Model B
on the same datasets as the production MCMC run, producing:
  - ln Z (log Bayesian evidence) for each model
  - Δln Z relative to ΛCDM (equivalent to Bayes factors)
  - JSON results appended to output/nested_results.json
  - Console summary table

Usage:
    python -m src.ultra_slow_de.sampler_nested
    python -m src.ultra_slow_de.sampler_nested --model lcdm
    python -m src.ultra_slow_de.sampler_nested --nlive 400
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .builtin_data import load_all_bao
from .evidence_contract import build_likelihood_metadata, validate_nested_vs_production
from .inference import joint_logposterior
from .ingest import load_pantheon_plus
from .likelihood import SNLikelihoodCached
from .model_b import ModelBParams
from .params import CosmoParams, ModelAParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Prior bounds — identical to production MCMC
# ---------------------------------------------------------------------------
_BOUNDS: dict[str, np.ndarray] = {
    "lcdm": np.array([[60.0, 80.0],   # H0
                      [0.1,  0.5]]),   # Omega_m
    "a":    np.array([[60.0, 80.0],   # H0
                      [0.1,  0.5],   # Omega_m
                      [-1.5, -0.5],  # w0
                      [-0.3,  0.3],  # B
                      [-0.3,  0.3],  # C
                      [0.1,   5.0]]),# omega
    "b":    np.array([[60.0, 80.0],  # H0
                      [0.1,  0.5],  # Omega_m
                      [0.01,  5.0]]),# mu
}

_PARAM_NAMES: dict[str, list[str]] = {
    "lcdm": ["h0", "omega_m"],
    "a":    ["h0", "omega_m", "w0", "B", "C", "omega"],
    "b":    ["h0", "omega_m", "mu"],
}


def _prior_transform(u: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Map unit-hypercube sample u ∈ [0,1]^n to physical parameters."""
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return lo + u * (hi - lo)


def _theta_to_params(
    model: str, theta: np.ndarray
) -> tuple[CosmoParams, ModelAParams | None, ModelBParams | None]:
    if model == "lcdm":
        return CosmoParams(h0=float(theta[0]), omega_m=float(theta[1])), None, None
    if model == "a":
        cosmo = CosmoParams(h0=float(theta[0]), omega_m=float(theta[1]))
        ma = ModelAParams(
            w0=float(theta[2]),
            B=float(theta[3]),
            C=float(theta[4]),
            omega=float(theta[5]),
        )
        return cosmo, ma, None
    if model == "b":
        cosmo = CosmoParams(h0=float(theta[0]), omega_m=float(theta[1]))
        mb = ModelBParams(mu=float(theta[2]))
        return cosmo, None, mb
    raise ValueError(f"Unknown model: {model}")


def run_nested(
    model: str,
    datasets: list,
    rd: float = 147.09,
    nlive: int = 300,
    seed: int = 42,
    dlogz: float = 0.5,
) -> dict[str, Any]:
    """Run dynesty dynamic nested sampling for one model.

    Parameters
    ----------
    model : 'lcdm', 'a', or 'b'
    datasets : list of (GaussianDataset, observable_str) pairs
    rd : sound horizon in Mpc
    nlive : number of live points (300 is adequate for evidence; 500+ for posteriors)
    seed : random seed
    dlogz : stopping criterion on remaining ln Z uncertainty

    Returns
    -------
    dict with keys: model, lnZ, lnZ_err, param_names, samples, weights, runtime_seconds
    """
    import dynesty  # type: ignore[import]

    bounds = _BOUNDS[model]
    ndim = bounds.shape[0]

    # Pre-factorise SN covariance once (O(N³)) so each loglike call is O(N²).
    # Without this, each of the ~50k nested sampling calls would do a fresh
    # 1624×1624 Cholesky solve, making the run ≈100× slower.
    sn_cache: SNLikelihoodCached | None = None
    for ds, obs in datasets:
        if obs.lower() == "mb":
            print(f"  [nested:{model}] Pre-factorising SN covariance ({len(ds.y_obs)}×{len(ds.y_obs)})…")
            sn_cache = SNLikelihoodCached(ds, use_gpu=False)
            break

    def loglike(theta: np.ndarray) -> float:
        # dynesty calls loglike with the PHYSICAL parameters (output of
        # prior_transform), NOT the unit-hypercube vector.
        cosmo, ma, mb = _theta_to_params(model, theta)
        try:
            ll = joint_logposterior(
                datasets=datasets,
                cosmo=cosmo,
                model_a=ma,
                model_b=mb,
                rd=rd,
                include_planck=True,
                sn_cache=sn_cache,
            )
            return float(ll) if np.isfinite(ll) else -1e300
        except Exception:
            return -1e300

    def prior_transform(u: np.ndarray) -> np.ndarray:
        # Uniform prior: map unit hypercube → physical bounds.
        return _prior_transform(u, bounds)

    # Use static NestedSampler: terminates cleanly at dlogz < threshold.
    # DynamicNestedSampler adds posterior-refinement batches that waste time
    # when only ln Z is needed.
    sampler = dynesty.NestedSampler(
        loglike,
        prior_transform,
        ndim=ndim,
        nlive=nlive,
        rstate=np.random.default_rng(seed),
        sample="rwalk",        # random-walk proposal — robust for correlated posteriors
        bound="multi",         # multi-ellipsoid bounding
    )

    print(f"  [nested:{model}] Starting static run (nlive={nlive}, dlogz={dlogz})…")
    t0 = time.perf_counter()
    sampler.run_nested(dlogz=dlogz, print_progress=True)
    elapsed = time.perf_counter() - t0

    res = sampler.results
    lnZ = float(res.logz[-1])
    lnZ_err = float(res.logzerr[-1])

    print(f"  [nested:{model}] ln Z = {lnZ:.3f} ± {lnZ_err:.3f}  ({elapsed/60:.1f} min)")

    return {
        "model": model,
        "lnZ": lnZ,
        "lnZ_err": lnZ_err,
        "param_names": _PARAM_NAMES[model],
        "nlive": nlive,
        "dlogz": dlogz,
        "n_samples": int(res.niter),
        "runtime_seconds": float(elapsed),
    }


def run_all_nested(
    models: list[str] | None = None,
    nlive: int = 300,
    seed: int = 42,
    dlogz: float = 0.5,
    strict_compatibility: bool = True,
) -> dict[str, Any]:
    if models is None:
        models = ["lcdm", "a", "b"]

    sn = load_pantheon_plus()
    bao = load_all_bao()
    datasets = [(sn, "mb")] + bao

    results: dict[str, Any] = {}
    for m in models:
        print(f"\n[nested] === Model: {m.upper()} ===")
        results[m] = run_nested(m, datasets, nlive=nlive, seed=seed, dlogz=dlogz)

    # Compute Δln Z vs ΛCDM
    if "lcdm" in results:
        lnz_ref = results["lcdm"]["lnZ"]
        lnz_ref_err = results["lcdm"]["lnZ_err"]
        for m in results:
            dlnz = results[m]["lnZ"] - lnz_ref
            # Error propagates in quadrature
            dlnz_err = float(np.sqrt(results[m]["lnZ_err"] ** 2 + lnz_ref_err ** 2))
            results[m]["delta_lnZ_vs_lcdm"] = dlnz
            results[m]["delta_lnZ_err"] = dlnz_err

    likelihood_metadata = build_likelihood_metadata(
        datasets=datasets,
        rd_mpc=147.09,
        include_planck=True,
        prior_bounds_by_model={k: _BOUNDS[k].tolist() for k in _BOUNDS},
        param_names_by_model={k: list(v) for k, v in _PARAM_NAMES.items()},
        growth_likelihood_mode=None,
        growth_backend_requested=None,
    )

    payload: dict[str, Any] = {
        "schema_version": "evidence-contract-v1",
        "nested": results,
        "nlive": nlive,
        "dlogz": dlogz,
        "seed": seed,
        "likelihood_metadata": likelihood_metadata,
    }

    prod_path = OUTPUT_ROOT / "production_results.json"
    if prod_path.exists():
        try:
            prod_payload = json.loads(prod_path.read_text(encoding="utf-8"))
            comp = validate_nested_vs_production(payload, prod_payload, strict=strict_compatibility)
            payload["compatibility_with_production"] = comp
            if comp.get("status") != "ok":
                print(f"[nested][WARNING] compatibility_with_production={comp['status']}")
        except Exception as exc:
            if strict_compatibility:
                raise
            payload["compatibility_with_production"] = {
                "status": "error",
                "metadata_mismatches": [f"compatibility check exception: {type(exc).__name__}: {exc}"],
                "bound_violations": [],
            }
            print(f"[nested][WARNING] compatibility check skipped: {exc}")

    # Save JSON FIRST — before any Unicode-heavy printing that could crash on Windows
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_ROOT / "nested_results.json"
    out_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote: {out_path}")

    # Print summary table (ASCII-only labels to avoid Windows CP1252 crashes)
    print("\n" + "=" * 70)
    print("Nested sampling summary (Bayesian evidence)")
    print("=" * 70)
    print(f"{'Model':<12} {'ln Z':>10} {'err':>6}  {'dln Z vs LCDM':>15} {'err':>6}  Jeffreys")
    print("-" * 70)
    for m in results:
        r = results[m]
        dlnz = r.get("delta_lnZ_vs_lcdm", 0.0)
        dlnz_err = r.get("delta_lnZ_err", 0.0)
        jeffreys = _jeffreys_label(dlnz)
        print(
            f"{m:<12} {r['lnZ']:>10.3f} {r['lnZ_err']:>6.3f}  "
            f"{dlnz:>+15.3f} {dlnz_err:>6.3f}  {jeffreys}"
        )
    print("=" * 70)
    print("Note: dln Z < 0 => model disfavoured vs LCDM.")
    print("Jeffreys scale: |dln Z| < 1 -> inconclusive; 1-2.5 -> weak;")
    print("                2.5-5 -> moderate; > 5 -> strong.")

    return results


def _jeffreys_label(delta_lnZ: float) -> str:
    """Jeffreys scale interpretation for |Δln Z|."""
    a = abs(delta_lnZ)
    if a < 1.0:
        return "inconclusive"
    if a < 2.5:
        return "weak"
    if a < 5.0:
        return "moderate"
    return "strong"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Nested sampling evidence for ultra-slow DE models.")
    p.add_argument("--model", choices=["lcdm", "a", "b"], default=None,
                   help="Run a single model (default: all three).")
    p.add_argument("--nlive", type=int, default=300,
                   help="Number of live points (default: 300).")
    p.add_argument("--dlogz", type=float, default=0.5,
                   help="Stopping criterion on remaining ln Z (default: 0.5).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-strict-compat",
        action="store_true",
        help="Do not fail run when nested/production compatibility checks fail.",
    )
    args = p.parse_args()

    models = [args.model] if args.model else None
    run_all_nested(
        models=models,
        nlive=args.nlive,
        dlogz=args.dlogz,
        seed=args.seed,
        strict_compatibility=not args.no_strict_compat,
    )
