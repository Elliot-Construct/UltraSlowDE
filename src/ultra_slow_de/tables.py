"""Generate summary tables for the analysis.

Produces Markdown-formatted tables written to ``output/tables/``:
  1. Parameter priors table: lists every free parameter with prior ranges.
  2. Observables × datasets matrix: which observables map to which dataset.
    3. Optional convergence diagnostics from multi-chain pilot sampling.
"""

import argparse

from pathlib import Path

import numpy as np

from .builtin_data import load_all_bao
from .ingest import load_pantheon_plus
from .sampler import run_mcmc_multichain

_PRIOR_TABLE = """\
# Parameter Priors

| Model | Parameter | Symbol | Prior | Range | Unit |
|---|---|---|---|---|---|
| Shared | Hubble constant | $H_0$ | Flat | [60, 80] | km s⁻¹ Mpc⁻¹ |
| Shared | Matter density | $\\Omega_m$ | Flat | [0.1, 0.5] | — |
| Model A | Baseline EoS | $w_0$ | Flat | [−1.5, −0.5] | — |
| Model A | Oscillation amplitude | $A$ | Flat | [−0.3, 0.3] | — |
| Model A | Oscillation frequency | $\\omega$ | Flat | [0.1, 10] | — |
| Model A | Oscillation phase | $\\phi$ | Flat | [$-\\pi$, $\\pi$] | rad |
| Model B | Scalar-field mass | $\\mu = m/H_0$ | Log-flat | [10⁻³, 10] | — |
| ΛCDM | (none beyond shared) | — | — | — | — |

## Notes
- All priors are flat (uniform) unless otherwise noted.
- $\\Omega_{\\rm DE}$ is derived from flat-closure: $\\Omega_{\\rm DE} = 1 - \\Omega_m$.
- Radiation density $\\Omega_r$ fixed at 0 for late-universe ($z < 3$) analysis; set to $\\sim 9 \\times 10^{-5}$ when extending to CMB redshifts.
"""

_OBS_MATRIX = """\
# Observables × Datasets Matrix

| Observable | Pantheon+ SN Ia | DESI BAO DR2 | eBOSS DR16 | Planck 2018 |
|---|:---:|:---:|:---:|:---:|
| $H(z)$ | — | $D_H/r_d$ | $D_H/r_d$ | $R$, $l_A$ (compressed priors) |
| $D_L(z)$ / $\\mu(z)$ | ✔ | — | — | — |
| $D_V(z) / r_d$ | — | ✔ | ✔ | — |
| $D_M(z) / r_d$ | — | ✔ | ✔ | — |
| $f\\sigma_8(z)$ | — | — | ✔ | — |
| $q(z)$ | derived | — | — | — |

## Notes
- $D_L$: luminosity distance (restframe); $\\mu$: distance modulus ($= 5\\log_{10}(D_L/10\\,{\\rm pc})$).
- BAO distances ($D_V$, $D_M$, $D_H$) are in units of $r_d$; the sound-horizon scale $r_d$ is either fixed to Planck best-fit or sampled jointly.
- $f\\sigma_8$ from eBOSS provides growth-rate information at the perturbation level. For the background-only pilot run, it constrains Model B via the Friedmann equation only. Full perturbation-level analysis is flagged for Phase 2.
"""


def write_tables(out_dir: Path | None = None) -> list[Path]:
    out = out_dir or Path(__file__).resolve().parents[2] / "output" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, content in [("parameter_priors.md", _PRIOR_TABLE),
                          ("observables_datasets.md", _OBS_MATRIX)]:
        p = out / name
        p.write_text(content, encoding="utf-8")
        saved.append(p)
    return saved


def _sampler_setup(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if model == "lcdm":
        return (
            np.array([67.4, 0.315]),
            np.array([[60.0, 80.0], [0.1, 0.5]]),
            np.array([0.5, 0.01]),
        )
    if model == "a":
        return (
            np.array([67.4, 0.315, -1.0, 0.02, 2.0, 0.0]),
            np.array([
                [60.0, 80.0],
                [0.1, 0.5],
                [-1.5, -0.5],
                [-0.2, 0.2],
                [0.1, 5.0],
                [-np.pi, np.pi],
            ]),
            np.array([0.6, 0.012, 0.03, 0.01, 0.08, 0.08]),
        )
    if model == "b":
        return (
            np.array([67.4, 0.315, 0.3]),
            np.array([[60.0, 80.0], [0.1, 0.5], [0.01, 5.0]]),
            np.array([0.6, 0.012, 0.06]),
        )
    raise ValueError(f"Unsupported model: {model}")


def write_convergence_table(
    out_dir: Path | None = None,
    n_steps: int = 120,
    n_chains: int = 4,
    seed: int = 42,
    models: tuple[str, ...] = ("lcdm", "a"),
) -> Path:
    """Run short multi-chain pilots and write convergence diagnostics table."""
    out = out_dir or Path(__file__).resolve().parents[2] / "output" / "tables"
    out.mkdir(parents=True, exist_ok=True)

    sn = load_pantheon_plus()
    bao = load_all_bao()
    datasets = [(sn, "mb")] + bao

    rows: list[str] = []
    detail_lines: list[str] = []
    model_names = {"lcdm": "$\\Lambda$CDM", "a": "Model A", "b": "Model B"}

    for i, model in enumerate(models):
        theta0, bounds, proposal = _sampler_setup(model)
        res = run_mcmc_multichain(
            datasets=datasets,
            model=model,
            theta0=theta0,
            bounds=bounds,
            proposal_sigma=proposal,
            n_steps=n_steps,
            n_chains=n_chains,
            seed=seed + 100 * i,
            include_planck=True,
        )
        mean_acc = float(np.mean(res.accept_rates))
        max_rhat = float(np.nanmax(res.rhat_per_param)) if res.rhat_per_param is not None else float("nan")
        min_ess = float(np.min(res.ess_per_param)) if res.ess_per_param is not None else float("nan")
        rows.append(
            f"| {model_names[model]} | {n_chains}×{n_steps} | {mean_acc:.3f} | {max_rhat:.3f} | {min_ess:.1f} |"
        )

        if res.rhat_per_param is not None and res.ess_per_param is not None:
            detail = ", ".join(
                f"{p}: R-hat={r:.3f}, ESS={e:.1f}"
                for p, r, e in zip(res.param_names, res.rhat_per_param, res.ess_per_param)
            )
            detail_lines.append(f"- **{model_names[model]}** — {detail}")

    md = "\n".join([
        "# Convergence Diagnostics",
        "",
        "Short multi-chain pilot diagnostics using SN + BAO + Planck compressed priors.",
        "",
        "| Model | Chains × Steps | Mean acceptance | Max split R-hat | Min ESS |",
        "|---|---:|---:|---:|---:|",
        *rows,
        "",
        "## Per-parameter diagnostics",
        *detail_lines,
        "",
        "## Notes",
        "- Split R-hat closer to 1 indicates better between-chain/within-chain consistency.",
        "- ESS is reported after burn-in aggregation across chains (pilot diagnostic).",
        "- These are intentionally short pilot runs; production constraints should use longer chains.",
    ])

    path = out / "convergence_diagnostics.md"
    path.write_text(md, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analysis tables")
    parser.add_argument("--with-convergence", action="store_true", help="also generate convergence diagnostics table")
    parser.add_argument("--n-steps", type=int, default=120)
    parser.add_argument("--n-chains", type=int, default=4)
    parser.add_argument("--include-model-b", action="store_true", help="include costly Model B in convergence table")
    args = parser.parse_args()

    for p in write_tables():
        print(f"Wrote: {p}")
    if args.with_convergence:
        models = ("lcdm", "a", "b") if args.include_model_b else ("lcdm", "a")
        p = write_convergence_table(n_steps=args.n_steps, n_chains=args.n_chains, models=models)
        print(f"Wrote: {p}")
