"""Regenerate corner plots from stored production_results.json.

Usage:
    python -m src.ultra_slow_de.regenerate_corners

Loads the pre-computed chain data from output/production_results.json and
regenerates output/figures/corner_{lcdm,a,b}.pdf with corrected LaTeX labels
and tight bounding boxes.  No MCMC sampling is performed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = PROJECT_ROOT / "output" / "production_results.json"
FIG_ROOT = PROJECT_ROOT / "output" / "figures"

_LATEX_LABELS: dict[str, str] = {
    "h0":      r"$H_0\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    "omega_m": r"$\Omega_m$",
    "w0":      r"$w_0$",
    "B":       r"$B$",
    "C":       r"$C$",
    "omega":   r"$\omega$",
    "mu":      r"$\mu$",
}


def _load_production_results() -> dict:
    print(f"Loading {JSON_PATH} …", flush=True)
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return data["production"]


def regenerate_corners(production: dict, out_dir: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; cannot regenerate corners.", file=sys.stderr)
        return []

    try:
        import corner  # type: ignore
    except ImportError:
        print("corner not available; cannot regenerate corners.", file=sys.stderr)
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for m in ("lcdm", "a", "b"):
        r = production[m]
        chains = np.asarray(r["chains"], dtype=float)
        burn = int(r["burn_in_steps"])
        flat = chains[:, burn:, :].reshape(-1, chains.shape[-1])

        latex_labels = [_LATEX_LABELS.get(p, p) for p in r["param_names"]]

        print(f"  Plotting corner_{m} ({flat.shape[0]} posterior samples, "
              f"params={r['param_names']}) …", flush=True)

        cfig = corner.corner(
            flat,
            labels=latex_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".4g",
            title_kwargs={"fontsize": 8},
            label_kwargs={"fontsize": 9},
        )

        cp = out_dir / f"corner_{m}.pdf"
        cfig.savefig(cp, bbox_inches="tight")
        plt.close(cfig)
        saved.append(cp)
        print(f"  Saved: {cp}", flush=True)

    return saved


def main() -> int:
    if not JSON_PATH.exists():
        print(f"[regenerate_corners] ERROR: {JSON_PATH} not found.", file=sys.stderr)
        return 1

    production = _load_production_results()
    saved = regenerate_corners(production, FIG_ROOT)
    print(f"\nRegenerated {len(saved)} corner plot(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
