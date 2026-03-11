"""Generate diagnostic figures for the ultra-slow dark-energy analysis.

Produces:
  1. H(z) comparison: ΛCDM vs Model A vs Model B
  2. ΔH(z) residual plot
  3. ΔD_L(z) residual plot
  4. q(z) deceleration parameter
  5. w(z) equation-of-state evolution (Model A + Model B)

All figures are saved to ``output/figures/`` relative to project root.
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from .baseline_lcdm import H_lcdm
from .model_a import H_model_a, w_model_a
from .model_b import H_model_b, ModelBParams, solve_model_b
from .observables import deceleration_parameter, luminosity_distance_flat
from .params import CosmoParams, ModelAParams
from .residuals import delta_dl, delta_h


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_all(
    z_max: float = 2.5,
    n_z: int = 400,
    cosmo: CosmoParams | None = None,
    model_a: ModelAParams | None = None,
    model_b_params: ModelBParams | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    if not HAS_MPL:
        raise RuntimeError("matplotlib is required for figure generation.  pip install matplotlib")

    cosmo = cosmo or CosmoParams(h0=70.0, omega_m=0.3)
    model_a = model_a or ModelAParams(w0=-1.0, B=0.05, C=0.0, omega=2.0)
    model_b_params = model_b_params or ModelBParams(mu=0.5)
    out = _ensure_dir(out_dir or Path(__file__).resolve().parents[2] / "output" / "figures")

    z = np.linspace(1e-4, z_max, n_z)

    # Compute observables
    h_lcdm = H_lcdm(z, cosmo)
    h_a = H_model_a(z, cosmo, model_a)
    h_b = H_model_b(z, cosmo.h0, cosmo.omega_m, cosmo.omega_r, model_b_params)

    dl_lcdm = luminosity_distance_flat(z, h_lcdm)
    dl_a = luminosity_distance_flat(z, h_a)
    dl_b = luminosity_distance_flat(z, h_b)

    q_lcdm = deceleration_parameter(z, h_lcdm)
    q_a = deceleration_parameter(z, h_a)
    q_b = deceleration_parameter(z, h_b)

    saved: list[Path] = []

    # --- Fig 1: H(z) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(z, h_lcdm, "k-", label=r"$\Lambda$CDM")
    ax.plot(z, h_a, "C0--", label="Model A")
    ax.plot(z, h_b, "C1-.", label="Model B")
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.legend()
    ax.set_title("Hubble parameter comparison")
    fig.tight_layout()
    p = out / "hz_comparison.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # --- Fig 2: ΔH(z) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(z, delta_h(h_a, h_lcdm), "C0--", label="Model A")
    ax.plot(z, delta_h(h_b, h_lcdm), "C1-.", label="Model B")
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\Delta_H(z)$")
    ax.legend()
    ax.set_title("Fractional Hubble residual vs $\\Lambda$CDM")
    fig.tight_layout()
    p = out / "delta_hz.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # --- Fig 3: ΔD_L(z) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(z, delta_dl(dl_a, dl_lcdm, z), "C0--", label="Model A")
    ax.plot(z, delta_dl(dl_b, dl_lcdm, z), "C1-.", label="Model B")
    ax.set_xlabel("$z$")
    ax.set_ylabel(r"$\Delta_{D_L}(z)$")
    ax.legend()
    ax.set_title("Fractional luminosity-distance residual")
    fig.tight_layout()
    p = out / "delta_dl.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # --- Fig 4: q(z) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.plot(z, q_lcdm, "k-", label=r"$\Lambda$CDM")
    ax.plot(z, q_a, "C0--", label="Model A")
    ax.plot(z, q_b, "C1-.", label="Model B")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$q(z)$")
    ax.legend()
    ax.set_title("Deceleration parameter")
    fig.tight_layout()
    p = out / "qz.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    # --- Fig 5: w(z) ---
    a = 1.0 / (1.0 + z)
    w_a = w_model_a(a, model_a)
    sol_b = solve_model_b(z, omega_m=cosmo.omega_m, omega_r=cosmo.omega_r,
                          model=model_b_params)
    w_b = sol_b["w_phi"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(-1, color="k", lw=0.5, ls=":")
    ax.plot(z, w_a, "C0--", label="Model A $w_{\\rm eff}(z)$")
    ax.plot(z, w_b, "C1-.", label="Model B $w_\\phi(z)$")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$w(z)$")
    ax.legend()
    ax.set_title("Dark-energy equation of state")
    fig.tight_layout()
    p = out / "wz.pdf"
    fig.savefig(p)
    plt.close(fig)
    saved.append(p)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis figures")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--z-max", type=float, default=2.5)
    args = parser.parse_args()
    out = Path(args.out_dir) if args.out_dir else None
    saved = generate_all(z_max=args.z_max, out_dir=out)
    for p in saved:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
