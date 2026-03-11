import argparse
import numpy as np

from .baseline_lcdm import H_lcdm
from .model_a import H_model_a
from .observables import deceleration_parameter, luminosity_distance_flat
from .params import CosmoParams, ModelAParams
from .residuals import delta_dl, delta_h


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultra-slow dark-energy Model A demo")
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--n-z", type=int, default=300)
    parser.add_argument("--h0", type=float, default=70.0)
    parser.add_argument("--omega-m", type=float, default=0.3)
    parser.add_argument("--omega-r", type=float, default=0.0)
    parser.add_argument("--omega-k", type=float, default=0.0)
    parser.add_argument("--w0", type=float, default=-1.0)
    parser.add_argument("--B", type=float, default=0.0)
    parser.add_argument("--C", type=float, default=0.0)
    parser.add_argument("--omega", type=float, default=1.0)
    return parser


def run_demo(args: argparse.Namespace) -> dict[str, np.ndarray]:
    z = np.linspace(0.0, args.z_max, args.n_z)
    cosmo = CosmoParams(
        h0=args.h0,
        omega_m=args.omega_m,
        omega_r=args.omega_r,
        omega_k=args.omega_k,
    )
    model = ModelAParams(w0=args.w0, B=args.B, C=args.C, omega=args.omega)

    h_base = H_lcdm(z, cosmo)
    h_model = H_model_a(z, cosmo, model)
    dl_base = luminosity_distance_flat(z, h_base)
    dl_model = luminosity_distance_flat(z, h_model)
    q_model = deceleration_parameter(z, h_model)

    return {
        "z": z,
        "h_base": h_base,
        "h_model": h_model,
        "dl_base": dl_base,
        "dl_model": dl_model,
        "q_model": q_model,
        "delta_h": delta_h(h_model, h_base),
        "delta_dl": delta_dl(dl_model, dl_base, z=z),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    out = run_demo(args)
    print("Computed arrays:")
    print(f"  n(z)          : {len(out['z'])}")
    print(f"  max |ΔH|      : {np.max(np.abs(out['delta_h'])):.6e}")
    print(f"  max |ΔD_L|    : {np.max(np.abs(out['delta_dl'][1:])):.6e}")
    print(f"  q(z=0)        : {out['q_model'][0]:.6f}")


if __name__ == "__main__":
    main()