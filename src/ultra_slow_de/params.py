from dataclasses import dataclass


@dataclass(frozen=True)
class CosmoParams:
    h0: float = 70.0
    omega_m: float = 0.3
    omega_r: float = 0.0
    omega_k: float = 0.0
    omega_de: float | None = None

    def resolved_omega_de(self) -> float:
        if self.omega_de is not None:
            return self.omega_de
        return 1.0 - self.omega_m - self.omega_r - self.omega_k


@dataclass(frozen=True)
class ModelAParams:
    w0: float = -1.0
    B: float = 0.0
    C: float = 0.0
    omega: float = 1.0