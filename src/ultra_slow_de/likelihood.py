import numpy as np

from .datasets import GaussianDataset, validate_dataset


def gaussian_loglike(residual: np.ndarray, cov: np.ndarray) -> float:
    r = np.asarray(residual, dtype=float)
    c = np.asarray(cov, dtype=float)

    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(c)):
        raise ValueError("Residual or covariance contains non-finite values")
    if c.shape != (len(r), len(r)):
        raise ValueError("Covariance shape mismatch")
    if not np.allclose(c, c.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Covariance must be symmetric")

    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0:
        raise ValueError("Covariance determinant sign must be positive")

    quad = r @ np.linalg.solve(c, r)
    n = len(r)
    return -0.5 * (quad + logdet + n * np.log(2.0 * np.pi))


def dataset_loglike(ds: GaussianDataset, y_model: np.ndarray) -> float:
    validate_dataset(ds)
    ym = np.asarray(y_model, dtype=float)
    if ym.shape != ds.y_obs.shape:
        raise ValueError("Model vector shape mismatch for dataset")
    residual = ds.y_obs - ym
    return gaussian_loglike(residual, ds.cov)


def joint_loglike(items: list[tuple[GaussianDataset, np.ndarray]]) -> float:
    return float(sum(dataset_loglike(ds, ym) for ds, ym in items))


def sn_loglike_marg(ds: GaussianDataset, mu_model: np.ndarray) -> float:
    r"""SN Ia log-likelihood with analytical marginalisation over M_B.

    For Pantheon+ the observed vector is m_b_corr = μ(z) + M_B.
    With a flat prior on M_B the marginalised likelihood is:

    .. math::
        -2\ln L_{\rm marg} = \chi^2_{\min} + \ln\det C
                             + (N-1)\ln(2\pi) + \ln E

    where Δ = m_obs − μ_model, E = 1ᵀ C⁻¹ 1,
    χ²_min = Δᵀ C⁻¹ Δ − (1ᵀ C⁻¹ Δ)² / E.
    """
    validate_dataset(ds)
    mu = np.asarray(mu_model, dtype=float)
    delta = ds.y_obs - mu  # = M_B + noise
    c_inv_delta = np.linalg.solve(ds.cov, delta)
    ones = np.ones_like(delta)
    c_inv_ones = np.linalg.solve(ds.cov, ones)

    a = float(delta @ c_inv_delta)
    b = float(ones @ c_inv_delta)
    e = float(ones @ c_inv_ones)

    chi2_min = a - b * b / e
    sign, logdet = np.linalg.slogdet(ds.cov)
    n = len(delta)
    return -0.5 * (chi2_min + logdet + (n - 1) * np.log(2.0 * np.pi) + np.log(e))


class SNLikelihoodCached:
    """Pre-factorised SN Ia marginalised likelihood for repeated evaluation.

    Performs the O(N^3) Cholesky decomposition once at construction,
    then each ``loglike(mu_model)`` call is O(N^2).

    Parameters
    ----------
    ds : GaussianDataset
        The SN Ia dataset (covariance + observations).
    use_gpu : bool, optional
        If True, attempt to accelerate using CuPy (NVIDIA GPU).
        Falls back silently to CPU if CuPy is unavailable or GPU init fails.
        Default is False (CPU).
    """

    def __init__(self, ds: GaussianDataset, use_gpu: bool = False) -> None:
        validate_dataset(ds)
        self.ds = ds
        self.n = len(ds.y_obs)
        self._use_gpu = False

        if use_gpu:
            self._use_gpu = self._init_gpu(ds)

        if not self._use_gpu:
            self._init_cpu(ds)

    # ------------------------------------------------------------------
    # CPU initialisation (original logic)
    # ------------------------------------------------------------------
    def _init_cpu(self, ds: GaussianDataset) -> None:
        self.L = np.linalg.cholesky(ds.cov)
        self.logdet = 2.0 * np.sum(np.log(np.diag(self.L)))
        ones = np.ones(self.n)
        from scipy.linalg import cho_solve
        self._c_inv_ones = cho_solve((self.L, True), ones)
        self._e = float(ones @ self._c_inv_ones)
        self._const = -0.5 * (self.logdet + (self.n - 1) * np.log(2.0 * np.pi)
                               + np.log(self._e))

    # ------------------------------------------------------------------
    # GPU initialisation via CuPy
    # ------------------------------------------------------------------
    @staticmethod
    def _fix_cuda_env() -> None:
        """On Windows, ensure CUDA_HOME points to an existing toolkit directory.

        The system may have a stale CUDA_HOME (e.g., pointing to a removed
        CUDA 12.1 install) while a valid CUDA toolkit lives under CUDA_PATH.
        CuPy's cuda-pathfinder prefers CUDA_HOME with no existence check, so
        we correct this before any CuPy import.  We also register the DLL
        directory explicitly with the Windows loader so cudart, cublasLt, etc.
        are found regardless of PATH ordering.
        """
        import os, sys
        if sys.platform != "win32":
            return
        cuda_home = os.environ.get("CUDA_HOME", "")
        cuda_path = os.environ.get("CUDA_PATH", "")
        from pathlib import Path as _P
        if cuda_home and not _P(cuda_home).exists():
            # CUDA_HOME is set but does not exist; fall back to CUDA_PATH
            if cuda_path and _P(cuda_path).exists():
                os.environ["CUDA_HOME"] = cuda_path
                cuda_home = cuda_path
            else:
                del os.environ["CUDA_HOME"]
                cuda_home = ""
        # Register the bin dir so Windows finds cudart*.dll / cublasLt*.dll
        effective = cuda_home or cuda_path
        if effective:
            bin_dir = str(_P(effective) / "bin")
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(bin_dir)   # Python ≥ 3.8 Windows only
            except (AttributeError, OSError):
                pass

    def _init_gpu(self, ds: GaussianDataset) -> bool:
        """Initialise GPU arrays.  Returns True on success, False otherwise.

        All working buffers are pre-allocated here so that loglike() never
        calls into the CuPy memory allocator at evaluation time.  This
        eliminates memory-pool pressure and prevents Windows WDDM scheduler
        stalls caused by repeated small allocations.
        """
        try:
            self._fix_cuda_env()
            import cupy as cp  # type: ignore[import]

            # Pre-warm CuPy's memory pool with a large block to avoid
            # pool-resize stalls during the MCMC hot loop.
            _pool = cp.get_default_memory_pool()
            _warmup = cp.empty(self.n * self.n, dtype=cp.float64)
            del _warmup
            _pool.free_all_blocks()

            cov_gpu = cp.asarray(ds.cov, dtype=cp.float64)
            # Cholesky for logdet
            L = cp.linalg.cholesky(cov_gpu)
            logdet = 2.0 * float(cp.sum(cp.log(cp.diag(L))))
            # Pre-compute full C^{-1} once (O(N^3)); stored on GPU for O(N^2) per step.
            # For N=1644 this is ~21 MB of VRAM — negligible on a 4090.
            c_inv = cp.linalg.inv(cov_gpu)
            y_obs_gpu = cp.asarray(ds.y_obs, dtype=cp.float64)
            ones = cp.ones(self.n, dtype=cp.float64)
            c_inv_ones = c_inv @ ones
            e = float(ones @ c_inv_ones)
            const = -0.5 * (logdet + (self.n - 1) * float(cp.log(cp.array(2.0 * np.pi)))
                            + float(cp.log(cp.array(e))))

            # --- Pre-allocated working buffers (zero allocations per loglike call) ---
            # _mu_gpu   : receives mu_model in-place via copyto
            # _delta_gpu: y_obs - mu, overwritten each call
            # _work_gpu : [a, b] packed into 2-element vector for single sync
            _mu_gpu    = cp.empty(self.n, dtype=cp.float64)
            _delta_gpu = cp.empty(self.n, dtype=cp.float64)
            _work_gpu  = cp.empty(2, dtype=cp.float64)

            # Pre-JIT all kernels by running a dummy loglike (eliminates first-call
            # NVRTC compilation pause that can look like a GPU dropout mid-run).
            cp.copyto(_mu_gpu, cp.asarray(ds.y_obs, dtype=cp.float64))
            cp.subtract(y_obs_gpu, _mu_gpu, out=_delta_gpu)
            _tmp = c_inv @ _delta_gpu
            _work_gpu[0] = _delta_gpu @ _tmp
            _work_gpu[1] = c_inv_ones @ _delta_gpu
            cp.cuda.Stream.null.synchronize()

            # Assign only if everything succeeded
            self._cp           = cp
            self._c_inv_gpu    = c_inv
            self._y_obs_gpu    = y_obs_gpu
            self._c_inv_ones_gpu = c_inv_ones
            self._e            = e
            self._const        = const
            self._mu_gpu       = _mu_gpu
            self._delta_gpu    = _delta_gpu
            self._work_gpu     = _work_gpu
            return True
        except Exception as exc:  # pragma: no cover
            import warnings
            warnings.warn(
                "SNLikelihoodCached GPU init failed; using CPU fallback. "
                f"Reason: {type(exc).__name__}: {str(exc)[:120]}",
                RuntimeWarning,
                stacklevel=3,
            )
            return False

    # ------------------------------------------------------------------
    # Likelihood evaluation
    # ------------------------------------------------------------------
    def loglike(self, mu_model: np.ndarray) -> float:
        if self._use_gpu:
            # Hot path: zero allocations, single GPU→CPU sync.
            # All buffers (_mu_gpu, _delta_gpu, _work_gpu) are pre-allocated.
            try:
                cp = self._cp
                # In-place copy into pre-allocated GPU buffer (no allocator, no round-trip).
                cp.copyto(self._mu_gpu, cp.asarray(mu_model, dtype=cp.float64))
                cp.subtract(self._y_obs_gpu, self._mu_gpu, out=self._delta_gpu)
                c_inv_d = self._c_inv_gpu @ self._delta_gpu   # GEMV; result lives on GPU
                # Pack both dot products into a 2-element vector → single sync.
                self._work_gpu[0] = self._delta_gpu @ c_inv_d
                self._work_gpu[1] = self._c_inv_ones_gpu @ self._delta_gpu
                ab = cp.asnumpy(self._work_gpu)               # ONE GPU→CPU sync
                chi2_min = ab[0] - ab[1] * ab[1] / self._e
                return self._const - 0.5 * chi2_min
            except Exception:
                # GPU context was lost (e.g. Windows TDR reset).  Switch permanently
                # to CPU for the remainder of this chain.
                import warnings
                warnings.warn(
                    "GPU error during loglike(); switching to CPU for this chain.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._use_gpu = False
                self._init_cpu(self.ds)
                # Fall through to CPU path below.

        from scipy.linalg import cho_solve
        delta = self.ds.y_obs - np.asarray(mu_model, dtype=float)
        c_inv_delta = cho_solve((self.L, True), delta, check_finite=False)
        a = float(delta @ c_inv_delta)
        b = float(self._c_inv_ones @ delta)  # = 1ᵀ C⁻¹ Δ  (symmetric)
        chi2_min = a - b * b / self._e
        return self._const - 0.5 * chi2_min