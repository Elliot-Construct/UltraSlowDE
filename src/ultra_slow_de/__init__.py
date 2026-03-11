"""Ultra-slow dark-energy modelling utilities."""

from .params import CosmoParams, ModelAParams
from .baseline_lcdm import E_lcdm, H_lcdm
from .model_a import E_model_a, H_model_a, w_model_a, xde_model_a
from .observables import luminosity_distance_flat, deceleration_parameter
from .residuals import delta_h, delta_dl
from .data_sources import SourceRecord, built_in_sources, acquire_dataset
from .datasets import GaussianDataset, covariance_from_sigma, validate_dataset
from .likelihood import gaussian_loglike, dataset_loglike, joint_loglike, sn_loglike_marg
from .inference import predict_observable, loglike_for_dataset, joint_logposterior
from .sampler import (
    MCMCResult,
    MultiChainMCMCResult,
    run_mcmc,
    run_mcmc_backend,
    run_mcmc_multichain,
    run_mcmc_multichain_backend,
)
from .sampler_jax import (
    jax_backend_available,
    jax_backend_info,
    run_mcmc_jax,
    run_mcmc_multichain_jax,
)

__all__ = [
    "CosmoParams",
    "ModelAParams",
    "E_lcdm",
    "H_lcdm",
    "E_model_a",
    "H_model_a",
    "w_model_a",
    "xde_model_a",
    "luminosity_distance_flat",
    "deceleration_parameter",
    "delta_h",
    "delta_dl",
    "SourceRecord",
    "built_in_sources",
    "acquire_dataset",
    "GaussianDataset",
    "covariance_from_sigma",
    "validate_dataset",
    "gaussian_loglike",
    "dataset_loglike",
    "joint_loglike",
    "sn_loglike_marg",
    "predict_observable",
    "loglike_for_dataset",
    "joint_logposterior",
    "MCMCResult",
    "MultiChainMCMCResult",
    "run_mcmc",
    "run_mcmc_backend",
    "run_mcmc_multichain",
    "run_mcmc_multichain_backend",
    "jax_backend_available",
    "jax_backend_info",
    "run_mcmc_jax",
    "run_mcmc_multichain_jax",
]