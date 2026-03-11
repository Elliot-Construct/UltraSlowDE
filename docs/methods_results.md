# Ultra-Slow Dark Energy: Methods and Results Summary

## 1. Motivation

The standard $\Lambda$CDM model treats dark energy as a cosmological
constant with equation of state $w = -1$.  Recent BAO measurements from
DESI Year-1 data hint at time-varying dark energy, motivating
exploration of models where $w(z)$ departs slowly from $-1$ —
the "ultra-slow" regime.

This project tests whether a slowly evolving dark-energy component
is consistent with combined distance and expansion-rate data, and
whether it improves the fit relative to $\Lambda$CDM.

---

## 2. Models

### 2.1 Baseline: $\Lambda$CDM

$$E^2(z) = \Omega_m (1+z)^3 + \Omega_r (1+z)^4 + \Omega_\Lambda$$

with flat-closure $\Omega_\Lambda = 1 - \Omega_m - \Omega_r$.

### 2.2 Model A: Oscillatory Effective EoS

$$w(a) = w_0 + A\sin(\omega\ln a + \phi)$$

The dark-energy density ratio is

$$X_{\rm DE}(a) = \exp\!\Bigl[-3\int_1^a \frac{1+w(a')}{a'}\,da'\Bigr]$$

evaluated in closed form using a numerically stable sinc rewrite that
avoids divergence as $\omega \to 0$.

### 2.3 Model B: Canonical Quintessence

A canonical scalar field $\phi$ with potential $V(\phi)$ evolves via the
Klein–Gordon equation on an FRW background.  We integrate in
dimensionless variables ($N = \ln a$, $\tilde{y}_1 = \phi/M_{\rm Pl}$,
$\tilde{y}_2 = d\phi/dN / M_{\rm Pl}$) using an implicit Radau solver,
with the algebraic Friedmann constraint replacing the second-order ODE
for $H$.

Supported potentials:
- **Quadratic:** $V = \tfrac12 m^2 \phi^2$  (mass parameter $\mu = m/H_0$)
- **Cosine:** $V = \Lambda^4[1 - \cos(\phi/f)]$  (axion-like)

---

## 3. Observables

| Observable | Definition | Use |
|---|---|---|
| $H(z)$ | Hubble parameter | Direct comparison to BAO $D_H/r_d$ |
| $D_L(z)$ | Luminosity distance | SN Ia distance moduli |
| $q(z)$ | Deceleration parameter $= (1+z)H'/H - 1$ | Phase-space diagnostic |
| $\Delta_H(z)$ | $(H_{\rm model} - H_{\Lambda{\rm CDM}}) / H_{\Lambda{\rm CDM}}$ | Residual sensitivity |
| $\Delta_{D_L}(z)$ | $(D_{L,{\rm model}} - D_{L,\Lambda{\rm CDM}}) / D_{L,\Lambda{\rm CDM}}$ | Residual sensitivity |
| $w(z)$ | Dark-energy EoS | Model comparison |

---

## 4. Data

| Dataset | Type | Redshift range | Reference |
|---|---|---|---|
| Pantheon+ | SN Ia $\mu(z)$ | $0.001 < z < 2.26$ | Scolnic et al. 2022 |
| DESI BAO DR1 | $D_V/r_d$, $D_M/r_d$, $D_H/r_d$ | $0.3 < z < 2.33$ | DESI 2024 |
| eBOSS DR16 | BAO + $f\sigma_8$ | $0.15 < z < 2.33$ | Alam et al. 2021 |
| Planck 2018 | Compressed CMB priors ($R$, $l_A$) | $z_* \approx 1090$ | Planck 2020 |

Full provenance, DOIs, and license constraints are recorded in
`data-sources.md` and the `data_sources.py` registry.

---

## 5. Statistical Method

### 5.1 Likelihood

For each dataset providing $N$ data points $\mathbf{y}_{\rm obs}$ with
covariance $\mathbf{C}$:

$$\ln\mathcal{L} = -\frac{1}{2}\left[\mathbf{r}^T \mathbf{C}^{-1} \mathbf{r} + \ln|\mathbf{C}| + N\ln(2\pi)\right]$$

where $\mathbf{r} = \mathbf{y}_{\rm obs} - \mathbf{y}_{\rm model}(\theta)$.

Joint log-likelihood across independent datasets is additive.

### 5.2 Priors

Flat (uniform) priors on all parameters.  See `output/tables/parameter_priors.md`
for ranges.

### 5.3 Sampling

Pilot runs use a Metropolis–Hastings random-walk sampler implemented
in `sampler.py`.  Production runs should use `emcee` or `cobaya`.

---

## 6. Validation

| Check | Status |
|---|---|
| $\Lambda$CDM recovery (Model A: $A=0, w_0=-1$; Model B: $\mu \to 0$) | ✅ |
| $E(z=0) = 1$ (closure) | ✅ |
| Continuity of $X_{\rm DE}$ as $\omega \to 0$ | ✅ |
| ODE integration convergence (Model B, Radau) | ✅ |
| Kinetic-domination guard ($\dot\phi^2 < 6$) | ✅ |
| $w_\phi$ near $-1$ in thawing limit | ✅ |
| Gaussian log-likelihood against analytic diagonal case | ✅ |

---

## 7. Assumptions and Caveats

1. **Spatial flatness** ($\Omega_k = 0$) assumed throughout.
2. **Minimal neutrino mass** ($\sum m_\nu = 0.06$ eV) and standard $N_{\rm eff} = 3.046$.
3. **Background-level analysis only.** Perturbation-level constraints
   ($P(k)$, $f\sigma_8$) are deferred to Phase 2 (see `perturbation_plan.md`).
4. **Model A is phenomenological** — the effective EoS may not
   correspond to a self-consistent Lagrangian.
5. Sound horizon $r_d$ is either fixed to Planck best-fit or sampled
   jointly with BAO data.

---

## 8. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Ultra-slow departures from $w=-1$ too small to detect | High | Quantify forecast sensitivity; report upper bounds |
| Degeneracy between $H_0$ and $\Omega_m$ masks DE signal | Moderate | Use SN+BAO+CMB combination to break degeneracy |
| Oscillatory EoS (Model A) aliased by sparse BAO bins | Moderate | Test multiple $\omega$ values; compare to Model B which is smooth |
| Numerical instability at high $z$ for Model B ODE | Low | Radau solver with tight tolerances; validated against known limits |

---

## 9. Outputs

- **Figures:** `output/figures/` — $H(z)$, $\Delta_H$, $\Delta_{D_L}$, $q(z)$, $w(z)$
- **Tables:** `output/tables/` — parameter priors, observables × datasets
- **Code:** `src/ultra_slow_de/` — fully tested Python package
- **Tests:** `tests/` — 21+ unit tests covering all modules
