# Perturbation-Level Compatibility Plan

## Context

The current ultra-slow dark-energy modelling pipeline operates at the
**background level** only — we compute $H(z)$, $D_L(z)$, $q(z)$, and
$w(z)$, then compare to distance or rate measurements.  A full
confrontation with CMB anisotropy power spectra, matter power spectrum
$P(k)$, or growth-rate data ($f\sigma_8$) requires linearised
perturbation theory on top of the background solution.

This document records the plan, scope boundary, and risk register for
extending to perturbation-level consistency.

---

## Phase 1 (Current): Background-Only

| Capability | Status |
|---|---|
| Friedmann integration (Model A, B, ΛCDM) | ✅ |
| $H(z)$, $D_L(z)$, $q(z)$ outputs | ✅ |
| SN distance-modulus likelihood | ✅ |
| BAO distance-ratio likelihood | ✅ (structure ready) |
| Compressed CMB priors ($R$, $l_A$) | ✅ |
| $f\sigma_8$ as standalone data points | ✅ (structure ready) |
| MCMC pilot sampler | ✅ |

## Phase 2 (Future): Perturbation Extension

### Requirements

1. **Linear growth ODE for Model B.**  The canonical quintessence field
   modifies the growth equation:

   $$\ddot{\delta}_m + 2H\dot{\delta}_m - 4\pi G \rho_m \delta_m = 0$$

   where $H(a)$ is sourced from the Model B integrator.  For Model A
   (effective EoS), the same holds with $H(a)$ from Model A.

2. **Growth rate $f(z) = d\ln\delta / d\ln a$** and normalised
   $f\sigma_8(z) = f(z) \cdot \sigma_8(z)$ need to be computed and
   compared to eBOSS (and future DESI) RSD measurements.

3. **CMB compatibility.**  Options (increasing fidelity):
   - (a) Compressed distance priors — already implemented.
   - (b) Modified CAMB/CLASS with custom dark-energy module.
   - (c) Full Boltzmann solver with quintessence perturbations.

### Implementation Steps

| Step | Description | Dependency |
|---|---|---|
| P2.1 | Implement cosmic growth ODE ($\delta_m(a)$) driven by Model A/B $H(a)$ | Phase 1 complete |
| P2.2 | Derive and validate $f\sigma_8(z)$ predictions | P2.1 |
| P2.3 | Add $f\sigma_8$ likelihood with eBOSS data | P2.2 + dataset ingestion |
| P2.4 | Interface with CLASS/CAMB for full $C_\ell$ comparison (optional) | P2.1 |
| P2.5 | Re-run MCMC with growth + background joint likelihood | P2.2 + P2.3 |

### Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Quintessence clustering at sub-horizon scales modifies $\delta_m$ growth beyond smooth-DE assumption | Moderate | Confirm scalar-field sound speed $c_s = 1$ (canonical kinetic term) keeps quintessence perturbations small at scales of interest |
| Model A effective EoS may not correspond to a consistent perturbation theory | High | Model A is phenomenological — document as EoS-only, use Model B for perturbation consistency |
| $\sigma_8$ normalisation ambiguity across datasets | Low–Moderate | Fix normalisation convention; use $\sigma_8(z=0)$ from Planck and propagate |

---

## Decision Log

- **Current decision:** Phase 1 (background-only) is the scope of the
  initial analysis.  Perturbation extension is deferred until
  background-level results establish whether the hypothesis is worth
  pursuing.
- **Rationale:** Background-level distance data (SN, BAO) already
  constrain $w(z)$ tightly; growth data adds complementary information
  but is computationally more expensive and requires careful treatment
  of Model A's phenomenological nature.
