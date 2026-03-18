# Ultra-Slow Mimics of Dark Energy

## Executive summary
We propose two FRW-consistent mechanisms—ultra–low-frequency oscillatory dark-energy dynamics (“longitudinal” in the scalar/background sense) and inhomogeneous-curvature backreaction—that can reproduce ΛCDM-like late-time acceleration while predicting constrained, structured residuals in \(H(z)\), \(D_L(z)\), and growth. The empirical anchor is the SN Ia distance–redshift acceleration inference (Riess et al. 1998; Perlmutter et al. 1999).

## FRW framework and observables
FRW metric:
\[
ds^2=-c^2dt^2+a(t)^2\left[\frac{dr^2}{1-kr^2}+r^2d\Omega^2\right],\quad H\equiv \dot a/a.
\]
Friedmann equations:
\[
H^2=\frac{8\pi G}{3}\rho-\frac{kc^2}{a^2},\qquad \frac{\ddot a}{a}=-\frac{4\pi G}{3}\left(\rho+\frac{3p}{c^2}\right).
\]
Continuity (each component):
\[
\frac{d\rho}{d\ln a}=-3(1+w)\rho,\quad w\equiv p/\rho c^2.
\]
For \(k=0\): \(D_L(z)=(1+z)c\int_0^z \!dz'/H(z')\), and \(q(z)\equiv-\ddot a/(aH^2)\).

## Model A
Parametric oscillatory equation of state:
\[
w(a)=w_0+A\sin(\omega\ln a+\phi).
\]
Integrating the continuity equation with \(\rho_{de}(a=1)=\rho_{de0}\) gives:
\[
\rho_{de}(a)=\rho_{de0}\,a^{-3(1+w_0)}\exp\!\left[\frac{3A}{\omega}\big(\cos(\omega\ln a+\phi)-\cos\phi\big)\right].
\]
Background expansion:
\[
H(z)=H_0\left[\Omega_r(1+z)^4+\Omega_m(1+z)^3+\Omega_k(1+z)^2+\Omega_{de}(z)\right]^{1/2}.
\]
Define residuals relative to ΛCDM (baseline \(w=-1\)):
\[
\Delta_H(z)\equiv\frac{H(z)-H_\Lambda(z)}{H_\Lambda(z)},\qquad 
\Delta_{D_L}(z)\equiv\frac{D_L(z)-D_{L,\Lambda}(z)}{D_{L,\Lambda}(z)}.
\]
(Operationally, Model A is a “search template” for low-frequency wiggles in \(H\) and distances.)

## Model B
Canonical quintessence:
\[
\rho_\phi=\tfrac12\dot\phi^2+V(\phi),\quad p_\phi=\tfrac12\dot\phi^2-V(\phi),\quad w_\phi=p_\phi/\rho_\phi.
\]
Klein–Gordon equation (coupled to Friedmann via \(\rho=\rho_m+\rho_r+\rho_\phi\)):
\[
\ddot\phi+3H\dot\phi+\frac{dV}{d\phi}=0.
\]
Potentials enabling ultra-slow evolution (and potentially ultra-long-period oscillations around extrema):
\[
V(\phi)=\tfrac12 m^2\phi^2,\qquad V(\phi)=\Lambda^4\left[1-\cos(\phi/f)\right],
\]
with slow dynamics typically requiring \(m\lesssim \mathcal{O}(H_0)\) and/or large \(f\) in thawing-like regimes.

## Peaks/valleys via Buchert averaging
For irrotational dust on a domain \(\mathcal D\) with \(a_\mathcal D\propto V_\mathcal D^{1/3}\), the Buchert-averaged equations are:
\[
3\left(\frac{\dot a_\mathcal D}{a_\mathcal D}\right)^2=8\pi G\langle\rho\rangle_\mathcal D-\tfrac12\langle\mathcal R\rangle_\mathcal D-\tfrac12\mathcal Q_\mathcal D,
\]
\[
3\frac{\ddot a_\mathcal D}{a_\mathcal D}=-4\pi G\langle\rho\rangle_\mathcal D+\mathcal Q_\mathcal D,
\]
so “peaks/valleys” operationalize into evolving averaged curvature \(\langle\mathcal R\rangle_\mathcal D\) and kinematical backreaction \(\mathcal Q_\mathcal D\), with acceleration possible when \(\mathcal Q_\mathcal D>4\pi G\langle\rho\rangle_\mathcal D\).

```mermaid
process
  title Task Step
  Theory : #TODO1 integrals + ODE solves
  Inference : #TODO2 sweeps + MCMC
  Perturbations : #TODO3 CMB + P(k)
  Paper : #TODO4 methods, results, figures, discussion
```
