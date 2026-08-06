# LLSVoigtFitter

[Back to IGM Tools](IGM_README.md) | [Back to Main](../main_readme.md)

Simultaneous fit of HI Lyman series Voigt profiles and the Lyman Limit System
(LLS) opacity break at 912 Å using a joint MCMC likelihood.

---

## Overview

`LLSVoigtFitter` wraps `rbvfit.vfit_mcmc.vfit` (Voigt profiles) and
`rbcodes.IGM.LLSFitter` (LLS break), combining their log-likelihoods over a
shared parameter vector. The total HI column density couples both models:
per-component `(logN_i, b_i, v_i)` drives the Voigt profiles, and
`N_total = Σ 10^logN_i` drives the Lyman limit opacity.

Both `emcee` and `zeus` samplers are supported.

---

## Parameter Vector

```
theta = [logN_1, ..., logN_n,  b_1, ..., b_n,  v_1, ..., v_n,  C0, C1]
```

| Parameter | Description |
|-----------|-------------|
| `logN_i`  | Log₁₀ HI column density of component i (cm⁻²) |
| `b_i`     | Doppler parameter of component i (km/s) |
| `v_i`     | Velocity offset of component i (km/s) |
| `C0, C1`  | Local linear continuum around the LLS break: `flux = C0 + C1*(λ_rest − 911)` |

For `n=1` (single component): `theta = [logNHI, b, v, C0, C1]`

---

## Quick Start

```python
import rbvfit.vfit_mcmc as mc
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel, mean_fwhm_pixels
from rbcodes.IGM.LLSFitter import LLSFitter
from rbcodes.IGM.LLSVoigtFitter import LLSVoigtFitter

# --- Voigt model (Lyman series lines, continuum-normalized spectrum) ---
config = FitConfiguration()
config.add_system(z=0.5287, ion='HI',
                  transitions=[1025.72, 972.54, 949.74, 937.80, 930.75, 926.23],
                  components=1)
fwhm_pix    = mean_fwhm_pixels(18.0, w_voigt)   # 18 km/s -> pixels
voigt_model = VoigtModel(config, FWHM=str(fwhm_pix))

vfit_obj = mc.vfit(
    {'COS': {'model': voigt_model, 'wave': w_voigt, 'flux': f_voigt, 'error': e_voigt}},
    theta=[17.5, 25.0, 0.0], lb=[14.0, 5.0, -500.0], ub=[22.0, 100.0, 500.0]
)

# --- LLSFitter (raw/normalized spectrum, fits its own local continuum) ---
lls = LLSFitter()
lls.wave  = wave
lls.flux  = flux_norm
lls.error = error_norm
lls.set_redshift(0.5287)
lls.set_continuum_regions([         # rest-frame Angstroms, free of strong lines
    (860, 872), (893, 899), (905, 910), (918.5, 919),
    (927, 928), (931, 933), (934, 936), (940, 944), (946, 948), (951, 970),
])
lls.set_domain_range(wmin=860, wmax=975)

# --- Joint fitter ---
fitter = LLSVoigtFitter(
    vfit_obj, lls,
    theta_init=[17.5, 25.0, 0.0, 1.0, 0.0],
    lb        =[14.0,  5.0, -500.0, 0.1, -2.0],
    ub        =[22.0, 100.0, 500.0, 5.0,  2.0],
)

fitter.fit_quick()                                        # scipy starting point
fitter.fit(sampler='zeus', nwalkers=64, nsteps=2000, burnin=400)

print(fitter.get_results())
fitter.plot_fit()
fitter.plot_diagnostics()
fitter.plot_corner()
```

---

## API Reference

### `LLSVoigtFitter(vfit_obj, lls_obj, theta_init, lb, ub)`

**Parameters**

| Argument | Type | Description |
|----------|------|-------------|
| `vfit_obj` | `rbvfit.vfit_mcmc.vfit` | Pre-built vfit instance with data baked in. Do **not** call `runmcmc()` on it. |
| `lls_obj`  | `LLSFitter` | Pre-configured LLSFitter. Must have `set_redshift()` and `set_continuum_regions()` called. |
| `theta_init` | array | Initial guess for full parameter vector. |
| `lb`, `ub` | array | Lower and upper bounds, same length as `theta_init`. |

---

### `fit_quick(verbose=True)`

Fast scipy L-BFGS-B optimization. Updates `self.theta` in place — use as a
starting point before MCMC.

---

### `fit(sampler='emcee', nwalkers=64, nsteps=1000, burnin=200, progress=True)`

Run MCMC. Returns `(sampler_obj, flat_samples)`.

| Argument | Default | Description |
|----------|---------|-------------|
| `sampler` | `'emcee'` | `'emcee'` or `'zeus'` |
| `nwalkers` | 64 | Number of walkers |
| `nsteps` | 1000 | Total steps per walker |
| `burnin` | 200 | Steps discarded from chain front |

---

### `get_results()`

Returns a dict with keys: `logN`, `b`, `v`, `C0`, `C1`, `logNHI_total`,
`err_lo`, `err_hi` (16th/84th percentile offsets).

---

### `plot_fit(n_draw=100, figsize=(12,8), outfile=None)`

Two-panel figure: Lyman series Voigt profiles (top) and LLS break (bottom),
with MCMC realizations overlaid.

---

### `plot_diagnostics(figsize=(10, 2.5), outfile=None)`

Walker trace plots for all parameters. Prints sampler-specific convergence
statistics to the terminal:

- **emcee**: mean acceptance fraction + autocorrelation time per parameter
- **zeus**: Gelman-Rubin R-hat per parameter (converged if R-hat < 1.1)

---

### `plot_corner(figsize=(8,8), outfile=None)`

Corner plot of posterior samples with 16/50/84 percentile titles.

---

## Notes

- The two spectra fed to `vfit_obj` and `lls_obj` can be **completely
  independent** — different files, instruments, or wavelength grids. Only the
  parameter vector couples them.
- `LLSVoigtFitter` does not own its own MCMC loop — it borrows `vfit.lnlike()`
  and `LLSFitter.lnlike()` and runs its own sampler.
- For multi-component fits, the LLS break uses `N_total = Σ 10^logN_i`, which
  is physically correct: the Lyman limit opacity depends on total integrated
  column, not velocity structure.

---

## Example Script

See `src/rbcodes/IGM/examples/lls_voigt_examples.py` for a complete worked
example on J1154+4635 at z = 0.528708.
