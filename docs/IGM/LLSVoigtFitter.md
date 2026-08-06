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

Multiple gratings/instruments with different spectral resolutions are supported
for the Voigt side — each grating gets its own `VoigtModel` with its own FWHM.

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
For `n=3` (three components): `theta = [logN1, logN2, logN3, b1, b2, b3, v1, v2, v3, C0, C1]`

**Important:** When building `vfit_obj`, strip the trailing `C0, C1` entries
from `theta_init`, `lb`, and `ub` — `LLSVoigtFitter` manages those internally.

---

## Quick Start

The example below fits J0906+015 at z = 0.5359 with two COS gratings (G160M
and G140L) and three HI components. Adapt the file paths and continuum windows
for your target.

```python
import numpy as np
import rbvfit.vfit_mcmc as mc
from rbvfit.core.fit_configuration import FitConfiguration
from rbvfit.core.voigt_model import VoigtModel, mean_fwhm_pixels
from rbcodes.utils.rb_spectrum import rb_read_spec
from rbcodes.IGM.LLSFitter import LLSFitter
from rbcodes.IGM.LLSVoigtFitter import LLSVoigtFitter

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
LLS_FILE    = 'j0906p015_g160m_allVisits_binned3.fits'
VOIGT_FILES = {
    'G160M': ('gal-J0906p015_g160m_final_spec.fits', 18.0),  # (file, FWHM km/s)
    'G140L': ('gal-J0906p015_g140l_final_spec.fits', 40.0),
}
ZABS   = 0.5359
TARGET = 'J0906+015'

# HI Lyman series rest wavelengths (Å) — include all lines covered by your data
HI_TRANSITIONS = [
    1025.7222,   # Ly beta
     972.5368,   # Ly gamma
     949.7431,   # Ly delta
     937.8035,   # Ly epsilon
     930.7483,   # Ly 6
     926.2257,   # Ly 7
     923.1504,   # Ly 8
     920.9631,   # Ly 9
     919.3514,   # Ly 10
     918.1294,   # Ly 11
     917.1806,   # Ly 12
     916.4293,   # Ly 13
     915.8239,   # Ly 14
     915.3288,   # Ly 15
     914.9191,   # Ly 16
     914.5760,   # Ly 17
     914.2860,   # Ly 18
     914.0393,   # Ly 19
     913.8260,   # Ly 20
]

# -----------------------------------------------------------------------
# Load and normalize LLS spectrum
# -----------------------------------------------------------------------
spec  = rb_read_spec(LLS_FILE)
wave  = spec.wavelength.value
flux  = spec.flux.value
error = spec.sig.value

# Normalize by median in a clean continuum window just redward of Ly-beta.
# Adjust these limits if there are strong features in your continuum window.
cont_mask  = (wave > 1025.7222 * (1 + ZABS) * 1.02) & \
             (wave < 1025.7222 * (1 + ZABS) * 1.08) & \
             (error > 0)
cont_level = np.nanmedian(flux[cont_mask])
flux_norm  = flux  / cont_level
error_norm = error / cont_level

# -----------------------------------------------------------------------
# Load and normalize Voigt spectra — one entry per grating/instrument
# -----------------------------------------------------------------------
wmin_obs = min(HI_TRANSITIONS) * (1 + ZABS) * 0.998
wmax_obs = max(HI_TRANSITIONS) * (1 + ZABS) * 1.002

voigt_data = {}
for grating, (vfile, fwhm_kms) in VOIGT_FILES.items():
    vs    = rb_read_spec(vfile)
    vwave = vs.wavelength.value * (1 + ZABS)   # rest → observed frame
    vflux = vs.flux.value
    verr  = vs.sig.value

    # Normalize each grating independently using the same continuum window
    vcont_mask  = (vwave > 1025.7222 * (1 + ZABS) * 1.02) & \
                  (vwave < 1025.7222 * (1 + ZABS) * 1.08) & \
                  (verr > 0)
    vcont_level = np.nanmedian(vflux[vcont_mask]) if vcont_mask.any() else 1.0
    vflux_norm  = vflux / vcont_level
    verr_norm   = verr  / vcont_level

    vmask = (vwave >= wmin_obs) & (vwave <= wmax_obs) & (verr_norm > 0)
    if vmask.sum() < 2:
        print(f"WARNING: {grating} has only {vmask.sum()} pixel(s) — skipping.")
        continue
    voigt_data[grating] = {
        'wave':     vwave[vmask],
        'flux':     vflux_norm[vmask],
        'error':    verr_norm[vmask],
        'fwhm_kms': fwhm_kms,
    }

# -----------------------------------------------------------------------
# 1. LLSFitter — handles its own local linear continuum (C0, C1)
# -----------------------------------------------------------------------
lls = LLSFitter()
lls.wave  = wave
lls.flux  = flux_norm
lls.error = error_norm
lls.set_redshift(ZABS)

# Continuum windows in rest-frame Å — regions free of strong lines.
# Adjust or remove windows that fall in COS gaps or noisy regions.
lls.set_continuum_regions([
    (860,   872  ),   # clean below Lyman limit
    (893,   899  ),   # between high-order Ly lines
    (905,   910  ),   # just blueward of 912 Å
    (918.5, 919  ),   # narrow window between high-order lines
    (927,   928  ),   # between Ly 6 and Ly 7
    (931,   933  ),
    (934,   936  ),
    (940,   944  ),
])
lls.set_domain_range(wmin=860, wmax=975)   # rest-frame Å

# -----------------------------------------------------------------------
# 2. VoigtModel — one model per grating (different FWHM per instrument)
# -----------------------------------------------------------------------
config = FitConfiguration()
config.add_system(z=ZABS, ion='HI', transitions=HI_TRANSITIONS, components=3)

vfit_instruments = {}
for grating, d in voigt_data.items():
    fwhm_pix = mean_fwhm_pixels(d['fwhm_kms'], d['wave'])
    vfit_instruments[grating] = {
        'model': VoigtModel(config, FWHM=str(fwhm_pix)),
        'wave':  d['wave'],
        'flux':  d['flux'],
        'error': d['error'],
    }

# -----------------------------------------------------------------------
# 3. Joint fitter
# theta layout (n=3): [logN1, logN2, logN3,  b1, b2, b3,  v1, v2, v3,  C0, C1]
# -----------------------------------------------------------------------
#            logN1    logN2    logN3     b1    b2   b3      v1      v2     v3    C0    C1
theta_init = [15.536, 16.630,  15.487,  30.6, 32.2,  3.7, -109.6, -18.2,  27.2,  1.0,  0.0]
lb         = [14.0,   14.0,    14.0,     1.0,  1.0,  1.0, -500.0,-500.0,-500.0,  0.1, -2.0]
ub         = [17.0,   17.8,    17.0,   100.0,100.0,100.0,  500.0, 500.0, 500.0,  5.0,  2.0]

# vfit_obj: strip the C0, C1 tail before passing to mc.vfit
n_voigt  = len(theta_init) - 2
vfit_obj = mc.vfit(
    vfit_instruments,
    theta=theta_init[:n_voigt],
    lb   =lb[:n_voigt],
    ub   =ub[:n_voigt],
)

fitter = LLSVoigtFitter(vfit_obj, lls, theta_init, lb, ub)

# Step 1: quick scipy optimization — good starting point for MCMC
fitter.fit_quick()

# Step 2: MCMC
fitter.fit(sampler='emcee', nwalkers=64, nsteps=1000, burnin=400,
           progress=True, use_pool=True)

# -----------------------------------------------------------------------
# 4. Results
# -----------------------------------------------------------------------
results = fitter.get_results()
for i in range(len(results['logN'])):
    print(f"Component {i+1}: "
          f"logNHI = {results['logN'][i]:.3f} "
          f"+{results['err_hi'][i]:.3f} / -{results['err_lo'][i]:.3f}  "
          f"b = {results['b'][i]:.2f} km/s  "
          f"v = {results['v'][i]:.2f} km/s")
print(f"logNHI total = {results['logNHI_total']:.3f}")
print(f"C0, C1       = {results['C0']:.4f}, {results['C1']:.4f}")

# -----------------------------------------------------------------------
# 5. Save to HDF5
# -----------------------------------------------------------------------
import h5py

outbase = f'LLSVoigtFit_{TARGET}'
with h5py.File(f'{outbase}.h5', 'w') as hf:
    hf.attrs['target'] = TARGET
    hf.attrs['zabs']   = ZABS

    res = hf.create_group('results')
    res.create_dataset('logN',   data=results['logN'])
    res.create_dataset('b',      data=results['b'])
    res.create_dataset('v',      data=results['v'])
    res.create_dataset('err_lo', data=results['err_lo'])
    res.create_dataset('err_hi', data=results['err_hi'])
    res.attrs['logNHI_total'] = results['logNHI_total']
    res.attrs['C0']           = results['C0']
    res.attrs['C1']           = results['C1']

    # full flat MCMC chain (post-burnin)
    hf.create_dataset('samples', data=fitter.samples,
                      compression='gzip', compression_opts=4)
    hf.attrs['param_labels'] = fitter._param_labels()

# -----------------------------------------------------------------------
# 6. Plots
# -----------------------------------------------------------------------
fitter.plot_fit(outfile=f'{outbase}.png')
fitter.plot_diagnostics(outfile=f'{outbase}_traces.png')
fitter.plot_corner(outfile=f'{outbase}_corner.png')
```

---

## API Reference

### `LLSVoigtFitter(vfit_obj, lls_obj, theta_init, lb, ub)`

**Parameters**

| Argument | Type | Description |
|----------|------|-------------|
| `vfit_obj` | `rbvfit.vfit_mcmc.vfit` | Pre-built vfit instance with data baked in. Strip the `C0, C1` tail from `theta/lb/ub` before passing. Do **not** call `runmcmc()` on it. |
| `lls_obj`  | `LLSFitter` | Pre-configured LLSFitter. Must have `set_redshift()` and `set_continuum_regions()` called. |
| `theta_init` | array | Initial guess for full parameter vector (including `C0, C1`). |
| `lb`, `ub` | array | Lower and upper bounds, same length as `theta_init`. |

---

### `fit_quick(verbose=True)`

Fast scipy L-BFGS-B optimization. Updates `self.theta` in place — use as a
starting point before MCMC.

---

### `fit(sampler='emcee', nwalkers=64, nsteps=1000, burnin=200, progress=True, use_pool=True)`

Run MCMC. Returns `(sampler_obj, flat_samples)`.

| Argument | Default | Description |
|----------|---------|-------------|
| `sampler` | `'emcee'` | `'emcee'` or `'zeus'` |
| `nwalkers` | 64 | Number of walkers |
| `nsteps` | 1000 | Total steps per walker |
| `burnin` | 200 | Steps discarded from chain front |
| `use_pool` | `True` | Parallelise likelihood evaluations with a multiprocessing pool. Uses `fork` on Mac/Linux and `spawn` on Windows. Set `False` when running inside a Jupyter notebook with `%autoreload`. |

After `fit()` completes, the post-burnin flat chain is stored in `fitter.samples`.

---

### `get_results()`

Returns a dict with keys:

| Key | Description |
|-----|-------------|
| `logN` | Array of per-component log₁₀ NHI (median posterior) |
| `b` | Array of per-component Doppler b (km/s) |
| `v` | Array of per-component velocity offset (km/s) |
| `err_lo`, `err_hi` | 16th/84th percentile offsets (same length as `logN`, `b`, `v`) |
| `logNHI_total` | log₁₀ of summed column density |
| `C0`, `C1` | LLS local continuum parameters |

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

### `_param_labels()`

Returns a list of human-readable parameter labels (e.g. `['logN_1', 'logN_2',
..., 'b_1', ..., 'C0', 'C1']`). Useful for labelling HDF5 datasets or corner
plot axes.

---

### `fitter.samples`

Flat post-burnin MCMC chain, shape `(n_samples, n_params)`. Available after
`fit()` has been called. Suitable for direct archival in HDF5.

---

## Notes

- The two spectra fed to `vfit_obj` and `lls_obj` can be **completely
  independent** — different files, instruments, or wavelength grids. Only the
  parameter vector couples them.
- For multi-grating Voigt fits, pass a dict of `{grating: {'model': ...,
  'wave': ..., 'flux': ..., 'error': ...}}` to `mc.vfit`. Each grating gets
  its own `VoigtModel` with the appropriate FWHM in pixels.
- `LLSVoigtFitter` does not own its own MCMC loop — it borrows `vfit.lnlike()`
  and `LLSFitter.lnlike()` and runs its own sampler.
- For multi-component fits, the LLS break uses `N_total = Σ 10^logN_i`, which
  is physically correct: the Lyman limit opacity depends on total integrated
  column, not velocity structure.
- **Multiprocessing**: `fit()` uses a `fork`-based pool on Mac/Linux
  (`spawn` on Windows) to parallelise likelihood evaluations across walkers,
  following the same pattern as `rbvfit`. The pool is always closed and joined
  cleanly, even if `run_mcmc` raises. If the pool cannot be created (e.g.,
  inside certain Jupyter environments), the fitter falls back to single-process
  automatically with a warning. Pass `use_pool=False` to opt out explicitly.
- Gratings that have fewer than 2 pixels in the Voigt wavelength window are
  skipped automatically with a warning.

---

## Example Script

See `example_LLSVoigtFitter.py` in the COS-Blue working directory for a complete
worked example covering two targets:

- **J1154+4635** at z = 0.528708 — single COS file covering both LLS and Voigt regions
- **J0906+015** at z = 0.5359 — two gratings (G160M + G140L) with different spectral resolutions
