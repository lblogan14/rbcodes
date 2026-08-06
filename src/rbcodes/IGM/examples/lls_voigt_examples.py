"""
LLSVoigtFitter — worked examples.

Two examples are provided:

  example_J1154_single_component()
      Single COS file, one HI component, Ly-beta through Ly-7 (6 transitions).
      Target: J1154+4635  z_abs = 0.528708

  example_J0906_multi_grating()
      Two COS gratings (G160M + G140L) with different spectral resolutions,
      three HI components, Ly-beta through Ly-20 (19 transitions).
      Target: J0906+015   z_abs = 0.5359

Run individual examples by calling their functions, or read this file as a
reference. Substitute your own spectrum files and redshift as needed.
"""

import numpy as np


def example_J1154_single_component(
    spec_file='/Users/bordoloi/Dropbox/COS-Pairs/Targets/J1154+4635/Data/J1154+4635_nbin3_coadd.fits',
    zabs=0.528708,
    sampler='zeus',
    nwalkers=64,
    nsteps=2000,
    burnin=400,
):
    """
    Simultaneous HI Voigt profile + LLS break fit for J1154+4635 at z = 0.528708.

    Voigt fit : HI Lyman series Ly-beta (1025.72 A) through Ly-7 (926.22 A)
    LLS fit   : Lyman limit break at 912 A (rest frame)
    Sampler   : zeus (default) or emcee

    Parameters
    ----------
    spec_file : str
        Path to the COS spectrum FITS file.
    zabs : float
        Absorber redshift.
    sampler : str
        'zeus' or 'emcee'.
    nwalkers : int
    nsteps : int
        Total MCMC steps per walker.
    burnin : int
        Steps discarded from the front of the chain.

    Returns
    -------
    fitter : LLSVoigtFitter
        Fitted object. Call fitter.get_results(), fitter.plot_fit(), etc.
    """
    from rbcodes.utils.rb_spectrum import rb_read_spec
    from rbcodes.IGM.LLSFitter import LLSFitter
    from rbcodes.IGM.LLSVoigtFitter import LLSVoigtFitter
    import rbvfit.vfit_mcmc as mc
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel, mean_fwhm_pixels

    # HI Lyman series rest wavelengths included in the Voigt fit
    HI_TRANSITIONS = [
        1025.7222,   # Ly beta
         972.5368,   # Ly gamma
         949.7431,   # Ly delta
         937.8035,   # Ly epsilon
         930.7483,   # Ly 6
         926.2257,   # Ly 7
        # 923.1504,  # Ly 8  — uncomment to include higher-order lines
        # 920.9631,  # Ly 9
    ]

    # ------------------------------------------------------------------
    # Load and normalize spectrum
    # ------------------------------------------------------------------
    spec  = rb_read_spec(spec_file)
    wave  = spec.wavelength.value
    flux  = spec.flux.value
    error = spec.sig.value

    # Normalize by median in a clean continuum window just redward of Ly-beta.
    # Adjust limits if there are strong features in your continuum window.
    cont_mask  = (wave > 1025.7222 * (1 + zabs) * 1.02) & \
                 (wave < 1025.7222 * (1 + zabs) * 1.08) & \
                 (error > 0)
    cont_level = np.nanmedian(flux[cont_mask])
    flux_norm  = flux  / cont_level
    error_norm = error / cont_level

    # ------------------------------------------------------------------
    # 1. LLSFitter — fits its own local linear continuum (C0, C1)
    # ------------------------------------------------------------------
    lls = LLSFitter()
    lls.wave  = wave
    lls.flux  = flux_norm
    lls.error = error_norm
    lls.set_redshift(zabs)

    # Continuum windows in rest-frame Angstroms — regions free of strong lines.
    # Adjust or remove windows that fall in COS gaps or noisy regions.
    lls.set_continuum_regions([
        (860,   872  ),   # clean below Lyman limit
        (893,   899  ),   # between high-order Ly lines
        (905,   910  ),   # just blueward of 912 A
        (918.5, 919  ),   # narrow window between high-order lines
        (927,   928  ),   # between Ly 6 and Ly 7
        (931,   933  ),   # between Ly 6 and Ly 5
        (934,   936  ),   # between Ly 5 and Ly epsilon
        (940,   944  ),   # between Ly epsilon and Ly delta
        (946,   948  ),   # between Ly delta and Ly gamma
        (951,   970  ),   # redward of Ly gamma, blueward of Ly beta
    ])
    lls.set_domain_range(wmin=860, wmax=975)

    # ------------------------------------------------------------------
    # 2. VoigtModel — continuum-normalized spectrum over Lyman series region
    # ------------------------------------------------------------------
    wmin_obs   = min(HI_TRANSITIONS) * (1 + zabs) * 0.998
    wmax_obs   = max(HI_TRANSITIONS) * (1 + zabs) * 1.002
    voigt_mask = (wave >= wmin_obs) & (wave <= wmax_obs) & (error_norm > 0)

    w_voigt = wave[voigt_mask]
    f_voigt = flux_norm[voigt_mask]
    e_voigt = error_norm[voigt_mask]

    config = FitConfiguration()
    config.add_system(z=zabs, ion='HI', transitions=HI_TRANSITIONS, components=1)

    # Convert 18 km/s instrumental FWHM to pixels on this wavelength grid
    fwhm_pix = mean_fwhm_pixels(18.0, w_voigt)
    print(f"FWHM 18.0 km/s = {fwhm_pix:.3f} pixels "
          f"(median wavelength {np.median(w_voigt):.1f} A)")

    voigt_model = VoigtModel(config, FWHM=str(fwhm_pix))

    # vfit instance — data baked in; do NOT call runmcmc() on this object.
    # Strip the C0, C1 tail — LLSVoigtFitter manages those.
    vfit_obj = mc.vfit(
        {'COS': {'model': voigt_model, 'wave': w_voigt,
                 'flux': f_voigt, 'error': e_voigt}},
        theta=[17.5,  25.0,   0.0],
        lb   =[14.0,   5.0, -500.0],
        ub   =[22.0, 100.0,  500.0],
    )

    # ------------------------------------------------------------------
    # 3. Joint fitter  —  theta = [logNHI, b, v, C0, C1]
    # ------------------------------------------------------------------
    theta_init = [17.5,  25.0,   0.0,  1.0,  0.0]
    lb         = [14.0,   5.0, -500.0, 0.1, -2.0]
    ub         = [22.0, 100.0,  500.0, 5.0,  2.0]

    fitter = LLSVoigtFitter(vfit_obj, lls, theta_init, lb, ub)

    print("\n--- Quick fit ---")
    fitter.fit_quick()

    print(f"\n--- MCMC ({sampler}) ---")
    # use_pool=True (default) parallelises likelihood evaluations across walkers.
    # Set use_pool=False when running inside a Jupyter notebook with %autoreload.
    fitter.fit(sampler=sampler, nwalkers=nwalkers,
               nsteps=nsteps, burnin=burnin, progress=True, use_pool=True)

    # ------------------------------------------------------------------
    # 4. Results + plots
    # ------------------------------------------------------------------
    results = fitter.get_results()
    print("\n=== Results ===")
    print(f"logNHI       = {results['logN'][0]:.3f} "
          f"+{results['err_hi'][0]:.3f} / -{results['err_lo'][0]:.3f}")
    print(f"logNHI total = {results['logNHI_total']:.3f}")
    print(f"b            = {results['b'][0]:.2f} km/s")
    print(f"v            = {results['v'][0]:.2f} km/s")
    print(f"C0, C1       = {results['C0']:.4f}, {results['C1']:.4f}")

    fitter.plot_fit()
    fitter.plot_diagnostics()
    fitter.plot_corner()

    return fitter


def example_J0906_multi_grating(
    lls_file='j0906p015_g160m_allVisits_binned3.fits',
    voigt_files=None,
    zabs=0.5359,
    target='J0906+015',
    sampler='emcee',
    nwalkers=64,
    nsteps=1000,
    burnin=400,
    outbase=None,
):
    """
    Simultaneous HI Voigt profile + LLS break fit for J0906+015 at z = 0.5359.

    Uses two COS gratings (G160M and G140L) with different spectral resolutions
    for the Voigt fit, and a single binned G160M file for the LLS break.
    Three HI components are fitted simultaneously.

    Voigt fit : HI Lyman series Ly-beta through Ly-20 (19 transitions)
    LLS fit   : Lyman limit break at 912 A (rest frame)

    Parameters
    ----------
    lls_file : str
        Path to the binned COS spectrum used for the LLS break fit.
    voigt_files : dict or None
        Dict mapping grating name to (file_path, fwhm_kms). Default:
        {'G160M': ('gal-J0906p015_g160m_final_spec.fits', 18.0),
         'G140L': ('gal-J0906p015_g140l_final_spec.fits', 40.0)}
    zabs : float
        Absorber redshift.
    target : str
        Target name, used for output file names.
    sampler : str
        'emcee' or 'zeus'.
    nwalkers : int
    nsteps : int
        Total MCMC steps per walker.
    burnin : int
        Steps discarded from the front of the chain.
    outbase : str or None
        Base name for output files (HDF5, PNG). Defaults to
        'LLSVoigtFit_{target}'.

    Returns
    -------
    fitter : LLSVoigtFitter
        Fitted object. Call fitter.get_results(), fitter.plot_fit(), etc.
    """
    import h5py
    from rbcodes.utils.rb_spectrum import rb_read_spec
    from rbcodes.IGM.LLSFitter import LLSFitter
    from rbcodes.IGM.LLSVoigtFitter import LLSVoigtFitter
    import rbvfit.vfit_mcmc as mc
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel, mean_fwhm_pixels

    if voigt_files is None:
        voigt_files = {
            'G160M': ('gal-J0906p015_g160m_final_spec.fits', 18.0),
            'G140L': ('gal-J0906p015_g140l_final_spec.fits', 40.0),
        }

    if outbase is None:
        outbase = f'LLSVoigtFit_{target}'

    # HI Lyman series rest wavelengths (A) — Ly-beta through Ly-20
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

    # ------------------------------------------------------------------
    # Load and normalize LLS spectrum
    # ------------------------------------------------------------------
    spec  = rb_read_spec(lls_file)
    wave  = spec.wavelength.value
    flux  = spec.flux.value
    error = spec.sig.value

    # Normalize by median in a clean continuum window just redward of Ly-beta.
    cont_mask  = (wave > 1025.7222 * (1 + zabs) * 1.02) & \
                 (wave < 1025.7222 * (1 + zabs) * 1.08) & \
                 (error > 0)
    cont_level = np.nanmedian(flux[cont_mask])
    flux_norm  = flux  / cont_level
    error_norm = error / cont_level

    # ------------------------------------------------------------------
    # Load and normalize Voigt spectra — one entry per grating/instrument
    # ------------------------------------------------------------------
    wmin_obs = min(HI_TRANSITIONS) * (1 + zabs) * 0.998
    wmax_obs = max(HI_TRANSITIONS) * (1 + zabs) * 1.002

    voigt_data = {}
    for grating, (vfile, fwhm_kms) in voigt_files.items():
        vs    = rb_read_spec(vfile)
        vwave = vs.wavelength.value * (1 + zabs)   # rest → observed frame
        vflux = vs.flux.value
        verr  = vs.sig.value

        print(f"{grating}: wavelength range {vwave.min():.1f} – {vwave.max():.1f} A  "
              f"(need {wmin_obs:.1f} – {wmax_obs:.1f} A)")

        # Normalize each grating independently using the same continuum window
        vcont_mask  = (vwave > 1025.7222 * (1 + zabs) * 1.02) & \
                      (vwave < 1025.7222 * (1 + zabs) * 1.08) & \
                      (verr > 0)
        vcont_level = np.nanmedian(vflux[vcont_mask]) if vcont_mask.any() else 1.0
        vflux_norm  = vflux / vcont_level
        verr_norm   = verr  / vcont_level

        vmask = (vwave >= wmin_obs) & (vwave <= wmax_obs) & (verr_norm > 0)
        if vmask.sum() < 2:
            print(f"  WARNING: {grating} has only {vmask.sum()} pixel(s) in the "
                  f"Voigt window — skipping.")
            continue
        voigt_data[grating] = {
            'wave':     vwave[vmask],
            'flux':     vflux_norm[vmask],
            'error':    verr_norm[vmask],
            'fwhm_kms': fwhm_kms,
        }
        print(f"  {grating}: {vmask.sum()} pixels in Voigt window, "
              f"cont_level = {vcont_level:.4f}")

    if not voigt_data:
        raise RuntimeError("No grating covers the Voigt wavelength window. "
                           "Check voigt_files and HI_TRANSITIONS.")

    # ------------------------------------------------------------------
    # 1. LLSFitter — handles its own local linear continuum (C0, C1)
    # ------------------------------------------------------------------
    lls = LLSFitter()
    lls.wave  = wave
    lls.flux  = flux_norm
    lls.error = error_norm
    lls.set_redshift(zabs)

    # Continuum windows in rest-frame Angstroms — regions free of strong lines.
    # Adjust or remove windows that fall in COS gaps or noisy regions.
    lls.set_continuum_regions([
        (860,   872  ),   # clean below Lyman limit
        (893,   899  ),   # between high-order Ly lines
        (905,   910  ),   # just blueward of 912 A
        (918.5, 919  ),   # narrow window between high-order lines
        (927,   928  ),   # between Ly 6 and Ly 7
        (931,   933  ),
        (934,   936  ),
        (940,   944  ),
    ])
    lls.set_domain_range(wmin=860, wmax=975)   # rest-frame A

    # ------------------------------------------------------------------
    # 2. VoigtModel — one model per grating (different FWHM per instrument)
    # ------------------------------------------------------------------
    config = FitConfiguration()
    config.add_system(z=zabs, ion='HI', transitions=HI_TRANSITIONS, components=3)

    vfit_instruments = {}
    for grating, d in voigt_data.items():
        fwhm_pix = mean_fwhm_pixels(d['fwhm_kms'], d['wave'])
        print(f"{grating}: FWHM {d['fwhm_kms']} km/s = {fwhm_pix:.3f} pixels "
              f"(median wavelength {np.median(d['wave']):.1f} A)")
        vfit_instruments[grating] = {
            'model': VoigtModel(config, FWHM=str(fwhm_pix)),
            'wave':  d['wave'],
            'flux':  d['flux'],
            'error': d['error'],
        }

    # ------------------------------------------------------------------
    # 3. Joint fitter
    # theta layout (n=3): [logN1, logN2, logN3,  b1, b2, b3,  v1, v2, v3,  C0, C1]
    # ------------------------------------------------------------------
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

    print("\n--- Quick fit ---")
    fitter.fit_quick()

    print(f"\n--- MCMC ({sampler}) ---")
    fitter.fit(sampler=sampler, nwalkers=nwalkers,
               nsteps=nsteps, burnin=burnin, progress=True, use_pool=True)

    # ------------------------------------------------------------------
    # 4. Results
    # ------------------------------------------------------------------
    results = fitter.get_results()
    print("\n=== Results ===")
    for i in range(len(results['logN'])):
        print(f"Component {i+1}: "
              f"logNHI = {results['logN'][i]:.3f} "
              f"+{results['err_hi'][i]:.3f} / -{results['err_lo'][i]:.3f}  "
              f"b = {results['b'][i]:.2f} km/s  "
              f"v = {results['v'][i]:.2f} km/s")
    print(f"logNHI total = {results['logNHI_total']:.3f}")
    print(f"C0, C1       = {results['C0']:.4f}, {results['C1']:.4f}")

    # ------------------------------------------------------------------
    # 5. Save results to HDF5
    # ------------------------------------------------------------------
    h5file = f'{outbase}.h5'
    with h5py.File(h5file, 'w') as hf:
        hf.attrs['target'] = target
        hf.attrs['zabs']   = zabs

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

    print(f"Results + MCMC chain saved to {h5file}")

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    fitter.plot_fit(outfile=f'{outbase}.png')
    fitter.plot_diagnostics(outfile=f'{outbase}_traces.png')
    fitter.plot_corner(outfile=f'{outbase}_corner.png')

    return fitter


if __name__ == '__main__':
    # Uncomment the example you want to run:
    # fitter = example_J1154_single_component()
    fitter = example_J0906_multi_grating()
