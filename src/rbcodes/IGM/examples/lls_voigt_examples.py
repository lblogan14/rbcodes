"""
LLSVoigtFitter — worked examples.

Run individual examples by calling their functions, or read this file as a
reference. Substitute your own spectrum file and redshift as needed.
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

    # vfit instance — data baked in; do NOT call runmcmc() on this object
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


if __name__ == '__main__':
    fitter = example_J1154_single_component()
