"""
LLSVoigtFitter: simultaneous fit of HI Lyman series Voigt profiles
and the Lyman Limit System (LLS) opacity break at 912 Angstrom.

Wraps rbvfit.vfit_mcmc.vfit and rbcodes.IGM.LLSFitter, combining their
log-likelihoods over a shared parameter vector. The total HI column density
enters both models: per-component (N_i, b_i, v_i) for the Voigt profiles,
and summed (N_total = sum 10^N_i) for the LLS break opacity.

Parameter vector layout:
    theta = [logN_1, ..., logN_n,  b_1, ..., b_n,  v_1, ..., v_n,  C0, C1]

where n is the number of velocity components, and C0, C1 are the local
linear continuum parameters for the LLS break region.

Example
-------
    import rbvfit.vfit_mcmc as mc
    from rbvfit.core.fit_configuration import FitConfiguration
    from rbvfit.core.voigt_model import VoigtModel
    from rbcodes.IGM.LLSFitter import LLSFitter
    from rbcodes.IGM.LLSVoigtFitter import LLSVoigtFitter

    # Build vfit object (data baked in, do NOT call runmcmc)
    config = FitConfiguration()
    config.add_system(z=0.5286, ion='HI',
                      transitions=[1215.67, 1025.72, 972.54], components=1)
    model = VoigtModel(config, FWHM='6.5')
    vfit_obj = mc.vfit(
        {'COS': {'model': model, 'wave': w, 'flux': f, 'error': e}},
        theta=[17.0, 25.0, 0.0], lb=[14, 5, -500], ub=[22, 100, 500]
    )

    # Build LLSFitter object (set spectrum directly, no file needed)
    lls = LLSFitter()
    lls.wave = w_lls; lls.flux = f_lls; lls.error = e_lls
    lls.set_redshift(0.5286)
    lls.set_continuum_regions()

    # Joint fit
    theta_init = [17.0, 25.0,  0.0,  1.0, 0.0]
    lb         = [14.0,  5.0, -500, -10.0, -10.0]
    ub         = [22.0, 100.0, 500,  10.0,  10.0]

    fitter = LLSVoigtFitter(vfit_obj, lls, theta_init, lb, ub)
    fitter.fit_quick()
    fitter.fit(sampler='emcee', nwalkers=64, nsteps=2000, burnin=300)
    print(fitter.get_results())
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
from scipy.optimize import minimize

try:
    import zeus
    HAS_ZEUS = True
except ImportError:
    HAS_ZEUS = False


class LLSVoigtFitter:
    """
    Simultaneous Voigt profile + LLS break fitter for HI absorbers.

    Parameters
    ----------
    vfit_obj : rbvfit.vfit_mcmc.vfit
        Pre-constructed vfit instance with instrument data baked in.
        Only lnlike() and ndim are used.
    lls_obj : rbcodes.IGM.LLSFitter.LLSFitter
        Pre-configured LLSFitter instance. Must have wave, flux, error
        and rest_wave set (call set_redshift()) and set_continuum_regions()
        called before passing here.
    theta_init : array_like
        Initial parameters [logN_1..n, b_1..n, v_1..n, C0, C1].
    lb, ub : array_like
        Lower and upper bounds, same length as theta_init.
    """

    def __init__(self, vfit_obj, lls_obj, theta_init, lb, ub):
        self.vfit  = vfit_obj
        self.lls   = lls_obj
        self.theta = np.asarray(theta_init, dtype=float)
        self.lb    = np.asarray(lb, dtype=float)
        self.ub    = np.asarray(ub, dtype=float)
        self.ndim  = len(self.theta)

        if self.ndim != len(self.lb) or self.ndim != len(self.ub):
            raise ValueError("theta_init, lb, and ub must have the same length")

        # n_voigt = vfit.ndim = 3 * n_components  [logN..., b..., v...]
        self.n_voigt = vfit_obj.ndim
        self.n_comp  = self.n_voigt // 3

        if self.ndim != self.n_voigt + 2:
            raise ValueError(
                f"theta length ({self.ndim}) must equal n_voigt ({self.n_voigt}) + 2 for [C0, C1]"
            )

        if lls_obj.rest_wave is None:
            raise RuntimeError(
                "lls_obj.rest_wave is None — call lls_obj.set_redshift(zabs) first"
            )

        # Apply continuum mask once at init and cache the masked arrays
        mask = lls_obj.get_continuum_mask()
        if not np.any(mask):
            raise RuntimeError(
                "LLS continuum mask is empty — check lls_obj.set_continuum_regions()"
            )
        self.lls_wave  = lls_obj.rest_wave[mask]  # rest-frame, as model_flx expects
        self.lls_flux  = lls_obj.flux[mask]
        self.lls_error = lls_obj.error[mask]

        # Results (populated after fit())
        self.sampler      = None
        self.sampler_name = None
        self.burnin       = None
        self.samples      = None
        self.best_theta   = None
        self.low_theta    = None
        self.high_theta   = None

    # ------------------------------------------------------------------
    # Probability functions
    # ------------------------------------------------------------------

    def lnprior(self, theta):
        """Uniform prior within bounds."""
        if np.any(theta < self.lb) or np.any(theta > self.ub):
            return -np.inf
        return 0.0

    def lnlike(self, theta):
        """
        Combined log-likelihood: Voigt profiles + LLS break.

        The Voigt portion uses theta[:n_voigt] passed directly to vfit.
        The LLS portion uses total NHI = sum(10^logN_i) and C0, C1.
        """
        # Voigt lines
        ll_voigt = self.vfit.lnlike(theta[:self.n_voigt])

        # LLS break: sum column densities across components
        N_total   = np.sum(10.0 ** theta[:self.n_comp])
        theta_lls = np.array([theta[-2], theta[-1], np.log10(N_total)])
        ll_lls    = self.lls.lnlike(theta_lls,
                                    self.lls_wave,
                                    self.lls_flux,
                                    self.lls_error)
        return ll_voigt + ll_lls

    def lnprob(self, theta):
        """Log posterior = log prior + log likelihood."""
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.lnlike(theta)
        return -np.inf if not np.isfinite(ll) else lp + ll

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_quick(self, verbose=True):
        """
        Fast scipy L-BFGS-B optimization. Updates self.theta in place.
        Run this before fit() to get a good MCMC starting point.
        """
        result = minimize(
            lambda t: -self.lnlike(t),
            self.theta,
            method='L-BFGS-B',
            bounds=list(zip(self.lb, self.ub))
        )
        if verbose:
            status = 'converged' if result.success else 'did not converge'
            logN_tot = np.log10(np.sum(10.0 ** result.x[:self.n_comp]))
            print(f"Quick fit {status}")
            print(f"  logN per component : {result.x[:self.n_comp]}")
            print(f"  logNHI total       : {logN_tot:.3f}")
            print(f"  b  per component   : {result.x[self.n_comp:2*self.n_comp]}")
            print(f"  v  per component   : {result.x[2*self.n_comp:3*self.n_comp]}")
            print(f"  C0, C1             : {result.x[-2]:.4f}, {result.x[-1]:.4f}")
        self.theta = result.x
        return result

    def fit(self, sampler='emcee', nwalkers=64, nsteps=1000,
            burnin=200, progress=True):
        """
        Run MCMC. Call fit_quick() first for a good starting point.

        Parameters
        ----------
        sampler : str
            'emcee' (default) or 'zeus'.
        nwalkers : int
            Number of walkers.
        nsteps : int
            Total steps per walker (including burnin).
        burnin : int
            Steps discarded from the front of the chain.
        progress : bool
            Show tqdm progress bar.

        Returns
        -------
        sampler_obj : emcee or zeus EnsembleSampler
        flat_samples : np.ndarray, shape (n_samples, ndim)
        """
        p0 = self.theta + 1e-4 * np.random.randn(nwalkers, self.ndim)
        p0 = np.clip(p0, self.lb + 1e-10, self.ub - 1e-10)

        if sampler == 'emcee':
            s = emcee.EnsembleSampler(nwalkers, self.ndim, self.lnprob)
        elif sampler == 'zeus':
            if not HAS_ZEUS:
                raise ImportError("zeus not installed: pip install zeus-mcmc")
            s = zeus.EnsembleSampler(nwalkers, self.ndim, self.lnprob)
        else:
            raise ValueError(f"sampler must be 'emcee' or 'zeus', got '{sampler}'")

        s.run_mcmc(p0, nsteps, progress=progress)
        self.sampler      = s
        self.sampler_name = sampler
        self.burnin       = burnin

        flat = s.get_chain(discard=burnin, flat=True)
        self.samples    = flat
        self.best_theta = np.median(flat, axis=0)
        self.low_theta  = np.percentile(flat, 16, axis=0)
        self.high_theta = np.percentile(flat, 84, axis=0)
        return s, flat

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self):
        """
        Return best-fit parameters and 1-sigma uncertainties.

        Returns
        -------
        dict with keys:
            logN, b, v          : median per-component values
            C0, C1              : LLS continuum parameters
            logNHI_total        : log10 of summed column density
            err_lo, err_hi      : 16th/84th percentile offsets (same shape as best_theta)
        """
        if self.best_theta is None:
            raise RuntimeError("Run fit() first")

        n   = self.n_comp
        med = self.best_theta
        lo  = self.low_theta
        hi  = self.high_theta

        return {
            'logN'        : med[:n],
            'b'           : med[n:2*n],
            'v'           : med[2*n:3*n],
            'C0'          : med[-2],
            'C1'          : med[-1],
            'logNHI_total': np.log10(np.sum(10.0 ** med[:n])),
            'err_lo'      : med - lo,
            'err_hi'      : hi - med,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _param_labels(self):
        """Human-readable parameter labels for plots."""
        n = self.n_comp
        labels  = [f'logN_{i+1}' for i in range(n)] if n > 1 else ['logNHI']
        labels += [f'b_{i+1} (km/s)' for i in range(n)] if n > 1 else ['b (km/s)']
        labels += [f'v_{i+1} (km/s)' for i in range(n)] if n > 1 else ['v (km/s)']
        labels += ['C0', 'C1']
        return labels

    def plot_fit(self, n_draw=100, figsize=(12, 8), outfile=None):
        """
        Two-panel plot: Lyman series Voigt profiles (top) and LLS break (bottom).

        Parameters
        ----------
        n_draw : int
            Number of random MCMC draws to overplot.
        figsize : tuple
        outfile : str, optional
            Save figure to this path if provided.
        """
        if self.best_theta is None:
            raise RuntimeError("Run fit() first")

        results = self.get_results()
        best    = self.best_theta
        draw_idx = np.random.choice(len(self.samples), size=min(n_draw, len(self.samples)),
                                    replace=False)

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
        fig.suptitle(
            f'log$N_{{HI}}$ = {results["logNHI_total"]:.2f} '
            f'+{results["err_hi"][0]:.2f}/$-${results["err_lo"][0]:.2f}  '
            f'$b$ = {results["b"][0]:.1f} km/s  '
            f'$v$ = {results["v"][0]:.1f} km/s',
            fontsize=11
        )

        # ---- Top panel: Voigt profiles (one panel per instrument) ----
        ax = axes[0]
        for inst_name, data in self.vfit.instrument_data.items():
            w   = data['wave']
            f   = data['flux']
            e   = data['error']
            rw  = w / (1 + self.lls.zabs)
            ax.step(rw, f, 'k-', lw=0.8, where='mid', alpha=0.7, label='Data')
            ax.fill_between(rw, f - e, f + e, color='gray', alpha=0.2)

            model_fn = data['model']
            for idx in draw_idx:
                mf = model_fn(self.samples[idx, :self.n_voigt], w)
                ax.plot(rw, mf, color='steelblue', alpha=0.05, lw=0.7)

            best_mf = model_fn(best[:self.n_voigt], w)
            ax.plot(rw, best_mf, 'r-', lw=1.5, label='Best fit')

        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_ylim(-0.15, 1.35)
        ax.set_xlabel('Rest Wavelength (Å)')
        ax.set_ylabel('Normalised Flux')
        ax.set_title('HI Lyman Series')
        ax.legend(fontsize=9)

        # ---- Bottom panel: LLS break ----
        ax = axes[1]
        rw_full  = self.lls.rest_wave
        fl_full  = self.lls.flux
        er_full  = self.lls.error
        dom      = self.lls.domain_range
        dom_mask = (rw_full >= dom[0]) & (rw_full <= dom[1])

        ax.step(rw_full[dom_mask], fl_full[dom_mask], 'k-', lw=0.8,
                where='mid', alpha=0.7, label='Data')
        ax.fill_between(rw_full[dom_mask],
                        fl_full[dom_mask] - er_full[dom_mask],
                        fl_full[dom_mask] + er_full[dom_mask],
                        color='gray', alpha=0.2)

        rw_dom = rw_full[dom_mask]
        for idx in draw_idx:
            th    = self.samples[idx]
            N_tot = np.sum(10.0 ** th[:self.n_comp])
            mf    = self.lls.model_flx([th[-2], th[-1], np.log10(N_tot)], rw_dom)
            ax.plot(rw_dom, mf, color='steelblue', alpha=0.05, lw=0.7)

        N_tot_best = np.sum(10.0 ** best[:self.n_comp])
        best_lls   = self.lls.model_flx([best[-2], best[-1], np.log10(N_tot_best)], rw_dom)
        ax.plot(rw_dom, best_lls, 'r-', lw=1.5, label='Best fit')

        ax.plot(self.lls_wave, self.lls_flux, 'r.', ms=3, alpha=0.5,
                label='Continuum pts used')
        ax.axvline(912, color='orange', lw=1.2, ls='--', label='912 Å')
        ax.axhline(0, color='gray', lw=0.5, ls='--')
        ax.set_xlim(dom[0], dom[1])
        ax.set_ylim(-0.15, 1.5)
        ax.set_xlabel('Rest Wavelength (Å)')
        ax.set_ylabel('Normalised Flux')
        ax.set_title('LLS Break')
        ax.legend(fontsize=9)

        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=150, bbox_inches='tight')
            print(f"Fit plot saved to {outfile}")
        plt.show()
        return fig

    def plot_diagnostics(self, figsize=(10, 2.5), outfile=None):
        """
        Convergence diagnostics: trace plots + sampler-specific statistics.

        emcee: acceptance fraction and autocorrelation time.
        zeus:  Gelman-Rubin R-hat per parameter.

        Parameters
        ----------
        figsize : tuple
            Per-panel figure size (width, height); total height scales with ndim.
        outfile : str, optional
            Save trace figure to this path if provided.
        """
        if self.sampler is None:
            raise RuntimeError("Run fit() first")

        labels    = self._param_labels()
        chain     = self.sampler.get_chain()   # (nsteps, nwalkers, ndim)
        nsteps    = chain.shape[0]

        # --- Print convergence statistics ---
        print("\n=== Convergence Diagnostics ===")

        if self.sampler_name == 'emcee':
            acc = np.mean(self.sampler.acceptance_fraction)
            print(f"Mean acceptance fraction : {acc:.3f}  ", end='')
            if acc < 0.2:
                print("(too low — try more walkers or smaller perturbation)")
            elif acc > 0.5:
                print("(high — chain may be exploring too freely)")
            else:
                print("(good)")

            try:
                tau   = self.sampler.get_autocorr_time()
                ratio = (nsteps - self.burnin) / np.max(tau)
                print(f"Autocorrelation times    : { {l: round(t,1) for l,t in zip(labels, tau)} }")
                print(f"Chain length / max(tau)  : {ratio:.1f}  ", end='')
                if ratio < 50:
                    print(f"(too short — recommend >= {int(50*np.max(tau) + self.burnin)} total steps)")
                else:
                    print("(good)")
            except Exception as e:
                print(f"Autocorrelation time could not be estimated: {e}")

        elif self.sampler_name == 'zeus':
            try:
                import zeus as _zeus
                # chain post-burnin reshaped to (nwalkers, nsteps, ndim) for G-R
                chain_gr = chain[self.burnin:].transpose(1, 0, 2)
                rhat     = _zeus.diagnostics.gelman_rubin(chain_gr)
                print("Gelman-Rubin R-hat per parameter:")
                for label, r in zip(labels, rhat):
                    status = 'good' if r < 1.1 else 'NOT converged — run longer'
                    print(f"  {label:15s} : {r:.4f}  ({status})")
                print(f"Max R-hat : {np.max(rhat):.4f}  "
                      f"({'converged' if np.max(rhat) < 1.1 else 'NOT converged'})")
            except Exception as e:
                print(f"Gelman-Rubin diagnostic failed: {e}")

        # --- Trace plots ---
        fig, axes = plt.subplots(self.ndim, 1,
                                 figsize=(figsize[0], figsize[1] * self.ndim),
                                 sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(chain[:, :, i], color='steelblue', alpha=0.3, lw=0.5)
            ax.axvline(self.burnin, color='red', lw=1, ls='--',
                       label='burn-in end' if i == 0 else '')
            ax.set_ylabel(labels[i], fontsize=9)
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[0].legend(fontsize=8, loc='upper right')
        axes[-1].set_xlabel('Step')
        fig.suptitle('Walker Traces', fontsize=11)
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=150, bbox_inches='tight')
            print(f"Trace plot saved to {outfile}")
        plt.show()
        return fig

    def plot_corner(self, figsize=(8, 8), outfile=None):
        """
        Corner plot of posterior samples.

        Parameters
        ----------
        figsize : tuple
        outfile : str, optional
            Save figure to this path if provided.
        """
        if self.samples is None:
            raise RuntimeError("Run fit() first")

        fig = corner.corner(
            self.samples,
            labels=self._param_labels(),
            truths=self.best_theta,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt='.3f',
            title_kwargs={'fontsize': 9},
            label_kwargs={'fontsize': 9},
        )
        plt.tight_layout()
        if outfile:
            plt.savefig(outfile, dpi=150, bbox_inches='tight')
            print(f"Corner plot saved to {outfile}")
        plt.show()
        return fig
