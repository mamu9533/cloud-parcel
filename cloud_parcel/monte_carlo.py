import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from cloud_parcel.utils import calculate_absorbance, calculate_reflectance, calculate_transmittance

class monte_carlo:
    """
    Plane-parallel Monte Carlo radiative transfer model.

    Simulates photon transport through a cloud layer characterized by
    its optical depth, scattering asymmetry, and single scattering albedo.
    Supports both conservative scattering and absorbing atmospheres.

    Parameters
    ----------
    g : float, optional
        Henyey-Greenstein asymmetry parameter (default 0.85).
    omega : float, optional
        Single scattering albedo (default 0.99).
    N : int, optional
        Number of photons per model run (default 100).
    with_absorbance : bool, optional
        If True, enables absorption. If False, runs conservative
        scattering model (default False).

    Attributes
    ----------
    tau : float
        Cloud optical depth of the most recent run.
    output : np.ndarray
        Raw photon output array from the most recent single run.
        Photon states are encoded as:
            1   = transmitted
           -1   = reflected
           nan  = absorbed (only when with_absorbance=True)
    results : pd.DataFrame
        DataFrame of T/R/A from the most recent ensemble run.
    taus : array_like
        Optical depths used in the most recent ensemble run.
    """
    def __init__(self, g=0.85, omega=0.99, N=100, with_absorbance=False):

        self.g = g
        self.omega = omega
        self.N = N
        self.with_absorbance = with_absorbance
        self.tau = None
        self.output = None
        self.results = None
        self.taus = None

    def run(self, tau=None, return_values=False):
        """Run the Monte Carlo model for a single optical depth.

        Parameters
        ----------
        tau : float
            Cloud optical depth.
        return_values : bool, optional
            If True, returns the raw photon output array (default False).

        Returns
        -------
        np.ndarray or None
            Raw photon output array if return_values is True, else None.
            Photon states are encoded as:
                1   = transmitted
               -1   = reflected
               nan  = absorbed (only when with_absorbance=True)

        Raises
        ------
        Exception
            If tau is not defined.
        """
        self.tau = tau
        if self.tau is None:
            raise Exception("Tau not defined")
        N = self.N
        g = self.g
        omega = self.omega

        # Initialise to nan so absorbed photons require no explicit marking —
        # any photon that neither transmits nor reflects stays nan
        output = np.full(N, np.nan)
        for i in range(N):
            # Sample the initial free path length; if it exceeds tau the
            # photon passes straight through without scattering
            tau_bar = -np.log(1 - np.random.uniform(0, 1))
            if tau_bar > tau:
                output[i] = 1
            elif tau_bar < tau:
                sign = 1
                while True:
                    scatter = np.random.uniform(0, 1)
                    new_tau = -np.log(1 - np.random.uniform(0, 1))

                    # Test absorption at the current interaction point.
                    # If absorbed, the photon is dropped and stays nan.
                    if self.with_absorbance:
                        absorb = np.random.uniform(0, 1)
                        if absorb > omega:
                            break

                    # Check boundaries at current position
                    if tau_bar <= 0:
                        output[i] = -1
                        break
                    elif tau_bar >= tau:
                        output[i] = 1
                        break

                    # Move photon to next interaction point
                    if scatter <= (1 + g) / 2:
                        tau_bar = tau_bar + sign * new_tau
                        continue
                    else:
                        sign = -sign
                        tau_bar = tau_bar + sign * new_tau
                        continue

        self.output = output

        if return_values:
            return self.output

    def output_array(self):
        """Return the raw photon output array from the most recent run.

        Returns
        -------
        np.ndarray
            Raw photon output array.

        Raises
        ------
        Exception
            If the model has not been run yet.
        """
        if self.output is None:
            raise Exception("Model not run")
        return self.output

    def transmittance(self, tau=None):
        """Compute the transmittance of the cloud layer.

        Parameters
        ----------
        tau : float, optional
            Cloud optical depth. If not provided, uses self.tau.

        Returns
        -------
        float
            Fraction of photons transmitted through the cloud.

        Raises
        ------
        Exception
            If tau is not defined.
        """
        if tau is None:
            tau = self.tau
        if tau is None:
            raise Exception("Tau not defined")
        elif self.output is None:
            self.output = self.run(tau)

        return calculate_transmittance(self.output, self.N)

    def reflectance(self, tau=None):
        """Compute the reflectance of the cloud layer.

        Parameters
        ----------
        tau : float, optional
            Cloud optical depth. If not provided, uses self.tau.

        Returns
        -------
        float
            Fraction of photons reflected by the cloud.

        Raises
        ------
        Exception
            If tau is not defined.
        """
        if tau is None:
            tau = self.tau
        if tau is None:
            raise Exception("Tau not defined")
        elif self.output is None:
            self.output = self.run(tau)

        return calculate_reflectance(self.output, self.N)

    def absorbance(self, tau=None):
        """Compute the absorbance of the cloud layer.

        Parameters
        ----------
        tau : float, optional
            Cloud optical depth. If not provided, uses self.tau.

        Returns
        -------
        float
            Fraction of photons absorbed by the cloud.

        Raises
        ------
        Exception
            If model was initialized with with_absorbance=False.
            If tau is not defined.
        """
        if not self.with_absorbance:
            raise Exception("Model not initiated for absorbance")
        if tau is None:
            tau = self.tau
        if tau is None:
            raise Exception("Tau not defined")
        elif self.output is None:
            self.output = self.run(tau)

        return calculate_absorbance(self.output, self.N)

    def run_ensemble(self, taus, n_jobs=-1):
        """Run the Monte Carlo model over a range of optical depths in parallel.

        Parameters
        ----------
        taus : array_like
            Array of optical depth values to simulate.
        n_jobs : int, optional
            Number of parallel jobs (default -1, uses all available cores).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns for Transmittance, Reflectance, and
            optionally Absorbance, indexed by position in taus.
        """
        self.taus = taus
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.run)(tau, return_values=True) for tau in taus
        )
        results_dict = {"Transmittance": [calculate_transmittance(r, self.N) for r in results],
                        "Reflectance":   [calculate_reflectance(r, self.N) for r in results]}
        if self.with_absorbance:
            results_dict["Absorbance"] = [calculate_absorbance(r, self.N) for r in results]

        self.results = pd.DataFrame(results_dict)

        return self.results