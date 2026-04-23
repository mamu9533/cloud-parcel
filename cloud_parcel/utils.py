import numpy as np
import math
 
 
def calculate_transmittance(model_output, photon_no):
    """
    Compute transmittance from Monte Carlo photon output.
 
    Parameters
    ----------
    model_output : np.ndarray
        Photon state array where 1 = transmitted, -1 = reflected, 0 = absorbed.
    photon_no : int
        Total number of photons simulated.
 
    Returns
    -------
    float
        Fraction of photons transmitted through the cloud.
    """
    return np.sum(np.where(model_output == 1, 1, 0)) / photon_no
 
 
def calculate_reflectance(model_output, photon_no):
    """
    Compute reflectance from Monte Carlo photon output.
 
    Parameters
    ----------
    model_output : np.ndarray
        Photon state array where 1 = transmitted, -1 = reflected, 0 = absorbed.
    photon_no : int
        Total number of photons simulated.
 
    Returns
    -------
    float
        Fraction of photons reflected by the cloud.
    """
    return np.sum(np.where(model_output == -1, 1, 0)) / photon_no
 
 
def calculate_absorbance(model_output, photon_no):
    """
    Compute absorbance from Monte Carlo photon output.
 
    Parameters
    ----------
    model_output : np.ndarray
        Photon state array where 1 = transmitted, -1 = reflected, 0 = absorbed.
    photon_no : int
        Total number of photons simulated.
 
    Returns
    -------
    float
        Fraction of photons absorbed by the cloud.
    """
    return np.sum(np.isnan(model_output)) / photon_no
 
 
def compute_effective_radius(radii, Nis):
    """
    Compute the height-resolved effective radius of cloud droplets.
 
    Uses the Hansen & Travis (1974) definition: the ratio of the third
    to second moment of the droplet size distribution.
 
    Parameters
    ----------
    radii : np.ndarray
        2D array of droplet radii (n_heights x n_bins), m.
    Nis : np.ndarray
        2D array of number concentrations (n_heights x n_bins), m^-3.
 
    Returns
    -------
    np.ndarray
        1D array of effective radii at each height, m.
    """
    third_moment = np.sum((radii ** 3) * Nis, axis=1)
    second_moment = np.sum((radii ** 2) * Nis, axis=1)
    return third_moment / second_moment
 
 
def compute_volume(radii, Nis):
    """
    Compute the height-resolved total droplet volume concentration.
 
    Parameters
    ----------
    radii : np.ndarray
        2D array of droplet radii (n_heights x n_bins), m.
    Nis : np.ndarray
        2D array of number concentrations (n_heights x n_bins), m^-3.
 
    Returns
    -------
    np.ndarray
        1D array of total droplet volume concentration at each height, m^3/m^3.
    """
    volume = (4/3) * math.pi * np.array(radii)**3
    return np.sum(volume * Nis, axis=1)
 
 
def compute_tau(lwc, r_eff, z):
    """
    Compute height-resolved layer optical depth using the geometric optics
    approximation for liquid water clouds.
 
    Applies the relation tau = 1.5 * (LWC / r_eff) * dZ, where the factor
    of 1.5 follows from Mie theory in the geometric optics limit and assumes
    a cloud droplet extinction efficiency of 2.
 
    Parameters
    ----------
    lwc : np.ndarray
        1D array of liquid water volume concentration at each height, m^3/m^3.
    r_eff : np.ndarray
        1D array of effective radii at each height, m.
    z : array_like
        1D array of heights, m.
 
    Returns
    -------
    np.ndarray
        1D array of layer optical depths. The first element is always 0
        since dZ is undefined at the base of the column.
    """
    dZ = np.diff(z)
    dZ = np.insert(dZ, 0, 0)
 
    return 1.5 * (lwc / r_eff) * dZ