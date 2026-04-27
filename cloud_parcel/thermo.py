"""
thermo.py
---------
Thermodynamic helper functions for cloud parcel modeling.

These functions provide basic humidity and saturation calculations that are
commonly needed when preparing input profiles for pyrcel-based microphysical
models. They are written to be lightweight, dependency-free, and easy to test.

Functions included:
- saturation_vapor_pressure(T)
- mixing_ratio_from_rh(T, RH, p)
- specific_humidity_from_rh(T, RH, p)

All temperatures are in Kelvin, pressure in Pascals, and relative humidity
expressed as a fraction (0–1).
"""

import numpy as np


def saturation_vapor_pressure(T):
    """
    Compute saturation vapor pressure (Pa) using the Bolton (1980) formula.

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.

    Returns
    -------
    es : float or array-like
        Saturation vapor pressure in Pascals.
    """
    T_C = T - 273.15  # convert to Celsius
    es = 611.2 * np.exp((17.67 * T_C) / (T_C + 243.5))
    return es


def mixing_ratio_from_rh(T, RH, p):
    """
    Compute water vapor mixing ratio (kg/kg) from temperature, RH, and pressure.

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.
    RH : float or array-like
        Relative humidity (0–1).
    p : float or array-like
        Ambient pressure in Pascals.

    Returns
    -------
    w : float or array-like
        Water vapor mixing ratio (kg/kg).
    """
    es = saturation_vapor_pressure(T)
    e = RH * es  # actual vapor pressure
    epsilon = 0.622  # Rd/Rv
    w = epsilon * e / (p - e)
    return w


def specific_humidity_from_rh(T, RH, p):
    """
    Compute specific humidity (kg/kg) from temperature, RH, and pressure.

    Parameters
    ----------
    T : float or array-like
        Temperature in Kelvin.
    RH : float or array-like
        Relative humidity (0–1).
    p : float or array-like
        Ambient pressure in Pascals.

    Returns
    -------
    q : float or array-like
        Specific humidity (kg/kg).
    """
    w = mixing_ratio_from_rh(T, RH, p)
    q = w / (1 + w)
    return q
