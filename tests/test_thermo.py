import numpy as np
from cloud_parcel.thermo import (
    saturation_vapor_pressure,
    mixing_ratio_from_rh,
    specific_humidity_from_rh,
)


def test_saturation_vapor_pressure_freezing_point():
    """Saturation vapor pressure at 0°C should be ~611 Pa."""
    es = saturation_vapor_pressure(273.15)
    assert np.isclose(es, 611, rtol=0.05)


def test_saturation_vapor_pressure_monotonic():
    """Saturation vapor pressure should increase with temperature."""
    es_cold = saturation_vapor_pressure(260)
    es_warm = saturation_vapor_pressure(300)
    assert es_warm > es_cold


def test_mixing_ratio_physical_bounds():
    """Mixing ratio should be positive and increase with RH."""
    T = 290
    p = 90000  # Pa

    w_low = mixing_ratio_from_rh(T, 0.2, p)
    w_high = mixing_ratio_from_rh(T, 0.8, p)

    assert w_low > 0
    assert w_high > w_low


def test_specific_humidity_consistency():
    """Specific humidity should be w/(1+w) and < 1."""
    T = 290
    RH = 0.5
    p = 85000

    w = mixing_ratio_from_rh(T, RH, p)
    q = specific_humidity_from_rh(T, RH, p)

    assert np.isclose(q, w / (1 + w))
    assert 0 < q < 1
