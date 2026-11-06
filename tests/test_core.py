import numpy as np
import pytest

from core.pricing import implied_volatility, price
from core.svi import SVIParams, arbitrage_checks, fit_svi


def test_put_call_parity():
    s, k, t, r, q, v = 100.0, 105.0, 0.8, 0.04, 0.01, 0.27
    c = price(s, k, t, r, v, "call", q)
    p = price(s, k, t, r, v, "put", q)
    assert c - p == pytest.approx(s * np.exp(-q * t) - k * np.exp(-r * t), abs=1e-10)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_iv_roundtrip(kind):
    market = price(100, 105, 0.75, 0.03, 0.31, kind)
    assert implied_volatility(market, 100, 105, 0.75, 0.03, kind) == pytest.approx(0.31, abs=1e-7)


def test_iv_rejects_out_of_bounds():
    with pytest.raises(ValueError, match="bounds"):
        implied_volatility(200, 100, 100, 1, 0.03, "call")


def test_svi_recovers_known_parameters():
    expected = SVIParams(0.025, 0.12, -0.3, 0.03, 0.2)
    k = np.linspace(-0.5, 0.5, 101)
    observed = expected.variance(k)
    result = fit_svi(k, observed)
    assert np.sqrt(np.mean((result.variance(k) - observed) ** 2)) < 1e-4


def test_arbitrage_checker_detects_calendar_violation():
    params = [SVIParams(0.08, 0.01, 0, 0, 0.2), SVIParams(0.02, 0.01, 0, 0, 0.2)]
    result = arbitrage_checks(params, [0.5, 1.0])
    assert not result["calendar_ok"]
    assert result["calendar_violations"]
