import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from core.market import IST, expiry_at_ist, is_market_open, time_to_expiry
from core.pipeline import estimate_forward_parity, prepare_chain
from data import provider as provider_module
from data.provider import BrokerProvider, NSEProvider, get_chain

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize(
    ("fixture", "symbol", "spot"),
    [
        ("nse_index_response.json", "NIFTY", 24125.55),
        ("nse_stock_response.json", "RELIANCE", 1450.0),
    ],
)
def test_nse_response_parser(fixture, symbol, spot):
    payload = load_fixture(fixture)
    parsed = NSEProvider.parse_response(payload, symbol)
    assert len(parsed) == 18
    assert set(parsed.option_type) == {"call", "put"}
    assert parsed.symbol.eq(symbol).all()
    assert parsed.spot.eq(spot).all()
    assert parsed.expiry.iloc[0].tzinfo is not None
    assert parsed.expiry.iloc[0].hour == 15 and parsed.expiry.iloc[0].minute == 30
    assert parsed.exchange_iv.between(0.1, 1.0).all()


def test_time_to_expiry_just_before_and_after_nse_close():
    expiry = expiry_at_ist("15-Dec-2026")
    assert expiry_at_ist("2026-12-15 15:30:00+05:30") == expiry
    before = expiry - timedelta(seconds=1)
    after = expiry + timedelta(seconds=1)
    assert time_to_expiry(expiry, before) == pytest.approx(1 / (365 * 86400))
    assert time_to_expiry(expiry, after) == 0
    assert time_to_expiry(expiry, before, day_count_basis=360) == pytest.approx(1 / (360 * 86400))
    assert expiry.hour == 15 and expiry.minute == 30 and expiry.tzinfo == IST


def test_market_open_hours_weekdays_and_holidays():
    assert is_market_open(datetime(2026, 11, 9, 10, 0, tzinfo=IST))
    assert not is_market_open(datetime(2026, 11, 9, 9, 14, tzinfo=IST))
    assert not is_market_open(datetime(2026, 11, 9, 15, 30, tzinfo=IST))
    assert not is_market_open(datetime(2026, 11, 10, 10, 0, tzinfo=IST))
    assert not is_market_open(datetime(2026, 11, 8, 10, 0, tzinfo=IST))
    assert not is_market_open(
        datetime(2026, 11, 9, 10, 0, tzinfo=IST), holidays={date(2026, 11, 9)}
    )


@pytest.mark.parametrize(
    "fixture,symbol",
    [("nse_index_response.json", "NIFTY"), ("nse_stock_response.json", "RELIANCE")],
)
def test_parity_forward_regression_from_fixture(fixture, symbol):
    raw = NSEProvider.parse_response(load_fixture(fixture), symbol)
    expiry_group = raw.groupby("expiry").get_group(raw.expiry.iloc[0])
    forward, discount, matched = estimate_forward_parity(expiry_group, 0.065)
    assert matched >= 3
    assert discount == pytest.approx(0.995, abs=0.015)
    assert forward == pytest.approx(float(raw.spot.iloc[0]) * 1.005, rel=0.015)


def test_symbol_specific_liquidity_filters_are_configurable():
    raw = NSEProvider.parse_response(load_fixture("nse_stock_response.json"), "RELIANCE")
    raw.loc[raw.option_type.eq("put"), "volume"] = 0
    _points, quality, _forwards = prepare_chain(
        raw,
        0.065,
        now=datetime(2026, 11, 6, 12, 0, tzinfo=IST),
        filters={"min_volume": 10, "max_spread_pct": 0.01},
    )
    assert quality["filters"]["min_volume"] == 10
    assert quality["reasons"]["low volume"] == 9
    assert quality["kept"] < len(raw)


def test_fyers_adapter_normalizes_option_chain_response(monkeypatch):
    monkeypatch.setattr(provider_module.time, "sleep", lambda _: None)

    class Client:
        def __init__(self):
            self.calls = []

        def optionchain(self, *, data):
            self.calls.append(data.copy())
            if data["timestamp"] == "":
                payload = {
                    "s": "ok",
                    "data": {
                        "expiryData": [{"date": "15-12-2026", "expiry": "1797343200"}],
                        "indiavixData": {"ltp": 16.4},
                    },
                }
            else:
                payload = {
                    "s": "ok",
                    "data": {
                        "optionsChain": [
                            {"strike_price": -1, "option_type": "", "ltp": 24125.55},
                            {
                                "strike_price": 24000,
                                "option_type": "CE",
                                "bid": 120,
                                "ask": 122,
                                "ltp": 121,
                                "volume": 100,
                                "oi": 500,
                                "greeks": {"iv": 22.1},
                            },
                            {
                                "strike_price": 24000,
                                "option_type": "PE",
                                "bid": 80,
                                "ask": 82,
                                "ltp": 81,
                                "volume": 90,
                                "oi": 450,
                                "greeks": {"iv": 23.2},
                            },
                        ]
                    },
                }
            return payload

    client = Client()
    adapter = BrokerProvider(client=client)
    chain = adapter.get_chain("NIFTY")
    assert len(client.calls) == 2
    assert client.calls[0]["symbol"] == "NSE:NIFTY50-INDEX"
    assert client.calls[1]["timestamp"] == "1797343200"
    assert set(chain.option_type) == {"call", "put"}
    assert chain.lot_size.eq(65).all()
    assert chain.exchange_iv.between(0.2, 0.3).all()
    assert chain.attrs["india_vix"] == 16.4


def test_auto_provider_order_ends_at_synthetic_and_records_actual_source(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_module, "CACHE_DIR", tmp_path)
    visited = []

    class FailingProvider:
        def __init__(self, name):
            self.name = name

        def get_chain(self, symbol):
            visited.append(self.name)
            raise RuntimeError(f"{self.name} unavailable")

    class Synthetic:
        def get_chain(self, symbol):
            visited.append("synthetic")
            frame = NSEProvider.parse_response(load_fixture("nse_index_response.json"), "NIFTY")
            frame["source"] = "synthetic"
            return frame

    frame, actual = get_chain(
        "NIFTY",
        source="auto",
        broker=FailingProvider("broker"),
        nse=FailingProvider("nse"),
        synthetic=Synthetic(),
    )
    assert visited == ["broker", "nse", "synthetic"]
    assert actual == "synthetic"
    assert frame.source.eq("synthetic").all()


def test_recent_broker_cache_is_reused_unless_refresh_is_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(provider_module, "CACHE_DIR", tmp_path)
    calls = []

    class Broker:
        def get_chain(self, symbol):
            calls.append("broker")
            frame = NSEProvider.parse_response(load_fixture("nse_index_response.json"), symbol)
            frame["source"] = "broker"
            frame["fetched_at"] = provider_module.now_ist()
            return frame

    class UnusedProvider:
        def get_chain(self, symbol):
            calls.append("unexpected fallback")
            raise AssertionError("fallback should not run")

    _first, first_source = get_chain(
        "NIFTY", source="broker", broker=Broker(), nse=UnusedProvider()
    )
    cached, cached_source = get_chain(
        "NIFTY", source="auto", broker=UnusedProvider(), nse=UnusedProvider()
    )
    assert first_source == "broker" and cached_source == "cache"
    assert cached.attrs["cache_origin"] == "broker"
    assert calls == ["broker"]

    _refreshed, refreshed_source = get_chain(
        "NIFTY", source="broker", refresh=True, broker=Broker(), nse=UnusedProvider()
    )
    assert refreshed_source == "broker"
    assert calls == ["broker", "broker"]
