"""Normalized Indian F&O option-chain providers with cache and deterministic fallback."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from core.market import IST, expiry_at_ist, time_to_expiry
from core.pricing import price
from core.svi import SVIParams

logger = logging.getLogger(__name__)
load_dotenv(override=False)
CACHE_DIR = Path(os.getenv("CHAIN_CACHE_DIR", ".cache/chains"))
NSE_BASE = "https://www.nseindia.com"
DEFAULT_INDIA_RISK_FREE_RATE = float(os.getenv("INDIA_RISK_FREE_RATE", "0.065"))
MAX_CHAIN_AGE_SECONDS = int(os.getenv("MAX_CHAIN_AGE_SECONDS", "300"))
CACHE_TTL_SECONDS = int(os.getenv("CHAIN_CACHE_TTL_SECONDS", "30"))
DEFAULT_LOT_SIZES = {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120}
FYERS_DEFAULT_SYMBOLS = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
    "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
}
NORMALIZED_COLUMNS = [
    "symbol",
    "spot",
    "expiry",
    "T",
    "strike",
    "option_type",
    "bid",
    "ask",
    "ltp",
    "volume",
    "open_interest",
    "exchange_iv",
    "lot_size",
    "fetched_at",
    "source",
]


def now_ist():
    return datetime.now(IST)


def _lot_size(symbol, payload=None):
    if payload is not None:
        value = payload.get("lotSize") or payload.get("marketLot") or payload.get("lot_size")
        if value:
            return int(value)
    env = os.getenv("NSE_LOT_SIZES", "")
    try:
        mapping = json.loads(env) if env else {}
        value = mapping.get(symbol)
        if value:
            return int(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Ignoring malformed NSE_LOT_SIZES configuration")
    return DEFAULT_LOT_SIZES.get(symbol)


def normalize_frame(rows, symbol, spot, lot_size=None, fetched_at=None, source="unknown"):
    """Normalize vendor rows and derive expiry datetime/T in Asia/Kolkata."""
    fetched_at = fetched_at or now_ist()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    aliases = {
        "ticker": "symbol",
        "kind": "option_type",
        "lastPrice": "ltp",
        "openInterest": "open_interest",
        "impliedVolatility_vendor": "exchange_iv",
        "implied_volatility": "exchange_iv",
    }
    df = df.rename(
        columns={k: v for k, v in aliases.items() if k in df.columns and v not in df.columns}
    )
    if "symbol" not in df:
        df["symbol"] = symbol
    if "spot" not in df:
        df["spot"] = spot
    if "option_type" not in df:
        df["option_type"] = "call"
    if "ltp" not in df:
        df["ltp"] = 0.0
    if "volume" not in df:
        df["volume"] = 0
    if "open_interest" not in df:
        df["open_interest"] = 0
    if "exchange_iv" not in df:
        df["exchange_iv"] = np.nan
    if "lot_size" not in df:
        df["lot_size"] = lot_size
    else:
        df["lot_size"] = df["lot_size"].fillna(lot_size)
    if "fetched_at" not in df:
        df["fetched_at"] = fetched_at
    df["expiry"] = [expiry_at_ist(x) for x in df["expiry"]]
    df["T"] = [time_to_expiry(x, now_ist()) for x in df["expiry"]]
    df["exchange_iv"] = pd.to_numeric(df["exchange_iv"], errors="coerce")
    df.loc[df["exchange_iv"] > 1, "exchange_iv"] = (
        df.loc[df["exchange_iv"] > 1, "exchange_iv"] / 100
    )
    df["fetched_at"] = [
        (
            pd.Timestamp(x).to_pydatetime().replace(tzinfo=IST)
            if pd.Timestamp(x).tzinfo is None
            else pd.Timestamp(x).tz_convert(IST).to_pydatetime()
        )
        for x in df["fetched_at"]
    ]
    df["source"] = source
    for col in NORMALIZED_COLUMNS:
        if col not in df:
            df[col] = None
    return df[NORMALIZED_COLUMNS].copy()


class ChainProvider(ABC):
    @abstractmethod
    def get_chain(self, symbol: str) -> pd.DataFrame: ...


class NSEProvider(ChainProvider):
    """Unofficial NSE web JSON adapter with cookie bootstrap, retries and throttling."""

    _last_request = 0.0
    _lock = threading.Lock()

    def __init__(self, *, session=None, timeout=12, min_interval=1.2, max_attempts=3):
        self.session = session or requests.Session()
        self.timeout = timeout
        self.min_interval = min_interval
        self.max_attempts = max_attempts
        self._bootstrapped = False
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": f"{NSE_BASE}/option-chain",
            "Connection": "keep-alive",
        }

    def _get_json(self, url, params=None):
        last = None
        for attempt in range(self.max_attempts):
            try:
                if not self._bootstrapped:
                    self.session.get(NSE_BASE + "/", headers=self.headers, timeout=self.timeout)
                    self._bootstrapped = True
                with self._lock:
                    wait = self.min_interval - (time.monotonic() - type(self)._last_request)
                    if wait > 0:
                        time.sleep(wait)
                    response = self.session.get(
                        url, params=params, headers=self.headers, timeout=self.timeout
                    )
                    type(self)._last_request = time.monotonic()
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(min(2**attempt + random.random() * 0.25, 5))
        raise RuntimeError(
            f"NSE request failed after {self.max_attempts} attempts: {last}"
        ) from last

    @staticmethod
    def parse_response(payload, symbol, fetched_at=None):
        """Parse NSE records.data / filtered.data raw option-chain response."""
        records = payload.get("records", payload)
        data = (
            records.get("data")
            or (payload.get("filtered", {}).get("data") if payload.get("filtered") else [])
            or []
        )
        spot = float(records.get("underlyingValue") or payload.get("underlyingValue") or 0)
        if not data or spot <= 0:
            raise ValueError("NSE response has no records or underlying value")
        rows = []
        lot = _lot_size(symbol, records)
        fetched = records.get("timestamp")
        if fetched:
            try:
                fetched_at = datetime.strptime(fetched, "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
            except ValueError:
                fetched_at = now_ist()
        else:
            fetched_at = now_ist()
        for item in data:
            expiry = item.get("expiryDate") or item.get("expiry")
            strike = float(item.get("strikePrice") or item.get("strike") or 0)
            for code, kind in (("CE", "call"), ("PE", "put")):
                leg = item.get(code) or item.get(kind.upper())
                if not leg:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "spot": float(leg.get("underlyingValue") or spot),
                        "expiry": expiry,
                        "strike": strike,
                        "option_type": kind,
                        "bid": float(leg.get("bidprice", leg.get("bidPrice", 0)) or 0),
                        "ask": float(leg.get("askPrice", leg.get("askprice", 0)) or 0),
                        "ltp": float(leg.get("lastPrice", 0) or 0),
                        "volume": float(leg.get("totalTradedVolume", leg.get("volume", 0)) or 0),
                        "open_interest": float(leg.get("openInterest", 0) or 0),
                        "exchange_iv": float(leg.get("impliedVolatility", 0) or 0),
                        "lot_size": lot,
                    }
                )
        return normalize_frame(rows, symbol, spot, lot, fetched_at, "nse")

    def get_chain(self, symbol):
        symbol = symbol.upper().strip()
        indices = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI"}
        endpoint = "option-chain-indices" if symbol in indices else "option-chain-equities"
        payload = self._get_json(f"{NSE_BASE}/api/{endpoint}", params={"symbol": symbol})
        return self.parse_response(payload, symbol)

    def get_india_vix(self):
        """Return current India VIX if NSE's all-indices response is accessible."""
        payload = self._get_json(f"{NSE_BASE}/api/allIndices")
        for row in payload.get("data", []):
            name = str(row.get("index") or row.get("indexSymbol") or "").upper()
            if name in {"INDIA VIX", "INDIA VIX INDEX"}:
                return (
                    float(row.get("last") or row.get("lastPrice") or row.get("indexValue")),
                    now_ist(),
                )
        return None, None


class BrokerProvider(ChainProvider):
    """FYERS v3 option-chain adapter (free Standard-plan limits; daily access token)."""

    def __init__(self, *, client=None, symbol_map=None, strike_count=50):
        self.client = client
        self.client_id = os.getenv("FYERS_CLIENT_ID", "")
        self.access_token = os.getenv("FYERS_ACCESS_TOKEN", "")
        configured = json.loads(os.getenv("FYERS_SYMBOL_MAP", "{}") or "{}")
        self.symbol_map = {**FYERS_DEFAULT_SYMBOLS, **configured, **(symbol_map or {})}
        self.strike_count = min(max(int(strike_count), 1), 50)
        self.india_vix = None

    def _client(self):
        if self.client is not None:
            return self.client
        if not self.client_id or not self.access_token:
            raise RuntimeError("FYERS credentials are not configured")
        try:
            from fyers_apiv3 import fyersModel
        except ImportError as exc:
            raise RuntimeError("Install the fyers-apiv3 SDK to use the FYERS provider") from exc
        self.client = fyersModel.FyersModel(
            client_id=self.client_id, token=self.access_token, is_async=False, log_path=""
        )
        return self.client

    @staticmethod
    def _check(response):
        if (
            not isinstance(response, dict)
            or response.get("s") == "error"
            or not response.get("data")
        ):
            raise RuntimeError(f"FYERS option-chain request failed: {response}")
        return response["data"]

    def get_chain(self, symbol):
        symbol = symbol.upper()
        fyers_symbol = self.symbol_map.get(symbol, f"NSE:{symbol}-EQ")
        client = self._client()
        request = {
            "symbol": fyers_symbol,
            "strikecount": self.strike_count,
            "timestamp": "",
            "greeks": "1",
        }
        meta = self._check(client.optionchain(data=request))
        self.india_vix = (meta.get("indiavixData") or {}).get("ltp")
        expiries = meta.get("expiryData") or []
        if not expiries:
            raise RuntimeError(f"FYERS returned no listed expiries for {symbol}")
        rows = []
        spot = None
        fetched = now_ist()
        for index, expiry_info in enumerate(expiries):
            if index == 0:
                time.sleep(0.25)  # keep the free Standard plan below its documented 5 req/s cap
            else:
                time.sleep(0.25)
            expiry_response = client.optionchain(
                data={**request, "timestamp": str(expiry_info.get("expiry", ""))}
            )
            expiry_data = self._check(expiry_response)
            chain_date = expiry_info.get("date") or expiry_data.get("date")
            for item in expiry_data.get("optionsChain", []):
                kind = str(item.get("option_type") or "").upper()
                if kind not in {"CE", "PE"}:
                    if item.get("strike_price", -1) == -1 and item.get("ltp"):
                        spot = float(item["ltp"])
                    continue
                if spot is None and item.get("fp"):
                    spot = float(item["fp"])
                greeks = item.get("greeks") or {}
                rows.append(
                    {
                        "symbol": symbol,
                        "spot": spot or float(item.get("fp") or 0),
                        "expiry": chain_date,
                        "strike": float(item.get("strike_price") or 0),
                        "option_type": "call" if kind == "CE" else "put",
                        "bid": float(item.get("bid") or 0),
                        "ask": float(item.get("ask") or 0),
                        "ltp": float(item.get("ltp") or 0),
                        "volume": float(item.get("volume") or 0),
                        "open_interest": float(item.get("oi") or 0),
                        "exchange_iv": float(greeks.get("iv") or item.get("iv") or 0),
                        "lot_size": _lot_size(symbol),
                    }
                )
        if not rows or not spot:
            raise RuntimeError(f"FYERS returned no usable option quotes for {symbol}")
        result = normalize_frame(rows, symbol, spot, _lot_size(symbol), fetched, "broker")
        result.attrs["india_vix"] = self.india_vix
        return result


class SyntheticProvider(ChainProvider):
    """Offline NIFTY-scale chain with skewed smiles and illustrative expiries."""

    def get_chain(self, symbol="NIFTY"):
        symbol = symbol.upper()
        spots = {"NIFTY": 24000.0, "BANKNIFTY": 55000.0, "FINNIFTY": 26000.0, "MIDCPNIFTY": 13500.0}
        spot = spots.get(symbol, 2500.0)
        rate = DEFAULT_INDIA_RISK_FREE_RATE
        now = now_ist()
        expiry_days = [7, 14, 28, 35, 56]
        params = [
            SVIParams(0.012, 0.11, -0.48, 0, 0.12),
            SVIParams(0.022, 0.14, -0.48, 0, 0.14),
            SVIParams(0.04, 0.17, -0.48, 0, 0.17),
            SVIParams(0.05, 0.19, -0.48, 0, 0.19),
            SVIParams(0.075, 0.22, -0.48, 0, 0.22),
        ]
        rows = []
        step = 50 if spot > 5000 else 20
        for days, p in zip(expiry_days, params):
            d = (now + pd.Timedelta(days=days)).date()
            expiry = datetime.combine(
                d, datetime.min.time().replace(hour=15, minute=30), tzinfo=IST
            )
            T = max(time_to_expiry(expiry, now), 1e-4)
            F = spot * np.exp(rate * T)
            for k in np.linspace(-0.25, 0.25, 21):
                strike = round((F * np.exp(k)) / step) * step
                kk = float(np.log(strike / F))
                w = float(p.variance(kk))
                vol = np.sqrt(max(w / T, 1e-6))
                kind = "put" if kk < 0 else "call"
                mid = price(spot, strike, T, rate, vol, kind)
                spread = max(1.0, mid * 0.03)
                rows.append(
                    {
                        "symbol": symbol,
                        "spot": spot,
                        "expiry": expiry,
                        "strike": strike,
                        "option_type": kind,
                        "bid": max(mid - spread / 2, 0.05),
                        "ask": mid + spread / 2,
                        "ltp": mid,
                        "volume": 1000,
                        "open_interest": 15000,
                        "exchange_iv": np.nan,
                        "lot_size": _lot_size(symbol),
                        "fetched_at": now,
                    }
                )
        return normalize_frame(rows, symbol, spot, _lot_size(symbol), now, "synthetic")


def _cache_paths(symbol):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(symbol.upper().encode()).hexdigest()[:16]
    return CACHE_DIR / f"{key}.csv", CACHE_DIR / f"{key}.json"


def _write_cache(df, symbol, actual_source):
    csv_path, meta_path = _cache_paths(symbol)
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps({"source": actual_source, "cached_at": now_ist().isoformat()}))


def _read_cache(symbol):
    csv_path, meta_path = _cache_paths(symbol)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, parse_dates=["expiry", "fetched_at"])
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        df["source"] = "cache"
        return df, meta
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("Could not read cached chain for %s: %s", symbol, exc)
        return None


def get_chain(symbol, source="auto", refresh=False, *, broker=None, nse=None, synthetic=None):
    """Fetch in broker→NSE→cache→synthetic order for auto; return the actual source label."""
    symbol = symbol.upper().strip()
    broker = broker or BrokerProvider()
    nse = nse or NSEProvider()
    synthetic = synthetic or SyntheticProvider()
    if not refresh and source in {"auto", "broker", "nse"}:
        cached = _read_cache(symbol)
        if cached:
            cached_df, meta = cached
            origin = meta.get("source")
            fetched = pd.to_datetime(cached_df["fetched_at"], errors="coerce").max()
            if pd.notna(fetched):
                fetched = (
                    fetched.tz_localize(IST) if fetched.tzinfo is None else fetched.tz_convert(IST)
                )
                is_recent = (
                    now_ist() - fetched.to_pydatetime()
                ).total_seconds() <= CACHE_TTL_SECONDS
                source_matches = source == "auto" or source == origin
                if origin in {"broker", "nse"} and is_recent and source_matches:
                    cached_df["source"] = "cache"
                    cached_df.attrs["cache_origin"] = origin
                    return cached_df, "cache"
    if source == "synthetic":
        df = synthetic.get_chain(symbol)
        _write_cache(df, symbol, "synthetic")
        return df, "synthetic"
    if source not in {"auto", "broker", "nse"}:
        raise ValueError("source must be auto, broker, nse, or synthetic")
    providers = (
        [("broker", broker), ("nse", nse)]
        if source == "auto"
        else [(source, broker if source == "broker" else nse)]
    )
    errors = []
    for name, provider in providers:
        try:
            df = provider.get_chain(symbol)
            _write_cache(df, symbol, name)
            return df, name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            logger.warning("%s provider failed for %s: %s", name, symbol, exc)
    cached = _read_cache(symbol)
    if cached and source != "synthetic":
        df, meta = cached
        df["source"] = "cache"
        df.attrs["cache_origin"] = meta.get("source")
        return df, "cache"
    if source == "broker":
        raise RuntimeError("Broker and cache unavailable: " + "; ".join(errors))
    if source == "nse":
        raise RuntimeError("NSE and cache unavailable: " + "; ".join(errors))
    df = synthetic.get_chain(symbol)
    _write_cache(df, symbol, "synthetic")
    return df, "synthetic"


def cached_chain(ticker, provider, *, refresh=False):
    """Backward-friendly cache wrapper retained for early callers."""
    df = provider.get_chain(ticker) if refresh else None
    if df is None:
        cached = _read_cache(ticker)
        if cached:
            return cached[0], "cache"
        df = provider.get_chain(ticker)
    _write_cache(df, ticker, getattr(provider, "source", "unknown"))
    return df, "live"


def fetch_risk_free_rate(default=None):
    """Return configured Indian annual risk-free assumption; no US proxy is used."""
    value = DEFAULT_INDIA_RISK_FREE_RATE if default is None else float(default)
    return value, "configured Indian rate"
