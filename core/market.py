"""Indian market clock and NSE expiry conventions."""

from __future__ import annotations

import csv
import os
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DAY_COUNT_BASIS = float(os.getenv("DAY_COUNT_BASIS", "365"))
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


def now_ist() -> datetime:
    """Return the current timezone-aware Indian Standard Time."""
    return datetime.now(IST)


def as_ist(value: datetime) -> datetime:
    """Convert an aware datetime to IST; interpret naive values as IST."""
    return value.replace(tzinfo=IST) if value.tzinfo is None else value.astimezone(IST)


def expiry_at_ist(value: str | date | datetime) -> datetime:
    """Resolve a chain expiry date to the NSE option close at 15:30 IST."""
    if isinstance(value, datetime):
        local = as_ist(value)
        return datetime.combine(local.date(), MARKET_CLOSE, tzinfo=IST)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if len(value) > 10:
                return datetime.combine(as_ist(parsed).date(), MARKET_CLOSE, tzinfo=IST)
        except ValueError:
            pass
        for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                d = datetime.strptime(value, fmt).date()
                return datetime.combine(d, MARKET_CLOSE, tzinfo=IST)
            except ValueError:
                continue
        raise ValueError(f"Unsupported NSE expiry date: {value}")
    return datetime.combine(value, MARKET_CLOSE, tzinfo=IST)


def time_to_expiry(
    expiry: str | date | datetime,
    now: datetime | None = None,
    *,
    day_count_basis: float = DAY_COUNT_BASIS,
) -> float:
    """ACT/day_count_basis year fraction to expiry 15:30 IST; zero after expiry."""
    if day_count_basis <= 0:
        raise ValueError("day_count_basis must be positive")
    current = as_ist(now or datetime.now(IST))
    end = expiry_at_ist(expiry)
    return max((end - current).total_seconds() / (day_count_basis * 24 * 60 * 60), 0.0)


def load_nse_holidays(path: str | Path | None = None) -> set[date]:
    """Read the maintained NSE derivatives holiday CSV bundled with the project."""
    csv_path = (
        Path(path) if path else Path(__file__).resolve().parents[1] / "data" / "nse_holidays.csv"
    )
    if not csv_path.exists():
        return set()
    with csv_path.open(newline="", encoding="utf-8") as stream:
        holidays = {
            date.fromisoformat(row["date"]) for row in csv.DictReader(stream) if row.get("date")
        }
    for raw_date in os.getenv("NSE_HOLIDAYS_EXTRA", "").split(","):
        if raw_date.strip():
            try:
                holidays.add(date.fromisoformat(raw_date.strip()))
            except ValueError:
                continue
    return holidays


def is_market_open(now: datetime | None = None, holidays: set[date] | None = None) -> bool:
    """Return true inside NSE weekday trading hours, excluding supplied exchange holidays."""
    current = as_ist(now or datetime.now(IST))
    closed_days = load_nse_holidays() if holidays is None else holidays
    return (
        current.weekday() < 5
        and current.date() not in closed_days
        and MARKET_OPEN <= current.time().replace(tzinfo=None) < MARKET_CLOSE
    )


def expiry_kind(expiry: str | date | datetime, expiries: list[str | date | datetime]) -> str:
    """Label an expiry as monthly when it is the last listed expiry in its calendar month."""
    target = expiry_at_ist(expiry).date()
    same_month = [
        expiry_at_ist(item).date()
        for item in expiries
        if expiry_at_ist(item).date().year == target.year
        and expiry_at_ist(item).date().month == target.month
    ]
    return "Monthly" if target == max(same_month) else "Weekly"
