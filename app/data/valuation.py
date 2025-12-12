from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os

import pandas as pd
import tushare as ts
from dotenv import load_dotenv


@dataclass(frozen=True)
class DynamicPEPoint:
    """A single point of valuation data.

    Note: In Chinese market context, "动态PE" is often used colloquially to
    refer to PE-TTM (rolling) rather than a forward/forecast PE.
    This object returns both static PE and PE-TTM if available.
    """

    ts_code: str
    trade_date: str  # YYYYMMDD
    close: float | None
    pe_static: float | None
    pe_ttm: float | None


def _get_pro() -> ts.pro_api:
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not set in environment/.env")
    ts.set_token(token)
    return ts.pro_api()


def get_dynamic_pe_latest(ts_code: str) -> DynamicPEPoint:
    """Get latest dynamic PE for an A-share stock via Tushare daily_basic.

    Returns:
        DynamicPEPoint with pe_ttm as the primary 'dynamic' PE.
    """

    pro = _get_pro()
    df = pro.daily_basic(
        ts_code=ts_code,
        fields="ts_code,trade_date,close,pe,pe_ttm",
        limit=1,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No daily_basic data for {ts_code}")

    row = df.iloc[0]
    return DynamicPEPoint(
        ts_code=str(row.get("ts_code")),
        trade_date=str(row.get("trade_date")),
        close=_to_float_or_none(row.get("close")),
        pe_static=_to_float_or_none(row.get("pe")),
        pe_ttm=_to_float_or_none(row.get("pe_ttm")),
    )


def get_dynamic_pe_series(
    ts_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Get a time series of PE(Static) and PE-TTM between dates.

    Args:
        ts_code: e.g. '600519.SH'
        start_date/end_date: YYYYMMDD

    Returns:
        DataFrame sorted by trade_date ascending with columns:
        ['trade_date', 'close', 'pe', 'pe_ttm']
    """

    pro = _get_pro()
    df = pro.daily_basic(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields="ts_code,trade_date,close,pe,pe_ttm",
    )
    if df is None:
        return pd.DataFrame(columns=["trade_date", "close", "pe", "pe_ttm"])
    if df.empty:
        return pd.DataFrame(columns=["trade_date", "close", "pe", "pe_ttm"])

    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["trade_date"]).sort_values("trade_date")
    df["trade_date"] = df["trade_date"].dt.strftime("%Y%m%d")

    for col in ["close", "pe", "pe_ttm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["trade_date", "close", "pe", "pe_ttm"]]


def _to_float_or_none(value) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        if pd.isna(f):
            return None
        return f
    except Exception:
        return None
