#!/usr/bin/env python
"""CLI: Get dynamic PE (PE-TTM) for an A-share stock.

Examples:
  python scripts/get_dynamic_pe.py --code 600519.SH
  python scripts/get_dynamic_pe.py --code 600519.SH --start 20240101 --end 20241201

Notes:
- In many A-share workflows, "动态PE" commonly refers to rolling PE (PE-TTM).
- True forward PE requires earnings forecasts (often paid / incomplete).
"""

from __future__ import annotations

import argparse

from pathlib import Path
import sys

# Ensure project root is on sys.path so `app` can be imported when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.valuation import get_dynamic_pe_latest, get_dynamic_pe_series


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True, help="ts_code, e.g. 600519.SH")
    parser.add_argument("--start", default=None, help="YYYYMMDD")
    parser.add_argument("--end", default=None, help="YYYYMMDD")
    args = parser.parse_args()

    if args.start and args.end:
        df = get_dynamic_pe_series(args.code, args.start, args.end)
        if df.empty:
            print(f"No data for {args.code} in {args.start}-{args.end}")
            return 2
        latest = df.iloc[-1]
        print(f"{args.code} latest in range {latest['trade_date']}")
        print(f"  close: {latest['close']}")
        print(f"  pe (static): {latest['pe']}")
        print(f"  pe_ttm (dynamic): {latest['pe_ttm']}")
        return 0

    point = get_dynamic_pe_latest(args.code)
    print(f"{point.ts_code} latest {point.trade_date}")
    print(f"  close: {point.close}")
    print(f"  pe (static): {point.pe_static}")
    print(f"  pe_ttm (dynamic): {point.pe_ttm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
