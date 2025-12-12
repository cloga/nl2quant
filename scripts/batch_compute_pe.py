"""批量计算A股全部股票的多维度市盈率并导出CSV。

Usage:
  .\.venv\Scripts\python.exe scripts\batch_compute_pe.py --limit 50
  .\.venv\Scripts\python.exe scripts\batch_compute_pe.py --exchange SSE --status L --outfile data/pe_ratios_all.csv

说明：
- 默认导出到 data/pe_ratios_YYYYMMDD.csv
- 支持 --limit 做抽样测试
- 自动跳过异常个股并记录错误
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

# Ensure project root on sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.pe_ratios import calculate_all_pe_ratios


def _get_pro():
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未设置 TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api()


def _iter_ts_codes(pro, exchange: str, status: str) -> pd.DataFrame:
    df = pro.stock_basic(
        exchange=exchange,
        list_status=status,
        fields="ts_code,symbol,name,area,industry,market,is_hs,list_date"
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="", help="交易所: SSE/SZSE/BSE, 空为全部")
    # 仅计算上市状态股票（强制 L），参数保留但忽略
    parser.add_argument("--status", default="L", help="[已忽略] 始终仅计算上市状态 L")
    parser.add_argument("--limit", type=int, default=0, help="只处理前N只股票用于测试")
    parser.add_argument("--outfile", default="", help="输出CSV路径")
    args = parser.parse_args()

    outdir = Path("data")
    outdir.mkdir(parents=True, exist_ok=True)
    if args.outfile:
        outfile = Path(args.outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
    else:
        outfile = outdir / f"pe_ratios_{datetime.now().strftime('%Y%m%d')}.csv"

    pro = _get_pro()
    status = "L"
    print("仅计算上市状态（L）股票", flush=True)
    basics = _iter_ts_codes(pro, args.exchange, status)
    if basics.empty:
        print("未获取到股票列表")
        return 1

    # 仅保留上市满5年（含5年）的企业
    now = datetime.now()
    cutoff_str = f"{now.year - 5}{now.strftime('%m%d')}"  # e.g., 20201212
    total_before = len(basics)
    basics = basics.copy()
    basics["list_date"] = basics["list_date"].astype(str).fillna("")
    basics = basics[basics["list_date"].str.len() == 8]
    basics = basics[basics["list_date"] <= cutoff_str]
    print(f"仅上市≥5年: {len(basics)}/{total_before}")
    if basics.empty:
        print("未获取到股票列表")
        return 1

    if args.limit and args.limit > 0:
        basics = basics.head(args.limit)

    rows = []
    errors = []

    for idx, row in basics.iterrows():
        ts_code = row["ts_code"]
        try:
            res = calculate_all_pe_ratios(ts_code)
            # 情景标注
            scenario = ""
            if res.static_pe and res.ttm_pe and res.forecast_pe_median:
                if res.static_pe > res.ttm_pe > res.forecast_pe_median:
                    scenario = "growth_accelerating"
                elif res.static_pe < res.ttm_pe < res.forecast_pe_median:
                    scenario = "decline_worsening"
            rows.append({
                "ts_code": res.ts_code,
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "area": row.get("area"),
                "industry": row.get("industry"),
                "market": row.get("market"),
                "is_hs": row.get("is_hs"),
                "list_date": row.get("list_date"),
                "trade_date": res.trade_date,
                "close": res.close_price,
                "market_cap": res.market_cap,
                "static_pe": res.static_pe,
                "ttm_pe": res.ttm_pe,
                "linear_pe": res.linear_extrapolate_pe,
                "latest_quarter": res.latest_quarter,
                "forecast_pe_mean": res.forecast_pe_mean,
                "forecast_pe_median": res.forecast_pe_median,
                "scenario": scenario,
                "linear_vs_ttm_pct": None if not(res.linear_extrapolate_pe and res.ttm_pe) else (res.linear_extrapolate_pe / res.ttm_pe - 1.0) * 100.0,
            })
            # 小批量时每条都打印，常规模式每10条打印一次
            if len(basics) <= 20 or (idx + 1) % 10 == 0:
                print(f"进度: {idx+1}/{len(basics)}... 已完成 {ts_code}", flush=True)
        except Exception as e:
            errors.append((ts_code, str(e)))
            continue

    # 写出CSV
    fieldnames = [
        "ts_code","symbol","name","area","industry","market","is_hs","list_date",
        "trade_date","close","market_cap",
        "static_pe","ttm_pe","linear_pe","latest_quarter",
        "forecast_pe_mean","forecast_pe_median","scenario","linear_vs_ttm_pct"
    ]
    with open(outfile, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"导出完成: {outfile}，成功 {len(rows)} 条，失败 {len(errors)} 条")
    if errors:
        errfile = outfile.with_suffix(".errors.txt")
        with open(errfile, "w", encoding="utf-8") as f:
            for code, msg in errors:
                f.write(f"{code}: {msg}\n")
        print(f"错误明细: {errfile}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
