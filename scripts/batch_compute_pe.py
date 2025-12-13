"""批量计算A股全部股票的多维度市盈率并导出CSV。

Usage:
  .\.venv\Scripts\python.exe scripts\batch_compute_pe.py --limit 50
  .\.venv\Scripts\python.exe scripts\batch_compute_pe.py --from-cache
  .\.venv\Scripts\python.exe scripts\batch_compute_pe.py --force-update
  .\.venv\Scripts\python.exe scripts\batch_compute_pe.py --no-skip-same-day

说明：
- 默认导出到 data/pe_ratios_YYYYMMDD.csv
- 支持 --limit 做抽样测试
- 支持 --from-cache 从缓存加载（快速）
- 支持 --force-update 强制全量更新
- 默认启用同天快速模式（同一天内跳过重复计算）
- 支持 --no-skip-same-day 禁用同天快速模式
- 自动跳过异常个股并记录错误
- 使用批量 API 优化性能
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

# Ensure project root on sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.pe_cache import PECache, batch_compute_and_cache


def _get_pro():
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未设置 TUSHARE_TOKEN")
    ts.set_token(token)
    return ts.pro_api()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="", help="交易所: SSE/SZSE/BSE, 空为全部")
    parser.add_argument("--limit", type=int, default=0, help="只处理前N只股票用于测试")
    parser.add_argument("--outfile", default="", help="输出CSV路径")
    parser.add_argument("--from-cache", action="store_true", help="从缓存加载数据（快速）")
    parser.add_argument("--force-update", action="store_true", help="强制全量更新（忽略缓存）")
    parser.add_argument("--no-skip-same-day", dest="skip_same_day", action="store_false", 
                        help="禁用同天跳过，每次都重新计算；默认启用同天快速模式")
    args = parser.parse_args()
    
    # 默认启用同天快速模式
    if not hasattr(args, 'skip_same_day'):
        args.skip_same_day = True

    outdir = Path("data")
    outdir.mkdir(parents=True, exist_ok=True)
    if args.outfile:
        outfile = Path(args.outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
    else:
        outfile = outdir / f"pe_ratios_{datetime.now().strftime('%Y%m%d')}.csv"

    # 从缓存加载
    if args.from_cache:
        print("从缓存加载数据...", flush=True)
        cache = PECache()
        cache_data = cache.load_cache()
        if not cache_data:
            print("缓存为空，请先运行更新")
            return 1
        
        metadata = cache.get_metadata()
        print(f"缓存更新时间: {metadata.get('last_update', 'unknown')}")
        print(f"缓存记录数: {len(cache_data)}")
        
        rows = list(cache_data.values())
        if args.limit > 0:
            rows = rows[:args.limit]
        
        # 直接写出CSV
        fieldnames = [
            "ts_code","trade_date","close_price","market_cap",
            "static_pe","ttm_pe","linear_extrapolate_pe","latest_quarter",
            "forecast_pe_mean","forecast_pe_median"
        ]
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                row_filtered = {k: r.get(k) for k in fieldnames}
                writer.writerow(row_filtered)
        
        print(f"导出完成: {outfile}，共 {len(rows)} 条")
        return 0

    # 获取股票列表
    pro = _get_pro()
    print("仅计算上市状态（L）股票", flush=True)
    
    basics = pro.stock_basic(
        exchange=args.exchange,
        list_status='L',
        fields="ts_code,symbol,name,area,industry,market,is_hs,list_date"
    )
    
    if basics.empty:
        print("未获取到股票列表")
        return 1

    # 仅保留上市满5年（含5年）的企业
    now = datetime.now()
    cutoff_str = f"{now.year - 5}{now.strftime('%m%d')}"
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

    ts_codes = basics["ts_code"].tolist()
    
    # 使用缓存批量计算（启用同天快速模式）
    print(f"\n开始批量计算 {len(ts_codes)} 只股票的PE数据...")
    print(f"强制更新模式: {args.force_update}")
    print(f"同天快速模式: {args.skip_same_day}")
    
    cache_data = batch_compute_and_cache(
        ts_codes=ts_codes,
        force_update=args.force_update,
        use_batch_daily=True,
        progress_callback=None,
        skip_same_day=args.skip_same_day
    )
    
    # 合并基础信息
    rows = []
    errors = []
    
    for _, row in basics.iterrows():
        ts_code = row["ts_code"]
        if ts_code not in cache_data:
            errors.append((ts_code, "计算失败或跳过"))
            continue
        
        res_dict = cache_data[ts_code]
        
        # 情景标注
        scenario = ""
        if res_dict.get("static_pe") and res_dict.get("ttm_pe") and res_dict.get("forecast_pe_median"):
            if res_dict["static_pe"] > res_dict["ttm_pe"] > res_dict["forecast_pe_median"]:
                scenario = "growth_accelerating"
            elif res_dict["static_pe"] < res_dict["ttm_pe"] < res_dict["forecast_pe_median"]:
                scenario = "decline_worsening"
        
        linear_vs_ttm_pct = None
        if res_dict.get("linear_extrapolate_pe") and res_dict.get("ttm_pe"):
            linear_vs_ttm_pct = (res_dict["linear_extrapolate_pe"] / res_dict["ttm_pe"] - 1.0) * 100.0
        
        rows.append({
            "ts_code": res_dict["ts_code"],
            "symbol": row.get("symbol"),
            "name": row.get("name"),
            "area": row.get("area"),
            "industry": row.get("industry"),
            "market": row.get("market"),
            "is_hs": row.get("is_hs"),
            "list_date": row.get("list_date"),
            "trade_date": res_dict["trade_date"],
            "close": res_dict["close_price"],
            "market_cap": res_dict["market_cap"],
            "static_pe": res_dict["static_pe"],
            "ttm_pe": res_dict["ttm_pe"],
            "linear_pe": res_dict["linear_extrapolate_pe"],
            "latest_quarter": res_dict["latest_quarter"],
            "forecast_pe_mean": res_dict["forecast_pe_mean"],
            "forecast_pe_median": res_dict["forecast_pe_median"],
            "scenario": scenario,
            "linear_vs_ttm_pct": linear_vs_ttm_pct,
        })

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
