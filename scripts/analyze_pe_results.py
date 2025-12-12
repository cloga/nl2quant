"""对批量计算结果进行分析汇总并打印要点。

Usage:
  .\.venv\Scripts\python.exe scripts\analyze_pe_results.py --file data/pe_ratios_YYYYMMDD.csv
"""
import argparse
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="CSV结果文件路径")
    args = p.parse_args()

    df = pd.read_csv(args.file)
    if df.empty:
        print("文件为空")
        return 1

    print(f"样本数: {len(df)}")

    # 基本统计
    for col in ["static_pe", "ttm_pe", "linear_pe", "forecast_pe_mean", "forecast_pe_median"]:
        s = df[col].dropna()
        if s.empty:
            continue
        print(f"\n[{col}] 统计:")
        print(f"  均值: {s.mean():.2f}")
        print(f"  中位数: {s.median():.2f}")
        print(f"  分位数(10/50/90): {s.quantile(0.1):.2f} / {s.quantile(0.5):.2f} / {s.quantile(0.9):.2f}")

    # 情景分布
    if "scenario" in df.columns:
        print("\n[情景分布]:")
        print(df["scenario"].value_counts(dropna=False))

    # 线性外推与TTM差异最大的公司
    if "linear_vs_ttm_pct" in df.columns:
        s = df[["ts_code", "linear_vs_ttm_pct"]].dropna()
        if not s.empty:
            print("\n[线性外推低于TTM最多 Top 10]:")
            print(s.sort_values("linear_vs_ttm_pct").head(10))
            print("\n[线性外推高于TTM最多 Top 10]:")
            print(s.sort_values("linear_vs_ttm_pct", ascending=False).head(10))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
