"""查询A股股票的多维度市盈率指标。

Usage:
  python scripts/get_pe_ratios.py --code 600519.SH
  python scripts/get_pe_ratios.py --code 000001.SZ --year 2025
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data.pe_ratios import calculate_all_pe_ratios


def main() -> int:
    parser = argparse.ArgumentParser(description="计算股票的多维度市盈率")
    parser.add_argument("--code", required=True, help="股票代码, 如 600519.SH")
    parser.add_argument("--year", type=int, help="机构预测年份, 默认当前年份")
    
    args = parser.parse_args()
    
    try:
        result = calculate_all_pe_ratios(args.code, args.year)
        print(result)
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
