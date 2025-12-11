#!/usr/bin/env python
"""
检查AkShare可用的市盈率和盈利预测接口
"""

try:
    import akshare as ak
    print("✓ AkShare已安装")
except ImportError:
    print("✗ AkShare未安装，请运行: pip install akshare")
    exit(1)

print("\n" + "="*60)
print("AkShare 市盈率和盈利预测相关接口")
print("="*60)

test_code = '000001'  # 平安银行

# 1. 个股估值指标（包含PE、PB等）
print("\n1. 个股估值指标 (stock_a_indicator_lg)")
print("   包含: PE(动)、PE(静)、PE(TTM)、PB、PS等")
try:
    df = ak.stock_a_indicator_lg(symbol=test_code)
    print(f"   ✓ 可用 - 获取到 {len(df)} 条历史数据")
    if not df.empty:
        latest = df.iloc[-1]
        print(f"   最新数据({latest['trade_date']}):")
        print(f"     市盈率(动): {latest.get('pe_ttm', 'N/A')}")
        print(f"     市净率: {latest.get('pb', 'N/A')}")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

# 2. 东方财富-数据中心-特色数据-机构调研
print("\n2. 机构调研统计 (stock_jgdy_tj_em)")
try:
    df = ak.stock_jgdy_tj_em(date="20241201")
    print(f"   ✓ 可用 - 获取到 {len(df)} 条记录")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

# 3. 东方财富-数据中心-研究报告-盈利预测
print("\n3. 盈利预测 (stock_profit_forecast_em)")
print("   ★★★ 关键接口：券商一致预期的EPS预测")
try:
    df = ak.stock_profit_forecast_em(symbol=test_code)
    print(f"   ✓ 可用 - 获取到 {len(df)} 条记录")
    if not df.empty:
        print("\n   字段:", df.columns.tolist())
        print("\n   最新预测数据:")
        print(df.head())
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

# 4. 个股详情实时数据
print("\n4. 个股实时行情 (stock_zh_a_spot_em)")
try:
    df = ak.stock_zh_a_spot_em()
    df_stock = df[df['代码'] == test_code]
    print(f"   ✓ 可用")
    if not df_stock.empty:
        print(f"   市盈率(动): {df_stock.iloc[0].get('市盈率-动态', 'N/A')}")
        print(f"   市净率: {df_stock.iloc[0].get('市净率', 'N/A')}")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

# 5. 同花顺-财务指标
print("\n5. 同花顺财务指标 (stock_financial_abstract_ths)")
try:
    df = ak.stock_financial_abstract_ths(symbol=test_code, indicator="按报告期")
    print(f"   ✓ 可用 - 获取到 {len(df)} 条记录")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

print("\n" + "="*60)
print("推荐方案:")
print("="*60)
print("""
方案1: 使用 stock_profit_forecast_em (东方财富盈利预测)
  - 包含券商一致预期的EPS、营收、净利润预测
  - 多年度预测（2024E、2025E、2026E等）
  - 数据来源权威，与雪球/同花顺一致
  - 免费无限制

方案2: 使用 stock_zh_a_spot_em (实时行情)
  - 直接包含市盈率(动)、市盈率(静)
  - 数据实时更新
  - 但无法看到预测明细

推荐: 方案1 + 当前股价计算动态PE
公式: 动态PE = 当前股价 ÷ 下一年度预测EPS
""")
