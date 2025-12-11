#!/usr/bin/env python
"""
检查Tushare接口权限和可用的盈利预测数据源
"""

import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv()
ts.set_token(os.getenv('TUSHARE_TOKEN'))
pro = ts.pro_api()

print("="*60)
print("Tushare 可用的盈利/预测相关接口测试")
print("="*60)

test_code = '600000.SH'  # 浦发银行作为测试

# 1. 业绩预告 (免费接口)
print("\n1. 业绩预告 (forecast) - 基础权限")
print("   文档: https://tushare.pro/document/2?doc_id=45")
try:
    df = pro.forecast(ts_code=test_code)
    print(f"   ✓ 可用 - 获取到 {len(df)} 条记录")
    if not df.empty:
        print(f"   最新预告: {df.iloc[0]['end_date']}, 类型: {df.iloc[0]['type']}")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

# 2. 业绩快报 (免费接口)
print("\n2. 业绩快报 (express) - 基础权限")
print("   文档: https://tushare.pro/document/2?doc_id=46")
try:
    df = pro.express(ts_code=test_code)
    print(f"   ✓ 可用 - 获取到 {len(df)} 条记录")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

# 3. 财务指标 (免费接口)
print("\n3. 财务指标 (fina_indicator) - 基础权限")
print("   文档: https://tushare.pro/document/2?doc_id=79")
try:
    df = pro.fina_indicator(ts_code=test_code)
    print(f"   ✓ 可用 - 获取到 {len(df)} 条记录")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

# 4. 盈利预测 (VIP接口)
print("\n4. 盈利预测和评级 (forecast_vip) - VIP权限")
print("   文档: https://tushare.pro/document/2?doc_id=206")
try:
    df = pro.forecast_vip(ts_code=test_code)
    print(f"   ✓ 可用 - 获取到 {len(df)} 条记录")
    if not df.empty:
        print(f"   最新预测: {df.iloc[0]}")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

# 5. 券商盈利预测明细 (高级VIP)
print("\n5. 券商盈利预测明细 (stk_surv) - 高级VIP")
try:
    df = pro.stk_surv(ts_code=test_code)
    print(f"   ✓ 可用 - 获取到 {len(df)} 条记录")
except Exception as e:
    print(f"   ✗ 不可用 - {e}")

print("\n" + "="*60)
print("替代方案建议:")
print("="*60)
print("""
如果VIP接口不可用，可以使用以下替代方案计算动态PE:

方案1: 使用业绩预告 (当前方案)
  - 优点: 免费，公司官方预告
  - 缺点: 覆盖率低（约30-40%），滞后性强
  
方案2: 使用业绩快报 + 历史增长率
  - 用最近几期业绩快报计算增长趋势
  - 线性外推或回归预测未来盈利
  
方案3: 结合财务指标计算
  - 用ROE、营收增长率等指标估算盈利增长
  - 公式: 预期增长率 ≈ ROE × (1 - 分红率)
  
方案4: 行业平均法
  - 按行业分组，用行业平均增长率
  - 适合无个股预告的公司
  
方案5: 爬取第三方数据 (高级)
  - 东方财富、同花顺等网站的一致预期
  - 需要网页爬虫技术
""")

print("\n当前推荐: 方案1(业绩预告) + 方案2(历史增长率)")
print("="*60)
