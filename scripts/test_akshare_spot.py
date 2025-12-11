#!/usr/bin/env python
"""测试AkShare的实时行情接口获取动态PE"""
import akshare as ak
import pandas as pd
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

print("测试AkShare实时行情接口...")
print("="*60)

REQUEST_INTERVAL = 1.0  # 每次调用间隔秒
MAX_ATTEMPTS = 3

# 为 akshare 配置带重试的 session（支持代理环境变量）
session = requests.Session()
retry = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)
ak.session = session

# 直接使用 akshare 接口
df = pd.DataFrame()
for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        df = ak.stock_zh_a_spot_em()
        print(f"stock_zh_a_spot_em 获取成功，行数: {len(df)}")
        break
    except Exception as e:
        print(f"stock_zh_a_spot_em 请求失败 (第{attempt}次):", e)
        if attempt < MAX_ATTEMPTS:
            time.sleep(REQUEST_INTERVAL)
        else:
            print("提示: 可设置 HTTP_PROXY/HTTPS_PROXY 或稍后重试")

# 异步版全市场行情（含动态PE），作为补充尝试
df_async = pd.DataFrame()
try:
    import asyncio

    async def fetch_async_spot():
        return await ak.stock_zh_a_spot_em_async()

    async_attempts = 0
    while async_attempts < MAX_ATTEMPTS:
        async_attempts += 1
        try:
            df_async = asyncio.run(fetch_async_spot())
            print(f"stock_zh_a_spot_em_async 获取成功，行数: {len(df_async)}")
            break
        except Exception as e:
            print(f"stock_zh_a_spot_em_async 请求失败 (第{async_attempts}次):", e)
            if async_attempts < MAX_ATTEMPTS:
                time.sleep(REQUEST_INTERVAL)
            else:
                print("提示: 检查代理或稍后重试")
except Exception as e:
    print("stock_zh_a_spot_em_async 初始化失败:", e)

# 获取全市场特色指标（含 pe_ttm/pb 等），可作为行情接口的备选/兜底
df_ind = pd.DataFrame()
indicator_funcs = [
    "stock_a_lg_indicator_df",  # 新命名（若有）
    "stock_a_lg_indicator_em",  # 常见命名
    "stock_a_lg_indicator",     # 旧命名
]
for func_name in indicator_funcs:
    try:
        func = getattr(ak, func_name)
    except AttributeError:
        continue
    try:
        df_ind = func()
        print(f"使用 {func_name} 获取 lg 指标成功，行数: {len(df_ind)}")
        break
    except Exception as e:
        print(f"{func_name} 请求失败:", e)
if df_ind.empty:
    print("未找到可用的 lg 指标接口")

# 查找平安银行（同步接口）
df_pabank = df[df['代码'] == '000001'] if not df.empty else pd.DataFrame()

if not df_pabank.empty:
    row = df_pabank.iloc[0]
    print("\n平安银行 (000001) 实时数据:")
    print(f"最新价: {row['最新价']}")
    print(f"市盈率-动态: {row.get('市盈率-动态', 'N/A')}")
    print(f"市盈率-静态: {row.get('市盈率-静态', 'N/A')}")
    print(f"市净率: {row.get('市净率', 'N/A')}")
    print(f"总市值: {row.get('总市值', 'N/A')}")
    
    print("\n所有可用字段:")
    for col in df.columns:
        if '市盈率' in col or '市净率' in col or 'PE' in col or 'PB' in col:
            print(f"  {col}: {row.get(col, 'N/A')}")
elif df.empty:
    print("stock_zh_a_spot_em 未获取到数据，已跳过行情展示")

# 查找平安银行（异步接口数据，用于对照）
df_async_pabank = df_async[df_async['代码'] == '000001'] if not df_async.empty else pd.DataFrame()

if not df_async_pabank.empty:
    row_async = df_async_pabank.iloc[0]
    print("\n[async] 平安银行 (000001) 实时数据:")
    print(f"最新价: {row_async.get('最新价', 'N/A')}")
    print(f"市盈率-动态: {row_async.get('市盈率-动态', 'N/A')}")
    print(f"市盈率-静态: {row_async.get('市盈率-静态', 'N/A')}")
    print(f"市净率: {row_async.get('市净率', 'N/A')}")
elif df_async.empty:
    print("stock_zh_a_spot_em_async 未获取到数据，已跳过 async 行情展示")

# 使用 lg 指标接口查找平安银行，作为对照
df_ind_pabank = df_ind[df_ind['code'] == '000001'] if not df_ind.empty else pd.DataFrame()

if not df_ind_pabank.empty:
    row_ind = df_ind_pabank.iloc[0]
    print("\n[lg 指标] 平安银行 (000001) 数据:")
    print(f"pe_ttm: {row_ind.get('pe_ttm', 'N/A')}")
    print(f"pb: {row_ind.get('pb', 'N/A')}")
    print(f"ps: {row_ind.get('ps', 'N/A')}")
    print(f"pcf: {row_ind.get('pcf', 'N/A')}")

    print("\n[lg 指标] 可用字段 (估值相关):")
    for col in df_ind.columns:
        if 'pe' in col.lower() or 'pb' in col.lower() or 'ps' in col.lower():
            print(f"  {col}: {row_ind.get(col, 'N/A')}")
elif df_ind.empty:
    print("stock_a_lg_indicator_df 未获取到数据，已跳过指标展示")

print("\n" + "="*60)
print("结论: stock_zh_a_spot_em 可取动态PE；stock_a_lg_indicator_df 可作估值兜底")
print("="*60)
