"""PE数据缓存管理页面

功能：
1. 显示缓存状态和更新日期
2. 异步触发全量更新
3. 实时显示更新进度
4. 导出缓存数据到CSV
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import tushare as ts
from dotenv import load_dotenv

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data.pe_cache import PECache, batch_compute_and_cache, export_cache_to_csv


st.set_page_config(
    page_title="PE数据缓存管理",
    page_icon="💾",
    layout="wide"
)

st.title("💾 PE数据缓存管理")

# 初始化会话状态
if "update_running" not in st.session_state:
    st.session_state.update_running = False
if "progress" not in st.session_state:
    st.session_state.progress = {"current": 0, "total": 0, "ts_code": "", "status": ""}


# 加载缓存信息
cache = PECache()
metadata = cache.get_metadata()

# 显示缓存状态
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("缓存记录数", metadata.get("total_stocks", 0))

with col2:
    last_update = metadata.get("last_update", "未更新")
    st.metric("最后更新时间", last_update)

with col3:
    is_fresh = cache.is_cache_fresh(max_age_days=1)
    status = "✅ 新鲜" if is_fresh else "⚠️ 需要更新"
    st.metric("缓存状态", status)

st.divider()

# 更新选项
st.subheader("📊 数据更新")

col_opt1, col_opt2 = st.columns(2)

with col_opt1:
    force_update = st.checkbox(
        "强制全量更新",
        help="勾选后将忽略缓存，重新计算所有股票的PE数据"
    )

with col_opt2:
    limit = st.number_input(
        "限制股票数量（测试用）",
        min_value=0,
        max_value=10000,
        value=0,
        help="仅用于测试，0表示不限制"
    )

st.divider()


def progress_callback(current, total, ts_code, status):
    """进度回调函数"""
    st.session_state.progress = {
        "current": current,
        "total": total,
        "ts_code": ts_code,
        "status": status
    }


def run_update_task(ts_codes, force_update):
    """后台更新任务"""
    try:
        batch_compute_and_cache(
            ts_codes=ts_codes,
            force_update=force_update,
            use_batch_daily=True,
            progress_callback=progress_callback
        )
    finally:
        st.session_state.update_running = False


# 更新按钮
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button(
        "🚀 开始更新",
        disabled=st.session_state.update_running,
        type="primary",
        use_container_width=True
    ):
        # 获取股票列表
        load_dotenv()
        pro = ts.pro_api()
        
        with st.spinner("获取股票列表..."):
            df_basic = pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,list_date'
            )
            
            # 仅保留上市≥5年的股票
            now = datetime.now()
            cutoff_str = f"{now.year - 5}{now.strftime('%m%d')}"
            df_basic = df_basic.copy()
            df_basic["list_date"] = df_basic["list_date"].astype(str).fillna("")
            df_basic = df_basic[df_basic["list_date"].str.len() == 8]
            df_basic = df_basic[df_basic["list_date"] <= cutoff_str]
            
            if limit > 0:
                df_basic = df_basic.head(limit)
            
            ts_codes = df_basic["ts_code"].tolist()
            st.info(f"将更新 {len(ts_codes)} 只股票的PE数据")
        
        # 启动后台线程
        st.session_state.update_running = True
        st.session_state.progress = {"current": 0, "total": len(ts_codes), "ts_code": "", "status": ""}
        
        thread = threading.Thread(
            target=run_update_task,
            args=(ts_codes, force_update),
            daemon=True
        )
        thread.start()
        st.rerun()

with col_btn2:
    if st.button(
        "💾 导出到CSV",
        use_container_width=True
    ):
        output_file = f"data/pe_ratios_cache_{datetime.now().strftime('%Y%m%d')}.csv"
        with st.spinner("导出中..."):
            export_cache_to_csv(output_file)
        st.success(f"导出成功: {output_file}")

with col_btn3:
    if st.button(
        "🔄 刷新状态",
        use_container_width=True
    ):
        st.rerun()

# 显示进度
if st.session_state.update_running:
    st.divider()
    st.subheader("⏳ 更新进度")
    
    progress = st.session_state.progress
    current = progress.get("current", 0)
    total = progress.get("total", 1)
    ts_code = progress.get("ts_code", "")
    status = progress.get("status", "")
    
    # 进度条
    progress_pct = current / total if total > 0 else 0
    st.progress(progress_pct, text=f"进度: {current}/{total} ({progress_pct*100:.1f}%)")
    
    # 状态信息
    status_text = {
        "cached": "✓ 使用缓存",
        "computed": "🔄 重新计算",
        "error": "❌ 计算失败"
    }.get(status, "⏳ 处理中")
    
    st.info(f"当前: {ts_code} - {status_text}")
    
    # 自动刷新
    st.empty()
    import time
    time.sleep(0.5)
    st.rerun()

elif st.session_state.progress.get("current", 0) > 0:
    st.success("✅ 更新完成！")
    if st.button("清除进度信息"):
        st.session_state.progress = {"current": 0, "total": 0, "ts_code": "", "status": ""}
        st.rerun()

st.divider()

# 显示缓存数据预览
st.subheader("📋 缓存数据预览")

cache_data = cache.load_cache()
if cache_data:
    # 转为DataFrame
    rows = []
    for ts_code, data in list(cache_data.items())[:100]:  # 只显示前100条
        rows.append(data)
    
    df = pd.DataFrame(rows)
    
    # 选择显示的列
    display_cols = [
        "ts_code", "trade_date", "close_price", "market_cap",
        "static_pe", "ttm_pe", "linear_extrapolate_pe",
        "forecast_pe_mean", "forecast_pe_median"
    ]
    df_display = df[[col for col in display_cols if col in df.columns]]
    
    st.dataframe(df_display, use_container_width=True, height=400)
    
    if len(cache_data) > 100:
        st.info(f"显示前100条记录，共 {len(cache_data)} 条")
else:
    st.warning("缓存为空，请先执行数据更新")

# 页面说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    ### 功能说明
    
    1. **缓存状态**: 显示当前缓存的记录数和更新时间
    2. **数据更新**: 
       - 默认增量更新（跳过已缓存的股票）
       - 勾选"强制全量更新"将重新计算所有股票
       - "限制股票数量"用于小规模测试
    3. **导出CSV**: 将缓存数据导出为CSV文件，便于分析
    
    ### 优化说明
    
    - **批量获取行情**: 一次性获取所有股票的 daily_basic 数据，减少网络请求
    - **本地缓存**: 计算结果保存在 `data/cache/pe_cache.json`
    - **增量更新**: 仅计算新增或需要更新的股票
    - **异步更新**: 后台线程执行更新，不阻塞页面操作
    
    ### 注意事项
    
    - 全量更新约5000只股票可能需要1-2小时（取决于网络和Tushare限流）
    - 更新期间可以查看实时进度
    - 建议每日更新一次即可
    """)
