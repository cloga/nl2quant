"""
Pairs Trading Screener - Streamlit Page
选择配对交易标的的完整工具
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.pairs_screener import PairsScreener

from app.data_cache import DataCache

st.set_page_config(page_title="配对交易筛选器", layout="wide", initial_sidebar_state="expanded")

st.title("🎯 A股配对交易标的筛选器")
st.markdown("使用 PCA + DBSCAN + 协整检验，自动发现配对交易机会")

with st.sidebar:
    st.header("⚙️ 参数配置")
    
    # 日期范围
    col1, col2 = st.columns(2)
    with col1:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now().date(),
            key="pairs_end_date"
        )
    with col2:
        days_back = st.slider("往前回溯天数", 90, 1000, 365)
        start_date = end_date - timedelta(days=days_back)
    
    st.write(f"📅 数据范围: {start_date} ~ {end_date} ({days_back}天)")
    
    # PCA 参数
    st.subheader("PCA 参数")
    n_components = st.slider(
        "主成分数量",
        min_value=5,
        max_value=30,
        value=15,
        help="保留的PCA主成分数。越多越详细，但可能过拟合"
    )
    
    # DBSCAN 参数
    st.subheader("DBSCAN 参数")
    eps = st.slider(
        "邻域半径 (eps)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="越小聚类越紧，簇数越多；越大越松散"
    )
    
    # 股票代码
    st.subheader("股票池")
    preset = st.radio(
        "选择预设股票池",
        ["沪深300 (前50)", "中证500 (前50)", "自定义"],
        horizontal=True
    )
    
    if preset == "沪深300 (前50)":
        # 沪深300成分股前50（主要蓝筹）
        codes_text = """
000858	五粮液
000651	格力电器
600887	伊利股份
000333	美的集团
000568	泸州老窖
600519	贵州茅台
600900	长江电力
601398	工商银行
601939	建设银行
601288	农业银行
600016	民生银行
600029	南方航空
600837	上海临港
600009	上海机场
601328	交通银行
601166	兴业银行
601169	北京银行
601988	中国银行
601818	光大银行
601658	邮储银行
601628	中国人寿
601318	中国平安
601336	新华保险
600048	保利发展
601225	上海电气
600023	浙能电力
000001	平安银行
000002	万科A
001979	招商银行
600000	浦发银行
601601	中国太保
601098	中南传媒
000858	五粮液
000996	中国中期
600030	中信证券
601688	华泰证券
601211	国泰君安
601099	太平洋
601377	兴业证券
        """
        codes = [line.strip().split()[0] for line in codes_text.strip().split('\n') if line.strip()]
    elif preset == "中证500 (前50)":
        # 中证500成分股前50（中小盘）
        codes_text = """
603392	该隆制造
002920	德固特
002963	新北洋
603659	璞泰来
601689	拓普集团
002968	蛋壳公寓
603501	韦尔股份
300418	昆仑万维
300482	中坚科技
600690	青岛海尔
301020	骑士股份
600720	祎鑫科技
601996	丰田
        """
        codes = []
        # 使用一些常见的中小盘股票
        codes = [
            "002920", "002963", "601689", "300418", "300482", "600690",
            "000858", "000651", "000333", "000568", "601328",
            "600837", "601169", "601658", "601318", "601336",
            "600048", "000001", "000002", "001979", "600000",
            "601601", "600030", "601211", "601099", "601377",
        ]
    else:
        codes_input = st.text_area(
            "输入股票代码（逗号或换行分隔）",
            value="601398,601939,601288,000858,600519,600016",
            height=100
        )
        codes = [c.strip() for c in codes_input.replace(',', '\n').split('\n') if c.strip()]
    
    st.write(f"📊 选定股票数: {len(codes)}")
    
    run_button = st.button("🚀 开始筛选", type="primary", use_container_width=True)

# 主体
if run_button:
    if not codes:
        st.error("❌ 请输入至少一只股票")
    else:
        try:
            # 转换日期格式
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            
            # 创建筛选器并运行
            screener = PairsScreener(start_str, end_str)
            results = screener.run(codes, eps=eps, n_components=n_components)
            
            # 将结果保存到 session state
            st.session_state.pairs_results = results
            st.success("✅ 筛选完成！")
            
        except Exception as e:
            st.error(f"❌ 筛选过程中出错: {str(e)}")
            st.exception(e)

# 显示结果
if 'pairs_results' in st.session_state:
    results = st.session_state.pairs_results
    
    # 选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["配对结果", "聚类可视化", "簇内详情", "原始数据"])
    
    with tab1:
        st.subheader("协整配对结果")
        pairs_df = results['pairs']
        
        if len(pairs_df) > 0:
            # 按correlation降序排序
            pairs_df = pairs_df.sort_values('correlation', ascending=False)
            
            st.metric("找到的协整配对数", len(pairs_df))
            
            # 显示表格
            display_df = pairs_df.copy()
            display_df['correlation'] = display_df['correlation'].apply(lambda x: f"{x:.4f}")
            display_df['coint_pvalue'] = display_df['coint_pvalue'].apply(lambda x: f"{x:.6f}")
            display_df['coint_score'] = display_df['coint_score'].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # 导出按钮
            csv = pairs_df.to_csv(index=False)
            st.download_button(
                "📥 下载配对结果 (CSV)",
                csv,
                "pairs_trading_results.csv",
                "text/csv"
            )
            
            # 显示Top 5
            st.markdown("#### 🏆 Top 5 最强配对")
            top5 = pairs_df.head(5)
            for idx, row in top5.iterrows():
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("配对", f"{row['stock_a']} ↔️ {row['stock_b']}")
                    with col2:
                        st.metric("相关系数", f"{row['correlation']:.4f}")
                    with col3:
                        st.metric("协整P值", f"{row['coint_pvalue']:.6f}")
                    with col4:
                        st.metric("协整得分", f"{row['coint_score']:.4f}")
        else:
            st.warning("⚠️ 未找到协整配对。尝试调整 eps 参数。")
    
    with tab2:
        st.subheader("聚类可视化 (t-SNE)")
        st.plotly_chart(results['cluster_fig'], use_container_width=True)
        st.info("💡 每个点代表一只股票，同一颜色的点属于同一聚类。")
    
    with tab3:
        st.subheader("聚类详情")
        labels = results['labels']
        stock_codes = results['stock_codes']
        
        clusters = pd.DataFrame({
            '股票代码': stock_codes,
            '聚类': labels,
        }).sort_values('聚类')
        
        # 统计
        col1, col2, col3 = st.columns(3)
        with col1:
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            st.metric("聚类数量", n_clusters)
        with col2:
            n_noise = list(labels).count(-1)
            st.metric("噪音点数", n_noise)
        with col3:
            st.metric("总股票数", len(stock_codes))
        
        # 按聚类显示
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                st.markdown("#### 🔴 噪音点 (未聚类)")
            else:
                st.markdown(f"#### 聚类 {cluster_id}")
            
            cluster_stocks = clusters[clusters['聚类'] == cluster_id]['股票代码'].tolist()
            st.write(f"包含 {len(cluster_stocks)} 只股票: {', '.join(cluster_stocks)}")
    
    with tab4:
        st.subheader("原始数据视图")
        
        view_type = st.radio("选择视图", ["配对表", "聚类表", "PCA成分"], horizontal=True)
        
        if view_type == "配对表":
            st.dataframe(results['pairs'], use_container_width=True)
        elif view_type == "聚类表":
            cluster_df = pd.DataFrame({
                '股票代码': results['stock_codes'],
                '聚类': results['labels'],
            })
            st.dataframe(cluster_df, use_container_width=True)
        else:
            # PCA成分
            pca = results['pca']
            components_df = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)]
            )
            st.dataframe(components_df, use_container_width=True)
            
            # 显示解释方差
            st.markdown("##### 解释方差比")
            var_df = pd.DataFrame({
                '成分': [f'PC{i+1}' for i in range(pca.n_components_)],
                '方差比': pca.explained_variance_ratio_,
                '累计方差比': pca.explained_variance_ratio_.cumsum(),
            })
            st.dataframe(var_df, use_container_width=True)

else:
    st.info("👈 在左侧设置参数后，点击 '开始筛选' 按钮")

    # 缓存管理
    st.subheader("缓存管理")
    cache = DataCache()
    cache_stats = cache.get_cache_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("缓存文件数", cache_stats['total_files'])
    with col2:
        st.metric("缓存大小 (MB)", cache_stats['total_size_mb'])
    
    if st.button("清除过期缓存 (>24h)", use_container_width=True):
        cache.clear_expired(max_age_hours=24)
        st.success(f"已清理过期缓存！")
    
    if st.button("清除所有缓存", use_container_width=True):
        cache.clear_all()
        st.success(f"所有缓存已清除！")
