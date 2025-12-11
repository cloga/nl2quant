"""
Pairs Trading Screener using Unsupervised Learning (PCA + DBSCAN + Cointegration)
================================================================
根据 Pairs Trading.md 的方案实现，从A股中自动筛选配对交易标的。
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from statsmodels.tsa.stattools import coint
from multiprocessing import Pool, cpu_count
import itertools
import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.dca_backtest_engine import DCABacktestEngine
from app.data_cache import DataCache

class PairsScreener:
    """A股配对交易标的筛选器"""

    def __init__(self, start_date: str, end_date: str):
        """
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
        """
        self.start_date = start_date
        self.end_date = end_date
        self.engine = DCABacktestEngine()
        self.cache = DataCache()
        self.returns_df = None
        self.pca = None
        self.X_pca = None
        self.labels = None
        self.stock_codes = None
        self.fund_meta = {}
        self.price_df = None

    def fetch_fund_meta(self, codes: list) -> dict:
        """获取ETF元数据（基准、类型等），用于过滤同质化标的。"""
        meta = {}
        try:
            df = self.engine.pro.fund_basic(
                market="E",
                status="L",
                fields="ts_code,name,fund_type,invest_type,benchmark",
            )
            if df is None or df.empty:
                return meta
            df = df[df["ts_code"].isin(codes)]
            for _, row in df.iterrows():
                meta[row["ts_code"]] = {
                    "name": row.get("name"),
                    "fund_type": row.get("fund_type"),
                    "invest_type": row.get("invest_type"),
                    "benchmark": row.get("benchmark"),
                }
        except Exception as e:
            print(f"[WARN] 获取基金元数据失败: {e}")
        return meta

    def fetch_stock_data(self, codes: list) -> pd.DataFrame:
        """获取多只股票的收盘价数据"""
        prices = {}
        failed_codes = []
        
        total = len(codes)
        for idx, code in enumerate(codes, 1):
            try:
                # 先检查缓存
                print(f"[进度] [{idx}/{total}] 正在处理 {code}...", end=" ", flush=True)
                cached_data = self.cache.get(code, self.start_date, self.end_date)
                if cached_data is not None:
                    prices[code] = cached_data
                    print("(缓存)")
                    continue

                series = self.engine.fetch_etf_close(code, self.start_date, self.end_date)
                if series is not None and not series.empty and len(series) > 50:
                    prices[code] = series
                    # 保存到缓存
                    self.cache.set(code, self.start_date, self.end_date, series)
                    print(f"[OK] {len(series)}行")
                else:
                    print("(数据不足)")
                    failed_codes.append(code)
            except Exception as e:
                print(f"[WARN] 失败: {e}")
                failed_codes.append(code)
        
        if not prices:
            raise ValueError("未能获取任何有效的价格数据")
        
        df = pd.DataFrame(prices)
        
        # 统计缺失情况
        missing_pct = df.isnull().sum() / len(df) * 100
        print(f"\n[数据质量检查]")
        print(f"  - 成功获取 {len(prices)}/{total} 只标的数据")
        print(f"  - 失败 {len(failed_codes)} 只")
        print(f"  - 原始数据行数: {len(df)}")
        
        # 删除缺失率过高的列（标的）
        threshold = 0.5  # 允许50%缺失
        valid_cols = missing_pct[missing_pct < threshold * 100].index.tolist()
        if len(valid_cols) == 0:
            raise ValueError(f"没有符合质量要求的标的（缺失率阈值：{threshold*100}%）")
        
        df = df[valid_cols]
        print(f"  - 删除高缺失率标的后剩余: {len(df.columns)} 只")
        
        # 使用forward fill填充缺失值（假设停牌日价格不变）
        df = df.fillna(method='ffill')
        # 再使用backward fill处理开头的缺失
        df = df.fillna(method='bfill')
        # 最后删除仍有缺失的行
        df = df.dropna()
        
        print(f"  - 填充后有效数据行数: {len(df)}")
        
        if len(df) == 0:
            raise ValueError("填充后无有效数据行")
        if len(df.columns) < 2:
            raise ValueError(f"有效标的数量不足（仅 {len(df.columns)} 只），无法进行配对分析")
        
        # 记录价格数据用于后续波动率过滤
        self.price_df = df

        return df

    def compute_returns(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """计算日对数收益率"""
        # 使用对数收益率
        log_returns = np.log(price_df / price_df.shift(1)).dropna()
        return log_returns

    def perform_pca(self, returns_df: pd.DataFrame, n_components: int = 15) -> tuple:
        """PCA 降维

        注意：PCA 输入应为 (股票数, 时间特征) 矩阵；输出 X_pca 形状匹配股票数。
        """
        # 转置：行=股票，列=时间
        returns_T = returns_df.T.values

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(returns_T)

        # PCA - 确保 n_components 不超过数据维度
        max_components = min(X_scaled.shape[0], X_scaled.shape[1])
        actual_n_components = min(n_components, max_components)

        pca = PCA(n_components=actual_n_components)
        X_pca = pca.fit_transform(X_scaled)

        explained_var_ratio = pca.explained_variance_ratio_.sum()
        print(f"[OK] PCA 降维完成")
        print(f"  - 输入矩阵形状 (股票, 时间): {X_scaled.shape}")
        print(f"  - 输出矩阵形状 (股票, 主成分): {X_pca.shape}")
        print(f"  - 请求主成分数: {n_components}")
        print(f"  - 实际主成分数: {actual_n_components}（受数据维度限制）")
        print(f"  - 解释方差比: {explained_var_ratio:.1%}")

        return X_pca, pca

    def perform_dbscan(self, X_pca: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
        """DBSCAN 聚类"""
        clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clf.fit(X_pca).labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"[OK] DBSCAN 聚类完成")
        print(f"  - 聚类数量: {n_clusters}")
        print(f"  - 噪音点数: {n_noise}")
        
        return labels

    @staticmethod
    def check_cointegration_pair(args: tuple) -> dict:
        """单个配对的协整检验（用于多进程）"""
        s1, s2, code1, code2, min_corr, pvalue_threshold = args

        # 快速相关性检查
        corr = np.corrcoef(s1, s2)[0, 1]
        if np.isnan(corr) or corr < min_corr:
            return None

        # 协整检验
        try:
            score, pvalue, _ = coint(s1, s2)
            if pvalue < pvalue_threshold:
                return {
                    'stock_a': code1,
                    'stock_b': code2,
                    'correlation': corr,
                    'coint_pvalue': pvalue,
                    'coint_score': score,
                }
        except Exception:
            pass

        return None

    def find_cointegrated_pairs(
        self,
        returns_df: pd.DataFrame,
        labels: np.ndarray,
        min_corr: float = 0.85,
        pvalue_threshold: float = 0.05,
        use_log_price: bool = True,
    ) -> pd.DataFrame:
        """在聚类内部寻找协整配对（可选择对价格取对数）"""
        results = []
        stock_codes = returns_df.columns.tolist()
        
        # 确保 labels 和 stock_codes 长度一致
        if len(labels) != len(stock_codes):
            print(f"[WARN] 警告: labels 长度 ({len(labels)}) != stock_codes 长度 ({len(stock_codes)})")
            # 取较短的长度
            min_len = min(len(labels), len(stock_codes))
            labels = labels[:min_len]
            stock_codes = stock_codes[:min_len]
        
        # 构建簇到股票的映射
        clustered = pd.DataFrame({
            'stock': stock_codes,
            'cluster': labels
        })
        clustered = clustered[clustered['cluster'] != -1]
        
        total_pairs = 0
        
        for cluster_id in sorted(clustered['cluster'].unique()):
            group = clustered[clustered['cluster'] == cluster_id]
            stocks = group['stock'].tolist()
            
            if len(stocks) < 2:
                continue
            
            print(f"检验聚类 {cluster_id} ({len(stocks)} 只股票)...", end=" ")
            
            # 生成该簇内的所有组合
            combinations = list(itertools.combinations(stocks, 2))
            total_pairs += len(combinations)
            
            # 准备多进程任务
            tasks = []
            for s1, s2 in combinations:
                # --- 同质化标的过滤：同一跟踪标的直接跳过 ---
                m1 = self.fund_meta.get(s1, {})
                m2 = self.fund_meta.get(s2, {})
                bm1 = (m1.get('benchmark') or '').strip().lower()
                bm2 = (m2.get('benchmark') or '').strip().lower()
                if bm1 and bm2 and bm1 == bm2:
                    continue

                # --- 跨品种过滤：强制资产类别不同 ---
                def infer_asset_class(meta):
                    name = (meta.get('name') or '').lower()
                    ftype = (meta.get('fund_type') or '').lower()
                    itype = (meta.get('invest_type') or '').lower()
                    text = ' '.join([name, ftype, itype])
                    if any(k in text for k in ['货币', 'money']):
                        return 'cash'
                    if any(k in text for k in ['债', 'bond']):
                        return 'bond'
                    if any(k in text for k in ['金', '银', '油', '商品', '期货', '有色', '能源', '钢', '铜']):
                        return 'commodity'
                    if any(k in text for k in ['指数', '股票', 'equity', '沪深', '创业板', '中证', '上证', '深证']):
                        return 'equity'
                    return 'other'

                ac1 = infer_asset_class(m1)
                ac2 = infer_asset_class(m2)
                # 如果两者资产类别相同且类别已知，则跳过；未知类别则放行
                if ac1 == ac2 and ac1 != 'other':
                    continue

                # --- 波动率过滤：价差波动不足则跳过 ---
                price_a = self.price_df[s1]
                price_b = self.price_df[s2]
                aligned_a, aligned_b = price_a.align(price_b, join='inner')
                if len(aligned_a) < 60:
                    continue
                # 简单对冲比率：OLS beta
                try:
                    beta = np.linalg.lstsq(aligned_b.values.reshape(-1, 1), aligned_a.values, rcond=None)[0][0]
                    spread = aligned_a - beta * aligned_b
                    price_mean = (aligned_a + aligned_b).mean()
                    if price_mean == 0:
                        continue
                    spread_vol = spread.std() / price_mean
                    if spread_vol < 0.01:
                        continue
                except Exception:
                    continue

                # 协整基于价格序列（可选log）
                price_a = self.price_df[s1]
                price_b = self.price_df[s2]
                aligned_a, aligned_b = price_a.align(price_b, join='inner')
                if len(aligned_a) > 100:
                    if use_log_price:
                        series1 = np.log(aligned_a.values)
                        series2 = np.log(aligned_b.values)
                    else:
                        series1 = aligned_a.values
                        series2 = aligned_b.values
                    tasks.append((series1, series2, s1, s2, min_corr, pvalue_threshold))
            
            # 多进程协整检验
            cluster_results = []
            if tasks:
                with Pool(processes=min(cpu_count(), 4)) as pool:
                    # 使用静态方法以避免序列化 self
                    cluster_results = pool.map(PairsScreener.check_cointegration_pair, tasks)
                
                for res in cluster_results:
                    if res:
                        results.append(res)
            
            print(f"找到 {sum(1 for r in cluster_results if r)} 对协整配对")
        
        print(f"[OK] 总计检验 {total_pairs} 对，找到 {len(results)} 对协整配对")
        
        # 诊断：打印p值统计（即使无结果也有助诊断）
        if results:
            pvals = [r['coint_pvalue'] for r in results]
            print(f"\n[诊断] 协整p值分布：")
            print(f"  - 最小值: {min(pvals):.6f}")
            print(f"  - 平均值: {np.mean(pvals):.6f}")
            print(f"  - 最大值: {max(pvals):.6f}")
            print(f"  - p < 0.05: {sum(1 for p in pvals if p < 0.05)} 对")
            print(f"  - p < 0.1:  {sum(1 for p in pvals if p < 0.1)} 对")
            print(f"  - p < 0.2:  {sum(1 for p in pvals if p < 0.2)} 对")
            print(f"  - p < 0.3:  {sum(1 for p in pvals if p < 0.3)} 对")
        
        return pd.DataFrame(results) if results else pd.DataFrame()

    def visualize_clusters(self, X_pca: np.ndarray, labels: np.ndarray, stock_codes: list) -> go.Figure:
        """使用 t-SNE 可视化聚类结果"""
        print("生成 t-SNE 可视化...", end=" ")
        
        # 只对非噪音点进行 t-SNE
        mask = labels != -1
        vis_labels = labels[mask]
        vis_codes = [c for i, c in enumerate(stock_codes) if mask[i]]
        
        # 如果没有非噪音点或点数太少，跳过 t-SNE
        if len(vis_codes) < 5:
            # 直接使用 PCA 结果用于可视化
            if X_pca.shape[1] >= 2:
                vis_tsne = X_pca[:, :2]
            else:
                vis_tsne = np.random.randn(len(stock_codes), 2)
            vis_labels_all = labels
            vis_codes_all = stock_codes
        else:
            # 根据样本数调整 perplexity
            perplexity = min(30, (len(vis_codes) - 1) // 3)
            X_tsne = TSNE(n_components=2, random_state=42, perplexity=max(1, perplexity)).fit_transform(X_pca[mask])
            
            # 标记噪音点
            noise_mask = labels == -1
            if noise_mask.any():
                X_noise_tsne = np.random.randn(noise_mask.sum(), 2)
                vis_tsne = np.vstack([X_tsne, X_noise_tsne])
                vis_labels_all = np.hstack([vis_labels, np.full(noise_mask.sum(), -1)])
                vis_codes_all = vis_codes + [c for i, c in enumerate(stock_codes) if noise_mask[i]]
            else:
                vis_tsne = X_tsne
                vis_labels_all = vis_labels
                vis_codes_all = vis_codes
        
        df_vis = pd.DataFrame({
            't-SNE 1': vis_tsne[:, 0],
            't-SNE 2': vis_tsne[:, 1],
            '聚类': vis_labels_all,
            '股票代码': vis_codes_all,
        })
        
        fig = px.scatter(
            df_vis,
            x='t-SNE 1',
            y='t-SNE 2',
            color='聚类',
            text='股票代码',
            title='A股聚类可视化 (t-SNE)',
            hover_data={'聚类': True},
        )
        fig.update_traces(textposition='top center', textfont=dict(size=8))
        fig.update_layout(height=700, width=900)
        
        print("[OK]")
        return fig

    def run(
        self,
        codes: list,
        eps: float = 0.5,
        n_components: int = 15,
        min_corr: float = 0.85,
        pvalue_threshold: float = 0.05,
        min_samples: int = 2,
        use_log_price: bool = True,
    ) -> dict:
        """运行完整的筛选流程"""
        print("\n" + "="*60)
        print("A股配对交易标的筛选 (PCA + DBSCAN + 协整检验)")
        print("="*60)
        
        # 1. 获取数据
        print("\n[步骤1] 获取股票数据...")
        price_df = self.fetch_stock_data(codes)
        self.stock_codes = price_df.columns.tolist()
        # 获取ETF元数据，便于后续过滤同质化标的
        self.fund_meta = self.fetch_fund_meta(self.stock_codes)
        
        # 2. 计算收益率
        print("\n[步骤2] 计算日对数收益率...")
        returns_df = self.compute_returns(price_df)
        self.returns_df = returns_df
        print(f"[OK] 收益率矩阵形状: {returns_df.shape}")
        
        # 3. PCA 降维
        print("\n[步骤3] PCA 降维...")
        X_pca, pca = self.perform_pca(returns_df, n_components=n_components)
        self.X_pca = X_pca
        self.pca = pca
        
        # 4. DBSCAN 聚类
        print("\n[步骤4] DBSCAN 聚类...")
        labels = self.perform_dbscan(X_pca, eps=eps, min_samples=min_samples)
        self.labels = labels

        # 5. 协整检验（带过滤器）
        print("\n[步骤5] 协整检验（含过滤器）...")
        pairs_df = self.find_cointegrated_pairs(
            returns_df,
            labels,
            min_corr=min_corr,
            pvalue_threshold=pvalue_threshold,
            use_log_price=use_log_price,
        )

        # 可视化（可选）
        cluster_fig = None
        try:
            cluster_fig = self.visualize_clusters(X_pca, labels, self.stock_codes)
        except Exception as e:
            print(f"[WARN] 可视化生成失败: {e}")

        print("\n" + "="*60)
        print("筛选完成！")
        print("="*60 + "\n")

        return {
            'pairs': pairs_df,
            'cluster_fig': cluster_fig,
            'pca': pca,
            'X_pca': X_pca,
            'labels': labels,
            'stock_codes': self.stock_codes,
            'returns_df': returns_df,
            'min_corr': min_corr,
            'pvalue_threshold': pvalue_threshold,
            'min_samples': min_samples,
            'use_log_price': use_log_price,
        }

