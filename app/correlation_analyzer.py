#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标的相关性分析模块
支持：股票、基金、指数
方法论：线性相关 | 协整 | Granger因果 | 滚动相关 | 尾部依赖 | 基本面逻辑
"""

import numpy as np
import pandas as pd
import tushare as ts
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import spearmanr, kendalltau
from statsmodels.tsa.stattools import coint, grangercausalitytests, adfuller
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import Config


class CorrelationAnalyzer:
    """
    标的相关性分析器
    支持多维度分析：线性相关、协整、Granger因果、滚动相关、尾部依赖
    """
    
    def __init__(self, adj_type: str = "hfq", price_mode: str = "log_return"):
        self.pro = ts.pro_api(Config.TUSHARE_TOKEN)
        self.cache = {}
        self.adj_type = adj_type  # qfq/hfq/None
        self.price_mode = price_mode  # "log_return" or "price"
    
    def fetch_data(self, code: str, asset_type: str = 'auto', adj_type: str | None = None) -> pd.DataFrame:
        """
        获取标的数据（股票、基金、指数）
        asset_type: 'stock' | 'fund' | 'index' | 'auto'
        adj_type: 'hfq' | 'qfq' | None (基金忽略复权)
        返回: DataFrame with columns [trade_date, close]
        """
        if code in self.cache:
            return self.cache[code]

        adj = self.adj_type if adj_type is None else adj_type

        try:
            # 自动识别资产类型
            if asset_type == 'auto':
                code_prefix = code.split('.')[0]
                # 通过代码前3位识别
                if code_prefix.startswith('1') or code_prefix.startswith('5') or code_prefix.startswith('16'):
                    # 基金: 10xxxx, 50xxxx, 16xxxx
                    asset_type = 'fund'
                elif code_prefix.startswith('3') or code_prefix.startswith('0') or code_prefix.startswith('6'):
                    # 股票: 0xxxx, 3xxxx, 6xxxx
                    asset_type = 'stock'
                else:
                    # 指数: 4xxxx, 9xxxx
                    asset_type = 'index'
            
            df = None
            
            # 尝试获取数据
            if asset_type == 'stock':
                try:
                    df = ts.pro_bar(ts_code=code, adj=adj)
                except:
                    df = None
            
            elif asset_type == 'fund':
                try:
                    df = self.pro.fund_daily(ts_code=code)
                except:
                    df = None
            
            elif asset_type == 'index':
                try:
                    df = self.pro.index_daily(ts_code=code)
                except:
                    df = None
            
            # 如果某类型失败，尝试其他类型
            if df is None or df.empty:
                for fallback_type in ['stock', 'fund', 'index']:
                    if fallback_type == asset_type:
                        continue
                    try:
                        if fallback_type == 'stock':
                            df = ts.pro_bar(ts_code=code, adj=adj)
                        elif fallback_type == 'fund':
                            df = self.pro.fund_daily(ts_code=code)
                        elif fallback_type == 'index':
                            df = self.pro.index_daily(ts_code=code)
                        
                        if df is not None and not df.empty:
                            asset_type = fallback_type
                            break
                    except:
                        continue
            
            if df is None or df.empty:
                raise ValueError(f"无法获取数据 (尝试了所有资产类型)")
            
            # 数据清洗
            df = df[['trade_date', 'close']].copy()
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date').reset_index(drop=True)
            df = df.dropna()
            
            self.cache[code] = df
            print(f"✓ 获取 {code} 数据: {len(df)} 条记录 ({df['trade_date'].min().date()} ~ {df['trade_date'].max().date()})")
            
            return df
        
        except Exception as e:
            print(f"✗ 获取数据失败 {code}: {str(e)}")
            raise
    
    def align_data(self, df1: pd.DataFrame, df2: pd.DataFrame) -> tuple:
        """对齐两个时间序列"""
        merged = df1.merge(df2, on='trade_date', how='inner', suffixes=('_a', '_b'))
        if len(merged) < 20:
            raise ValueError(f"对齐后数据过少: {len(merged)} 条")
        
        return merged['close_a'].values, merged['close_b'].values, merged['trade_date'].values
    
    # ==================== 维度1: 线性与方向角度 ====================
    
    def pearson_correlation(self, code1: str, code2: str) -> dict:
        """Pearson相关系数分析"""
        df1 = self.fetch_data(code1)
        df2 = self.fetch_data(code2)
        
        prices_a, prices_b, dates = self.align_data(df1, df2)
        if self.price_mode == "log_return":
            returns_a = np.diff(np.log(prices_a))
            returns_b = np.diff(np.log(prices_b))
            series_a, series_b = returns_a, returns_b
            dates_out = dates[1:]
        else:
            series_a, series_b = prices_a, prices_b
            dates_out = dates

        # Pearson相关系数
        corr_pearson, p_value = stats.pearsonr(series_a, series_b)

        # Spearman秩相关系数
        corr_spearman, p_value_spearman = spearmanr(series_a, series_b)

        # Kendall Tau
        corr_kendall, p_value_kendall = kendalltau(series_a, series_b)
        
        return {
            'pearson': {
                'corr': corr_pearson,
                'p_value': p_value,
                'strength': self._interpret_correlation(corr_pearson)
            },
            'spearman': {
                'corr': corr_spearman,
                'p_value': p_value_spearman,
                'strength': self._interpret_correlation(corr_spearman)
            },
            'kendall': {
                'corr': corr_kendall,
                'p_value': p_value_kendall,
            },
            'dates': dates_out,
            'prices_a': prices_a,
            'prices_b': prices_b,
            'returns_a': np.diff(np.log(prices_a)),
            'returns_b': np.diff(np.log(prices_b)),
        }
    
    def beta_coefficient(self, code_a: str, code_b: str) -> dict:
        """计算Beta系数（code_b相对于code_a的敏感度）"""
        df1 = self.fetch_data(code_a)
        df2 = self.fetch_data(code_b)
        
        prices_a, prices_b, dates = self.align_data(df1, df2)
        
        # 计算日收益率
        returns_a = np.diff(np.log(prices_a))
        returns_b = np.diff(np.log(prices_b))
        
        # OLS回归: returns_b = alpha + beta * returns_a
        X = sm.add_constant(returns_a)
        model = sm.OLS(returns_b, X).fit()
        
        alpha = model.params[0]
        beta = model.params[1]
        r_squared = model.rsquared
        p_value = model.pvalues[1]
        
        return {
            'beta': beta,
            'alpha': alpha,
            'r_squared': r_squared,
            'p_value': p_value,
            'interpretation': f"{code_b}相对{code_a}的敏感度: {beta:.3f}倍" if abs(beta) > 0.1 else "无显著敏感度"
        }
    
    # ==================== 维度2: 协整与长期均衡 ====================
    
    def cointegration_test(self, code1: str, code2: str) -> dict:
        """Engle-Granger协整检验"""
        df1 = self.fetch_data(code1)
        df2 = self.fetch_data(code2)
        
        prices_a, prices_b, dates = self.align_data(df1, df2)
        
        # Engle-Granger检验
        score, pvalue, _ = coint(prices_a, prices_b)
        
        # ADF检验（价差序列是否平稳）
        spread = prices_a - prices_b
        adf_result = adfuller(spread, maxlag=1, regression='c')
        
        return {
            'engle_granger': {
                'score': score,
                'p_value': pvalue,
                'cointegrated': pvalue < 0.05,
                'threshold': 0.05
            },
            'adf_spread': {
                'test_stat': adf_result[0],
                'p_value': adf_result[1],
                'stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            },
            'spread': spread,
            'dates': dates
        }
    
    def spread_analysis(self, code1: str, code2: str, window: int = 20) -> dict:
        """价差分析与Z-Score"""
        df1 = self.fetch_data(code1)
        df2 = self.fetch_data(code2)
        
        prices_a, prices_b, dates = self.align_data(df1, df2)
        
        # 计算价差
        spread = prices_a - prices_b
        
        # 滚动均值与标准差
        spread_df = pd.DataFrame({
            'spread': spread,
            'date': dates
        })
        
        spread_df['ma'] = spread_df['spread'].rolling(window).mean()
        spread_df['std'] = spread_df['spread'].rolling(window).std()
        spread_df['z_score'] = (spread_df['spread'] - spread_df['ma']) / spread_df['std']
        
        # 统计指标
        zscore_values = spread_df['z_score'].dropna().values
        extreme_count = np.sum(np.abs(zscore_values) > 2)
        extreme_pct = extreme_count / len(zscore_values) * 100
        
        return {
            'current_spread': spread[-1],
            'spread_mean': spread_df['ma'].iloc[-1],
            'spread_std': spread_df['std'].iloc[-1],
            'current_zscore': spread_df['z_score'].iloc[-1],
            'extreme_events': {
                'count': extreme_count,
                'percentage': extreme_pct,
                'interpretation': "高频率的极端价差事件，套利机会较多" if extreme_pct > 5 else "低频率，机会稀少"
            },
            'spread_series': spread_df
        }
    
    # ==================== 维度3: Granger因果检验 ====================
    
    def granger_causality_test(self, code_lead: str, code_lag: str, maxlag: int = 5) -> dict:
        """Granger因果检验：检查code_lead是否领先code_lag"""
        try:
            df1 = self.fetch_data(code_lead)
            df2 = self.fetch_data(code_lag)
            
            prices_a, prices_b, dates = self.align_data(df1, df2)
            
            # 计算对数收益率
            returns_a = np.diff(np.log(prices_a))
            returns_b = np.diff(np.log(prices_b))
            
            # 构建数据框
            data = pd.DataFrame({
                'lead': returns_a,
                'lag': returns_b
            })
            
            # 限制maxlag
            max_possible_lag = len(data) // 2 - 1
            maxlag = min(maxlag, max(1, max_possible_lag))
            
            # Granger检验
            result = grangercausalitytests(data[['lag', 'lead']], maxlag=maxlag, verbose=False)
            
            # 提取各阶段的p值
            p_values = [result[i][0][0][1] for i in range(1, maxlag + 1)]
            min_pval = min(p_values)
            best_lag = p_values.index(min_pval) + 1
            
            causality_exists = min_pval < 0.05
            
            return {
                'causality_exists': causality_exists,
                'p_values_by_lag': {f'lag_{i+1}': p for i, p in enumerate(p_values)},
                'min_pvalue': min_pval,
                'best_lag': best_lag,
                'interpretation': f"{code_lead} Granger-causes {code_lag} (最优滞后: {best_lag} 期)" if causality_exists else f"无显著因果关系"
            }
        except Exception as e:
            return {
                'causality_exists': False,
                'p_values_by_lag': {},
                'min_pvalue': 1.0,
                'best_lag': 1,
                'interpretation': f"Granger检验失败: {str(e)}"
            }
    
    def cross_correlation(self, code1: str, code2: str, max_lag: int = 10) -> dict:
        """互相关分析：找到最大相关性的滞后期"""
        df1 = self.fetch_data(code1)
        df2 = self.fetch_data(code2)
        
        prices_a, prices_b, dates = self.align_data(df1, df2)
        
        # 标准化
        prices_a_norm = (prices_a - np.mean(prices_a)) / np.std(prices_a)
        prices_b_norm = (prices_b - np.mean(prices_b)) / np.std(prices_b)
        
        # 互相关
        cross_corr = np.correlate(prices_a_norm, prices_b_norm, mode='full')
        cross_corr = cross_corr / len(prices_a)
        
        # 寻找峰值
        center = len(cross_corr) // 2
        lags = np.arange(-max_lag, max_lag + 1)
        ccf_values = cross_corr[center - max_lag: center + max_lag + 1]
        
        max_idx = np.argmax(np.abs(ccf_values))
        max_lag_found = lags[max_idx]
        max_corr = ccf_values[max_idx]
        
        return {
            'lags': lags.tolist(),
            'ccf_values': ccf_values.tolist(),
            'optimal_lag': max_lag_found,
            'max_correlation': max_corr,
            'interpretation': f"最大相关性在 lag={max_lag_found} (相关系数={max_corr:.3f})"
        }
    
    # ==================== 维度4: 动态时变分析 ====================
    
    def rolling_correlation(self, code1: str, code2: str, window: int = 30) -> dict:
        """滚动相关系数分析"""
        df1 = self.fetch_data(code1)
        df2 = self.fetch_data(code2)
        
        prices_a, prices_b, dates = self.align_data(df1, df2)
        
        # 计算对数收益率
        returns_a = np.diff(np.log(prices_a))
        returns_b = np.diff(np.log(prices_b))
        
        # 滚动相关系数
        rolling_corr = pd.Series(returns_a).rolling(window).corr(pd.Series(returns_b))
        
        corr_series = pd.DataFrame({
            'date': dates[1:],
            'rolling_corr': rolling_corr
        }).dropna()
        
        return {
            'window': window,
            'current_correlation': rolling_corr.iloc[-1],
            'mean_correlation': rolling_corr.mean(),
            'std_correlation': rolling_corr.std(),
            'min_correlation': rolling_corr.min(),
            'max_correlation': rolling_corr.max(),
            'volatility': rolling_corr.std(),
            'series': corr_series,
            'decoupling_events': self._detect_decoupling(rolling_corr.values)
        }
    
    # ==================== 维度5: 尾部依赖分析 ====================
    
    def tail_dependence(self, code1: str, code2: str, quantile: float = 0.05) -> dict:
        """尾部相关性分析（极端行情下的联动）"""
        df1 = self.fetch_data(code1)
        df2 = self.fetch_data(code2)
        
        prices_a, prices_b, dates = self.align_data(df1, df2)
        
        # 计算收益率
        returns_a = np.diff(np.log(prices_a))
        returns_b = np.diff(np.log(prices_b))
        
        # 定义左尾（下跌）和右尾（上涨）
        left_tail_a = returns_a < np.percentile(returns_a, quantile * 100)
        left_tail_b = returns_b < np.percentile(returns_b, quantile * 100)
        right_tail_a = returns_a > np.percentile(returns_a, (1 - quantile) * 100)
        right_tail_b = returns_b > np.percentile(returns_b, (1 - quantile) * 100)
        
        # 计算条件概率
        left_tail_joint = np.sum(left_tail_a & left_tail_b)
        right_tail_joint = np.sum(right_tail_a & right_tail_b)
        
        left_tail_dependence = left_tail_joint / np.sum(left_tail_a) if np.sum(left_tail_a) > 0 else 0
        right_tail_dependence = right_tail_joint / np.sum(right_tail_a) if np.sum(right_tail_a) > 0 else 0
        
        return {
            'quantile': quantile,
            'left_tail_dependence': {
                'probability': left_tail_dependence,
                'interpretation': f"当{code1}暴跌时，{code2}也暴跌的概率: {left_tail_dependence:.1%}"
            },
            'right_tail_dependence': {
                'probability': right_tail_dependence,
                'interpretation': f"当{code1}暴涨时，{code2}也暴涨的概率: {right_tail_dependence:.1%}"
            },
            'asymmetry': abs(left_tail_dependence - right_tail_dependence),
            'risk_assessment': "风险高" if left_tail_dependence > 0.7 else "风险中等" if left_tail_dependence > 0.4 else "风险低"
        }
    
    # ==================== 综合分析 ====================
    
    def comprehensive_analysis(self, code1: str, code2: str, report_path: str = None) -> dict:
        """综合分析：6维分析框架"""
        print(f"\n{'='*70}")
        print(f"开始综合分析: {code1} ↔ {code2}")
        print(f"{'='*70}\n")
        
        results = {}
        
        # 维度1：线性相关
        print("[维度1] 线性与方向分析...")
        results['linear'] = self.pearson_correlation(code1, code2)
        results['beta'] = self.beta_coefficient(code1, code2)
        
        # 维度2：协整
        print("[维度2] 协整与长期均衡分析...")
        results['cointegration'] = self.cointegration_test(code1, code2)
        results['spread'] = self.spread_analysis(code1, code2)
        
        # 维度3：Granger因果
        print("[维度3] Granger因果检验...")
        results['granger_lead'] = self.granger_causality_test(code1, code2)
        results['granger_lag'] = self.granger_causality_test(code2, code1)
        results['cross_corr'] = self.cross_correlation(code1, code2)
        
        # 维度4：滚动相关
        print("[维度4] 动态时变分析...")
        results['rolling'] = self.rolling_correlation(code1, code2)
        
        # 维度5：尾部依赖
        print("[维度5] 极端风险分析...")
        results['tail'] = self.tail_dependence(code1, code2)
        
        # 生成报告
        report = self._generate_report(code1, code2, results)
        
        if report_path:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n✓ 报告已保存: {report_path}")
        
        print(f"\n{'='*70}")
        print(report)
        print(f"{'='*70}\n")
        
        return results
    
    # ==================== 辅助方法 ====================
    
    def _interpret_correlation(self, corr: float) -> str:
        """解释相关系数强度"""
        abs_corr = abs(corr)
        if abs_corr >= 0.9:
            return "极强正相关" if corr > 0 else "极强负相关"
        elif abs_corr >= 0.7:
            return "强正相关" if corr > 0 else "强负相关"
        elif abs_corr >= 0.5:
            return "中等正相关" if corr > 0 else "中等负相关"
        elif abs_corr >= 0.3:
            return "弱正相关" if corr > 0 else "弱负相关"
        else:
            return "极弱或无相关"
    
    def _detect_decoupling(self, rolling_corr: np.ndarray) -> dict:
        """检测解耦事件（相关性突然下降）"""
        if len(rolling_corr) < 2:
            return {}
        
        changes = np.diff(rolling_corr)
        significant_drops = np.sum(changes < -0.2)  # 相关性下降超过0.2
        
        return {
            'significant_decouplings': significant_drops,
            'interpretation': f"发现 {significant_drops} 次显著解耦事件"
        }
    
    def _generate_report(self, code1: str, code2: str, results: dict) -> str:
        """生成综合分析报告"""
        report = f"""
{'='*70}
相关性分析综合报告
{'='*70}

标的对: {code1} ↔ {code2}
分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*70}
[维度1] 线性与方向分析
{'='*70}

Pearson相关系数: {results['linear']['pearson']['corr']:.4f} ({results['linear']['pearson']['strength']})
  - P值: {results['linear']['pearson']['p_value']:.6f}

Spearman秩相关系数: {results['linear']['spearman']['corr']:.4f}
  - P值: {results['linear']['spearman']['p_value']:.6f}

Kendall Tau系数: {results['linear']['kendall']['corr']:.4f}
  - P值: {results['linear']['kendall']['p_value']:.6f}

Beta系数 ({code2}相对{code1}):
  - β = {results['beta']['beta']:.4f}
  - α = {results['beta']['alpha']:.6f}
  - R² = {results['beta']['r_squared']:.4f}
  - 解释: {results['beta']['interpretation']}

{'='*70}
[维度2] 协整与长期均衡
{'='*70}

Engle-Granger协整检验:
  - 检验统计量: {results['cointegration']['engle_granger']['score']:.4f}
  - P值: {results['cointegration']['engle_granger']['p_value']:.6f}
  - 结论: {'存在协整关系 (可进行套利)' if results['cointegration']['engle_granger']['cointegrated'] else '不存在协整关系'}

价差平稳性检验 (ADF):
  - 检验统计量: {results['cointegration']['adf_spread']['test_stat']:.4f}
  - P值: {results['cointegration']['adf_spread']['p_value']:.6f}
  - 结论: {'价差序列平稳 (适合均值回归套利)' if results['cointegration']['adf_spread']['stationary'] else '价差序列非平稳'}

价差分析 (Z-Score):
  - 当前价差: {results['spread']['current_spread']:.4f}
  - 价差均值: {results['spread']['spread_mean']:.4f}
  - 价差标准差: {results['spread']['spread_std']:.4f}
  - 当前Z-Score: {results['spread']['current_zscore']:.4f}
  - 极端事件频率: {results['spread']['extreme_events']['percentage']:.2f}% ({results['spread']['extreme_events']['interpretation']})

{'='*70}
[维度3] 时间领先与因果
{'='*70}

Granger因果检验 ({code1} → {code2}):
  - {results['granger_lead']['interpretation']}
  - 各阶段P值: {results['granger_lead']['p_values_by_lag']}

Granger因果检验 ({code2} → {code1}):
  - {results['granger_lag']['interpretation']}
  - 各阶段P值: {results['granger_lag']['p_values_by_lag']}

互相关分析:
  - {results['cross_corr']['interpretation']}
  - 峰值相关性: {results['cross_corr']['max_correlation']:.4f}

{'='*70}
[维度4] 动态时变性
{'='*70}

滚动相关系数 (窗口={results['rolling']['window']}天):
  - 当前相关系数: {results['rolling']['current_correlation']:.4f}
  - 均值: {results['rolling']['mean_correlation']:.4f}
  - 标准差: {results['rolling']['volatility']:.4f}
  - 范围: [{results['rolling']['min_correlation']:.4f}, {results['rolling']['max_correlation']:.4f}]
  - {results['rolling']['decoupling_events']['interpretation']}

{'='*70}
[维度5] 极端风险与尾部依赖
{'='*70}

左尾相关性 (暴跌):
  - {results['tail']['left_tail_dependence']['interpretation']}

右尾相关性 (暴涨):
  - {results['tail']['right_tail_dependence']['interpretation']}

非对称性: {results['tail']['asymmetry']:.4f}
风险评估: {results['tail']['risk_assessment']}

{'='*70}
综合建议
{'='*70}

基于上述6维分析，建议：

1. 相关性稳定性: {'稳定' if results['rolling']['volatility'] < 0.1 else '不稳定' if results['rolling']['volatility'] > 0.3 else '中等'}
   → 滚动相关系数波动: {results['rolling']['volatility']:.4f}

2. 套利机会: {'较好' if results['cointegration']['engle_granger']['cointegrated'] and results['spread']['extreme_events']['percentage'] > 2 else '一般' if results['cointegration']['engle_granger']['cointegrated'] else '无'}
   → 协整关系: {results['cointegration']['engle_granger']['cointegrated']}
   → 极端事件频率: {results['spread']['extreme_events']['percentage']:.2f}%

3. 风险对冲: {'有效' if results['linear']['pearson']['corr'] > 0.5 and results['tail']['left_tail_dependence']['probability'] < 0.5 else '有限' if results['linear']['pearson']['corr'] > 0.3 else '不推荐'}
   → 正常时相关性: {results['linear']['pearson']['corr']:.4f}
   → 极端时相关性: {results['tail']['left_tail_dependence']['probability']:.1%}

{'='*70}
"""
        return report
