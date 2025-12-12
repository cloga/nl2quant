"""计算A股股票的多维度市盈率指标。

根据 design/pe_ration_relationship.md 定义:
1. 静态市盈率 (Static PE/LYR) - 市值 / 上一年度净利润
2. TTM市盈率 - 市值 / 过去4季度净利润之和
3. 动态市盈率(线性外推) - 市值 / (最新季报净利润 × 年化系数)
4. 动态市盈率(机构预测-平均) - 市值 / 分析师预测净利润平均值
5. 动态市盈率(机构预测-中位数) - 市值 / 分析师预测净利润中位数
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
import tushare as ts
from dotenv import load_dotenv


@dataclass
class PERatios:
    """市盈率指标集合"""
    ts_code: str
    trade_date: str
    close_price: float
    market_cap: float  # 亿元
    
    # 第一维度：过去与现在的"事实"
    static_pe: Optional[float] = None  # 静态PE (LYR)
    ttm_pe: Optional[float] = None  # TTM市盈率
    
    # 第二维度：当下的"脉冲"
    linear_extrapolate_pe: Optional[float] = None  # 线性外推PE
    latest_quarter: Optional[str] = None  # 最新季报期间
    
    # 第三维度：未来的"预期"
    forecast_pe_mean: Optional[float] = None  # 机构预测PE(平均)
    forecast_pe_median: Optional[float] = None  # 机构预测PE(中位数)
    
    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"股票代码: {self.ts_code}",
            f"交易日期: {self.trade_date}",
            f"收盘价: {self.close_price:.2f} 元",
            f"总市值: {self.market_cap:.2f} 亿元",
            f"{'='*60}",
            "\n【第一维度：过去与现在的'事实'】",
            f"  静态市盈率 (Static PE): {self.static_pe:.2f}" if self.static_pe else "  静态市盈率: N/A",
            f"  TTM市盈率 (滚动): {self.ttm_pe:.2f}" if self.ttm_pe else "  TTM市盈率: N/A",
        ]
        
        if self.static_pe and self.ttm_pe:
            if self.static_pe < self.ttm_pe:
                lines.append("  → 信号: 业绩在下滑 (静态PE < TTM PE)")
            elif self.static_pe > self.ttm_pe:
                lines.append("  → 信号: 业绩在增长 (静态PE > TTM PE)")
        
        lines.extend([
            "\n【第二维度:当下的'脉冲'】",
            f"  动态市盈率 (线性外推): {self.linear_extrapolate_pe:.2f}" if self.linear_extrapolate_pe else "  动态市盈率 (线性外推): N/A",
            f"  最新季报: {self.latest_quarter}" if self.latest_quarter else "  最新季报: N/A",
        ])
        
        if self.linear_extrapolate_pe and self.ttm_pe:
            ratio = (self.linear_extrapolate_pe - self.ttm_pe) / self.ttm_pe * 100
            if self.linear_extrapolate_pe < self.ttm_pe * 0.7:
                lines.append(f"  → 信号: 最新季度大爆发! (低于TTM {abs(ratio):.1f}%)")
            elif self.linear_extrapolate_pe > self.ttm_pe * 1.3:
                lines.append(f"  → 信号: 最新季度业绩疲软 (高于TTM {ratio:.1f}%)")
        
        lines.extend([
            "\n【第三维度:未来的'预期'】",
            f"  机构预测PE (平均): {self.forecast_pe_mean:.2f}" if self.forecast_pe_mean else "  机构预测PE (平均): N/A",
            f"  机构预测PE (中位数): {self.forecast_pe_median:.2f}" if self.forecast_pe_median else "  机构预测PE (中位数): N/A",
        ])
        
        if self.forecast_pe_median and self.linear_extrapolate_pe:
            if self.linear_extrapolate_pe < self.forecast_pe_median * 0.7:
                lines.append("  → 信号: 分析师认为Q1爆发不可持续")
            elif self.linear_extrapolate_pe > self.forecast_pe_median * 1.3:
                lines.append("  → 信号: 分析师认为公司后劲很足")
        
        # 整体趋势判断
        if all([self.static_pe, self.ttm_pe, self.forecast_pe_median]):
            lines.append(f"\n【趋势判断】")
            if self.static_pe > self.ttm_pe > self.forecast_pe_median:
                lines.append("  ★ 完美成长股: 业绩加速中 (静态PE > TTM > 预测PE)")
            elif self.static_pe < self.ttm_pe < self.forecast_pe_median:
                lines.append("  ⚠ 衰退信号: 业绩恶化中 (静态PE < TTM < 预测PE)")
        
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


def _get_pro():
    """获取Tushare Pro API实例"""
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("未设置 TUSHARE_TOKEN 环境变量")
    ts.set_token(token)
    return ts.pro_api()


def get_latest_price_and_market_cap(pro, ts_code: str) -> tuple[float, float, str]:
    """获取最新价格和市值"""
    # 获取最新交易日
    today = datetime.now().strftime("%Y%m%d")
    df = pro.daily(ts_code=ts_code, start_date="20241201", end_date=today)
    if df.empty:
        raise ValueError(f"无法获取 {ts_code} 的最新交易数据")
    
    latest = df.sort_values("trade_date", ascending=False).iloc[0]
    close = float(latest["close"])
    trade_date = str(latest["trade_date"])
    
    # 获取总股本(万股)
    df_basic = pro.daily_basic(ts_code=ts_code, trade_date=trade_date, fields="ts_code,trade_date,total_share,total_mv")
    if df_basic.empty or pd.isna(df_basic.iloc[0]["total_mv"]):
        # 计算市值: 收盘价 * 总股本(万股) / 10000 = 亿元
        total_share = float(df_basic.iloc[0]["total_share"])
        market_cap = close * total_share / 10000
    else:
        market_cap = float(df_basic.iloc[0]["total_mv"]) / 10000  # 转为亿元
    
    return close, market_cap, trade_date


def get_static_pe(pro, ts_code: str, current_year: int, market_cap: float) -> Optional[float]:
    """计算静态市盈率 (LYR): 市值 / 上一年度净利润"""
    last_year = current_year - 1
    period = f"{last_year}1231"
    
    try:
        df = pro.income(ts_code=ts_code, period=period, fields="ts_code,end_date,n_income")
        if df.empty or pd.isna(df.iloc[0]["n_income"]):
            return None
        
        net_profit = float(df.iloc[0]["n_income"]) / 100000000  # 转为亿元
        if net_profit <= 0:
            return None
        
        return market_cap / net_profit
    except Exception as e:
        print(f"获取静态PE失败: {e}")
        return None


def get_ttm_pe(pro, ts_code: str, trade_date: str) -> Optional[float]:
    """获取TTM市盈率 (直接从daily_basic获取)"""
    try:
        df = pro.daily_basic(ts_code=ts_code, trade_date=trade_date, fields="ts_code,trade_date,pe_ttm")
        if df.empty or pd.isna(df.iloc[0]["pe_ttm"]):
            return None
        return float(df.iloc[0]["pe_ttm"])
    except Exception as e:
        print(f"获取TTM PE失败: {e}")
        return None


def _q3_q4_share_median(pro, ts_code: str, current_year: int, lookback_years: int = 5) -> Optional[float]:
    """计算历史Q4占比中位数: share = (FY - C3) / FY over last N years.

    返回 Q4在全年中的占比中位数 (范围 0..1)。
    需要 FY>0 且 FY>C3，筛掉异常/缺失值。
    """
    shares: list[float] = []
    start_year = max(2009, current_year - lookback_years)
    for y in range(current_year - 1, start_year - 1, -1):
        try:
            df_c3 = pro.income(ts_code=ts_code, period=f"{y}0930", fields="n_income")
            df_fy = pro.income(ts_code=ts_code, period=f"{y}1231", fields="n_income")
            if df_c3.empty or df_fy.empty:
                continue
            c3 = float(df_c3.iloc[0]["n_income"]) / 100000000
            fy = float(df_fy.iloc[0]["n_income"]) / 100000000
            if pd.isna(c3) or pd.isna(fy) or fy <= 0 or c3 < 0 or fy <= c3:
                continue
            share = (fy - c3) / fy
            if 0 < share < 1:
                shares.append(share)
        except Exception:
            continue
    if not shares:
        return None
    return float(pd.Series(shares).median())


def get_linear_extrapolate_pe(pro, ts_code: str, market_cap: float) -> tuple[Optional[float], Optional[str]]:
    """计算线性外推PE: 基于最新季报，优先历史权重法(季节性)，失败则简单年化。

    - 优先：若已披露当年Q3，使用“历史Q4占比(近N年中位数)”推算FY_y = C3_y / (1 - share)。
    - 次之：若不可得，使用“去年单年比例” S4_l/C3_l。
    - 回退：Q1×4，Q2×2，Q3用(前三季累计/3×4)，Q4直接用FY。
    """
    today = datetime.now()
    current_year = today.year
    current_month = today.month

    # 1) 尝试Q3历史权重法（多年中位数）
    try:
        df_y_q3 = pro.income(ts_code=ts_code, period=f"{current_year}0930", fields="n_income")
        if not df_y_q3.empty and pd.notna(df_y_q3.iloc[0]["n_income"]):
            C3_y = float(df_y_q3.iloc[0]["n_income"]) / 100000000
            # 历史Q4占比中位数
            share = _q3_q4_share_median(pro, ts_code, current_year, lookback_years=5)
            if share is not None and 0 < share < 1:
                FY_y = C3_y / (1.0 - share)
                if FY_y > 0:
                    pe = market_cap / FY_y
                    return pe, f"{current_year}Q3(seasonal-median)"
            # 若不可得，用去年单年比例法
            df_l_q3 = pro.income(ts_code=ts_code, period=f"{current_year-1}0930", fields="n_income")
            df_l_fy = pro.income(ts_code=ts_code, period=f"{current_year-1}1231", fields="n_income")
            if (not df_l_q3.empty and pd.notna(df_l_q3.iloc[0]["n_income"]) and
                not df_l_fy.empty and pd.notna(df_l_fy.iloc[0]["n_income"])):
                C3_l = float(df_l_q3.iloc[0]["n_income"]) / 100000000
                FY_l = float(df_l_fy.iloc[0]["n_income"]) / 100000000
                if C3_l > 0 and FY_l > C3_l:
                    S4_l = FY_l - C3_l
                    S4_y = S4_l * (C3_y / C3_l)
                    FY_y = C3_y + S4_y
                    if FY_y > 0:
                        pe = market_cap / FY_y
                        return pe, f"{current_year}Q3(seasonal-lastyear)"
    except Exception:
        pass

    # 2) 回退方案：找出可用的最近季度并简单年化
    possible_quarters: list[tuple[int, str, str]] = []
    if current_month >= 11:
        possible_quarters.append((current_year, "0930", "Q3"))
    if current_month >= 9:
        possible_quarters.append((current_year, "0630", "Q2"))
    if current_month >= 5:
        possible_quarters.append((current_year, "0331", "Q1"))
    # 永远可以回退到去年年报
    possible_quarters.append((current_year - 1, "1231", "Q4"))

    for year, end_date, quarter_name in possible_quarters:
        try:
            df = pro.income(ts_code=ts_code, period=f"{year}{end_date}", fields="n_income")
            if df.empty or pd.isna(df.iloc[0]["n_income"]):
                continue
            cumulative = float(df.iloc[0]["n_income"]) / 100000000
            if cumulative <= 0:
                continue

            if quarter_name == "Q1":
                annualized = cumulative * 4
                return market_cap / annualized, f"{year}Q1(simple)"
            if quarter_name == "Q2":
                annualized = cumulative * 2
                return market_cap / annualized, f"{year}Q2(simple)"
            if quarter_name == "Q3":
                annualized = cumulative / 3 * 4
                return market_cap / annualized, f"{year}Q3(simple)"
            # Q4 年报：直接用全年
            return market_cap / cumulative, f"{year}Q4"
        except Exception:
            continue

    return None, None


def get_forecast_pe(pro, ts_code: str, market_cap: float, forecast_year: int) -> tuple[Optional[float], Optional[float]]:
    """计算机构预测PE (平均值和中位数)"""
    # 尝试从 report_rc 获取EPS预测
    try:
        df = pro.report_rc(ts_code=ts_code)
        if df is None or df.empty or "eps" not in df.columns:
            return None, None
        
        # 提取目标年份的预测
        df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
        df = df.dropna(subset=["eps"])
        
        if df.empty:
            return None, None
        
        # 优先使用目标年份Q4的数据
        target_q = f"{forecast_year}Q4"
        if "quarter" in df.columns:
            df_year = df[df["quarter"].astype(str).str.startswith(str(forecast_year))]
            if not df_year.empty:
                df_q4 = df_year[df_year["quarter"].astype(str) == target_q]
                eps_list = df_q4["eps"].tolist() if not df_q4.empty else df_year["eps"].tolist()
            else:
                # 使用最新的EPS
                eps_list = df.head(20)["eps"].tolist()
        else:
            eps_list = df.head(20)["eps"].tolist()
        
        if not eps_list:
            return None, None
        
        # 计算平均值和中位数
        eps_mean = sum(eps_list) / len(eps_list)
        eps_median = sorted(eps_list)[len(eps_list) // 2]
        
        # 获取总股本
        df_basic = pro.daily_basic(ts_code=ts_code, fields="ts_code,total_share", limit=1)
        if df_basic.empty:
            return None, None
        
        total_share = float(df_basic.iloc[0]["total_share"])  # 万股
        
        # 预测净利润(亿元) = EPS * 总股本(万股) / 10000
        forecast_profit_mean = eps_mean * total_share / 10000
        forecast_profit_median = eps_median * total_share / 10000
        
        if forecast_profit_mean <= 0 or forecast_profit_median <= 0:
            return None, None
        
        pe_mean = market_cap / forecast_profit_mean
        pe_median = market_cap / forecast_profit_median
        
        return pe_mean, pe_median
        
    except Exception as e:
        print(f"获取机构预测PE失败: {e}")
        return None, None


def calculate_all_pe_ratios(ts_code: str, forecast_year: Optional[int] = None) -> PERatios:
    """计算所有市盈率指标"""
    if forecast_year is None:
        forecast_year = datetime.now().year
    
    pro = _get_pro()
    
    # 获取基础数据
    close, market_cap, trade_date = get_latest_price_and_market_cap(pro, ts_code)
    current_year = int(trade_date[:4])
    
    # 计算各维度PE
    static_pe = get_static_pe(pro, ts_code, current_year, market_cap)
    ttm_pe = get_ttm_pe(pro, ts_code, trade_date)
    linear_pe, latest_quarter = get_linear_extrapolate_pe(pro, ts_code, market_cap)
    forecast_pe_mean, forecast_pe_median = get_forecast_pe(pro, ts_code, market_cap, forecast_year)
    
    return PERatios(
        ts_code=ts_code,
        trade_date=trade_date,
        close_price=close,
        market_cap=market_cap,
        static_pe=static_pe,
        ttm_pe=ttm_pe,
        linear_extrapolate_pe=linear_pe,
        latest_quarter=latest_quarter,
        forecast_pe_mean=forecast_pe_mean,
        forecast_pe_median=forecast_pe_median,
    )
