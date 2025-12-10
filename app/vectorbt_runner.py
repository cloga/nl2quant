import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Optional
from app.dca_backtest_engine import DCABacktestEngine


def run_vectorbt_dca_backtest(
    code: str,
    monthly_investment: float,
    start_date: str,
    end_date: str,
    rebalance_freq: str = "M",
    freq_day: Optional[str] = None,
    commission_rate: float = 0.0,
    min_commission: float = 0.0,
    slippage: float = 0.0,
    initial_capital: float = 0.0,
    max_total_investment: float = 0.0,
) -> Dict[str, Any]:
    """Run a plain DCA using vectorbt for cross-validation.

    Notes:
        - Supports plain fixed-amount investing only.
        - Does not implement take-profit / smart PE/PB.
    """

    engine = DCABacktestEngine()
    price_series = engine.fetch_etf_close(code, start_date, end_date)
    if price_series is None or price_series.empty:
        raise ValueError("未获取到价格数据")

    price_series = price_series.sort_index()
    all_dates = price_series.index

    invest_dates = DCABacktestEngine._generate_investment_dates(all_dates, rebalance_freq, freq_day)
    entries = pd.Series(False, index=price_series.index)
    entries.loc[price_series.index.isin(invest_dates)] = True
    entries = entries & price_series.notna()

    invest_count = entries.sum()
    size_amount = pd.Series(0.0, index=price_series.index)
    size_amount[entries] = monthly_investment

    est_total = monthly_investment * invest_count + initial_capital
    init_cash = max_total_investment if max_total_investment > 0 else (est_total * 1.1 if est_total > 0 else 1_000_000)

    # Use amount-based orders; fees approximate commission+slippage as rate on notional
    portfolio = vbt.Portfolio.from_orders(
        close=price_series,
        size=size_amount,
        size_type="amount",
        direction="longonly",
        fees=commission_rate + slippage,
        fixed_fees=min_commission if min_commission > 0 else 0.0,
        init_cash=init_cash,
        cash_sharing=True,
        when="start",
    )

    equity_curve = portfolio.value()
    returns = equity_curve.pct_change().dropna()

    total_invested = float(est_total)
    final_value = float(equity_curve.iloc[-1]) if len(equity_curve) else 0.0
    total_return_pct = (final_value - total_invested) / total_invested * 100 if total_invested > 0 else np.nan

    days = (equity_curve.index[-1] - equity_curve.index[0]).days if len(equity_curve) > 1 else 0
    cagr_pct = ((final_value / total_invested) ** (365 / days) - 1) * 100 if days > 0 and total_invested > 0 else np.nan

    vol_pct = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else np.nan
    sharpe = (cagr_pct / 100) / (vol_pct / 100) if vol_pct not in [0, np.nan, None] else np.nan

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown_pct = drawdown.min() * 100 if len(drawdown) else np.nan

    calmar = (cagr_pct / abs(max_drawdown_pct)) if max_drawdown_pct and not pd.isna(max_drawdown_pct) and max_drawdown_pct != 0 else np.nan

    metrics = {
        "total_invested": total_invested,
        "final_value": final_value,
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "volatility_pct": vol_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": np.nan,
        "max_drawdown_pct": max_drawdown_pct,
        "calmar_ratio": calmar,
        "total_days": days,
    }

    diagnostics = {
        "price_rows": len(price_series),
        "price_start": price_series.index[0] if len(price_series) else None,
        "price_end": price_series.index[-1] if len(price_series) else None,
        "investment_dates": int(invest_count),
    }

    return {
        "equity_curve": equity_curve,
        "metrics": metrics,
        "price_series": price_series,
        "transactions": portfolio.orders.records_readable if hasattr(portfolio, "orders") else pd.DataFrame(),
        "diagnostics": diagnostics,
    }
