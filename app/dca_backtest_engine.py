"""
DCA (Dollar Cost Averaging) Backtest Engine
============================================
Provides core functionality for simulating regular investment strategies.
Supports multiple rebalancing frequencies and portfolio compositions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import tushare as ts
from app.config import Config


class DCABacktestEngine:
    """Engine for running DCA (定投) backtests on ETF baskets."""

    # Simple in-memory caches for fetched market/valuation data
    PRICE_CACHE: Dict = {}
    VALUATION_CACHE: Dict = {}

    def __init__(self, tushare_token: str = None):
        """Initialize the DCA backtest engine."""
        self.token = tushare_token or Config.TUSHARE_TOKEN
        if not self.token:
            raise ValueError("Tushare token not found in config or parameters")
        self.pro = ts.pro_api(self.token)
        self.max_retries = 3
        self.retry_delay = 2
        self.price_df = None  # Compatibility hook for callers that expect cached price data
        self.last_price_cache_hit = False
        self.last_valuation_cache_hit = False

    @staticmethod
    def candidate_ts_codes(code: str) -> List[str]:
        """Generate plausible Tushare ts_code variants for an ETF code."""
        code = code.strip().upper()
        if "." in code:
            return [code]

        if len(code) == 5:
            code6 = code + "0"
        else:
            code6 = code

        guesses = []
        if code6[0] in {"5", "6"}:
            guesses.append(f"{code6}.SH")
            guesses.append(f"{code6}.SZ")
        elif code6[0] in {"1", "3"}:
            guesses.append(f"{code6}.SZ")
            guesses.append(f"{code6}.SH")
        else:
            guesses.append(f"{code6}.SH")
            guesses.append(f"{code6}.SZ")

        seen = set()
        dedup = []
        for g in guesses:
            if g not in seen:
                seen.add(g)
                dedup.append(g)
        return dedup

    def fetch_with_chunks(
        self, api_name: str, ts_code: str, sd: str, ed: str
    ) -> pd.DataFrame:
        """Fetch data with chunking to handle large date ranges."""
        frames = []
        sd_dt = datetime.strptime(sd, "%Y%m%d")
        ed_dt = datetime.strptime(ed, "%Y%m%d")
        cursor = sd_dt

        while cursor <= ed_dt:
            chunk_end = min(cursor + timedelta(days=365), ed_dt)
            chunk_sd = cursor.strftime("%Y%m%d")
            chunk_ed = chunk_end.strftime("%Y%m%d")
            api_fn = getattr(self.pro, api_name)
            df_chunk = api_fn(ts_code=ts_code, start_date=chunk_sd, end_date=chunk_ed)
            if df_chunk is not None and not df_chunk.empty:
                frames.append(df_chunk)
            cursor = chunk_end + timedelta(days=1)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def fetch_etf_close(
        self, code: str, start_date: str, end_date: str
    ) -> Optional[pd.Series]:
        """Fetch asset close prices (stocks/ETFs/indices) with fallbacks and retries."""
        import time
        last_exc = None
        self.last_price_cache_hit = False
        
        # Normalize code to ts_code format for consistent cache key
        ts_codes = self.candidate_ts_codes(code)
        normalized_code = ts_codes[0] if ts_codes else code
        
        cache_key = (normalized_code, start_date, end_date)
        if cache_key in self.PRICE_CACHE:
            self.last_price_cache_hit = True
            return self.PRICE_CACHE[cache_key].copy()

        # Try multiple data sources: fund (ETF), stock, and index
        api_sources = [
            ("fund_daily", "fund"),    # ETF funds
            ("daily", "stock"),        # Stocks
            ("index_daily", "index"),  # Indices
        ]

        for ts_code in ts_codes:
            for api_name, source_type in api_sources:
                for attempt in range(self.max_retries):
                    try:
                        if source_type == "index":
                            # For indices, ts_code format might be different
                            # Try original code with common suffixes
                            index_codes = [code, f"{code}.CSI", f"{code}.SH", f"{code}.SZ"]
                            for idx_code in index_codes:
                                try:
                                    df = self.fetch_with_chunks(api_name, idx_code, start_date, end_date)
                                    if df is not None and not df.empty:
                                        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                                        df = df.sort_values("trade_date")
                                        df = df.drop_duplicates(subset=["trade_date"], keep="first")
                                        series = df.set_index("trade_date")["close"]
                                        # Cache with the normalized key, not the original code
                                        self.PRICE_CACHE[cache_key] = series
                                        return series.copy()
                                except:
                                    continue
                        else:
                            df = self.fetch_with_chunks(api_name, ts_code, start_date, end_date)
                            if df is not None and not df.empty:
                                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                                df = df.sort_values("trade_date")
                                df = df.drop_duplicates(subset=["trade_date"], keep="first")
                                series = df.set_index("trade_date")["close"]
                                # Cache with the normalized key, not the original code
                                self.PRICE_CACHE[cache_key] = series
                                return series.copy()
                    except Exception as e:
                        last_exc = e
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                        continue

        return None

    def build_price_frame(
        self, codes: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Build aligned price dataframe for multiple ETFs."""
        prices = {}
        for code in codes:
            series = self.fetch_etf_close(code, start_date, end_date)
            if series is not None and not series.empty:
                prices[code] = series

        if not prices:
            raise ValueError(f"Failed to fetch any ETF data for codes: {codes}")

        df = pd.DataFrame(prices)
        df = df.dropna()  # Keep only dates with all prices available
        return df

    def run_dca_backtest(
        self,
        codes: List[str],
        weights: Dict[str, float],
        monthly_investment: float,
        start_date: str,
        end_date: str,
        rebalance_freq: str = "M",  # M=Monthly, Q=Quarterly, Y=Yearly
    ) -> Dict:
        """
        Run a DCA backtest simulation.

        Args:
            codes: List of ETF codes
            weights: Dict of {code: weight} for allocation
            monthly_investment: Amount to invest each period
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            rebalance_freq: M (monthly), Q (quarterly), Y (yearly)

        Returns:
            Dictionary containing:
            - equity_curve: Daily portfolio values
            - metrics: Performance metrics (CAGR, Sharpe, etc.)
            - positions: Final position holdings
            - transactions: All investment transactions
        """
        # Fetch price data
        price_df = self.build_price_frame(codes, start_date, end_date)
        self.price_df = price_df  # Expose for any downstream consumers

        # Initialize tracking
        portfolio_value = []
        portfolio_dates = []
        holdings = {code: 0 for code in codes}
        cost_basis = {code: 0.0 for code in codes}
        transactions = []

        # Convert dates
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        # Generate investment dates
        if rebalance_freq == "M":
            freq_str = "ME"
        elif rebalance_freq == "Q":
            freq_str = "QE"
        elif rebalance_freq == "Y":
            freq_str = "YE"
        else:
            freq_str = "ME"

        # Create date range for investments
        investment_dates = pd.date_range(
            start=price_df.index.min(),
            end=price_df.index.max(),
            freq=freq_str,
        )

        # Simulate daily portfolio value
        for trade_date in price_df.index:
            current_prices = price_df.loc[trade_date]

            # Check if it's an investment date
            if trade_date in investment_dates:
                # Allocate monthly investment according to weights
                for code in codes:
                    if current_prices[code] > 0:
                        investment_amount = monthly_investment * weights.get(code, 0)
                        shares = investment_amount / current_prices[code]
                        holdings[code] += shares
                        cost_basis[code] += investment_amount
                        transactions.append(
                            {
                                "date": trade_date,
                                "code": code,
                                "action": "BUY",
                                "price": current_prices[code],
                                "shares": shares,
                                "amount": investment_amount,
                            }
                        )

            # Calculate daily portfolio value
            portfolio_val = 0
            for code in codes:
                if code in current_prices and current_prices[code] > 0:
                    portfolio_val += holdings[code] * current_prices[code]

            portfolio_value.append(portfolio_val)
            portfolio_dates.append(trade_date)

        # Build equity curve
        equity_curve = pd.Series(portfolio_value, index=portfolio_dates)

        # Calculate metrics
        metrics = self._compute_metrics(equity_curve, cost_basis)

        # Prepare final positions
        final_prices = price_df.iloc[-1]
        positions = {}
        for code in codes:
            if holdings[code] > 0:
                positions[code] = {
                    "shares": holdings[code],
                    "price": final_prices[code],
                    "value": holdings[code] * final_prices[code],
                    "cost_basis": cost_basis[code],
                    "gain": (holdings[code] * final_prices[code]) - cost_basis[code],
                    "gain_pct": (
                        (holdings[code] * final_prices[code] - cost_basis[code])
                        / cost_basis[code]
                        * 100
                        if cost_basis[code] > 0
                        else 0
                    ),
                }

        return {
            "equity_curve": equity_curve,
            "metrics": metrics,
            "positions": positions,
            "transactions": pd.DataFrame(transactions),
            "total_invested": sum(cost_basis.values()),
            "final_value": equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
        }

    @staticmethod
    def _compute_metrics(
        equity_curve: pd.Series,
        cost_basis: Optional[Dict] = None,
        total_invested: Optional[float] = None,
    ) -> Dict:
        """Compute performance metrics from an equity curve.

        Args:
            equity_curve: Time-indexed equity series.
            cost_basis: Optional per-asset cost basis dictionary.
            total_invested: Optional explicit total invested override.
        """
        if len(equity_curve) < 2:
            return {}

        if total_invested is None:
            total_invested = sum(cost_basis.values()) if cost_basis else 0.0

        final_value = equity_curve.iloc[-1]
        total_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0

        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        cagr = (final_value / total_invested) ** (1 / years) - 1 if years > 0 and total_invested > 0 else 0

        returns = equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)

        sharpe = (cagr - 0) / volatility * np.sqrt(1) if volatility > 0 else 0

        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()

        return {
            "total_invested": total_invested,
            "final_value": final_value,
            "total_return_pct": total_return * 100,
            "total_days": days,
            "years": years,
            "cagr_pct": cagr * 100,
            "volatility_pct": volatility * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown * 100,
        }

    @staticmethod
    def compute_metrics_from_equity(equity_curve: pd.Series, total_invested: float) -> Dict:
        """Public helper to compute metrics when only equity curve and invested amount are known."""
        return DCABacktestEngine._compute_metrics(
            equity_curve=equity_curve,
            cost_basis=None,
            total_invested=total_invested,
        )

    @staticmethod
    def get_monthly_snapshots(
        equity_curve: pd.Series, positions_history: List[Dict]
    ) -> pd.DataFrame:
        """Generate month-end snapshots of portfolio."""
        monthly = equity_curve.resample("ME").last()
        snapshots = []
        for date, value in monthly.items():
            snapshots.append({"date": date, "value": value})
        return pd.DataFrame(snapshots)

    def fetch_valuation_data(
        self, ts_code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch PE-TTM and PB valuation data from Tushare."""
        try:
            self.last_valuation_cache_hit = False
            # Normalize ts_code
            if "." not in ts_code:
                codes = self.candidate_ts_codes(ts_code)
                ts_code = codes[0]

            cache_key = (ts_code, start_date, end_date)
            if cache_key in self.VALUATION_CACHE:
                self.last_valuation_cache_hit = True
                return self.VALUATION_CACHE[cache_key].copy()

            df = self.fetch_with_chunks(
                "stock_valuation", ts_code, start_date, end_date
            )
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
                df = df.sort_values("trade_date")
                df = df.drop_duplicates(subset=["trade_date"], keep="first")
                df = df.set_index("trade_date")[ ["pe_ttm", "pb"] ]
                self.VALUATION_CACHE[cache_key] = df
                return df.copy()
        except Exception as e:
            print(f"Warning: Failed to fetch valuation data for {ts_code}: {e}")
        return None

    @staticmethod
    def _generate_investment_dates(all_dates, rebalance_freq, freq_day=None):
        """
        Generate investment dates based on frequency and specific day.
        
        Args:
            all_dates: All available trading dates
            rebalance_freq: 'D' (daily), 'W' (weekly), 'M' (monthly)
            freq_day: For W: weekday name; For M: day of month (1-31)
        """
        if rebalance_freq == "D":
            return all_dates  # Every trading day
        
        elif rebalance_freq == "W":
            # Weekly on specific weekday
            weekday_map = {
                "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4
            }
            target_weekday = weekday_map.get(freq_day, 0) if freq_day else 0
            return [d for d in all_dates if d.weekday() == target_weekday]
        
        elif rebalance_freq == "M":
            # Monthly on specific day
            target_day = int(freq_day) if freq_day else 1
            selected = []
            for year_month, group in pd.Series(all_dates).groupby([all_dates.year, all_dates.month]):
                # Find trading day closest to target day
                days_in_month = group.values
                target_candidates = [d for d in days_in_month if d.day >= target_day]
                if target_candidates:
                    selected.append(target_candidates[0])
                elif len(days_in_month) > 0:
                    selected.append(days_in_month[-1])  # Last trading day if target > month end
            return selected
        
        return all_dates

    def run_smart_dca_backtest(
        self,
        code: str,
        monthly_investment: float,
        start_date: str,
        end_date: str,
        strategy_type: str = "plain",  # plain, smart_pe, smart_pb, value_averaging
        smart_params: Optional[Dict] = None,
        rebalance_freq: str = "M",
        commission_rate: float = 0.0001,
        min_commission: float = 5.0,
        slippage: float = 0.001,
        initial_capital: float = 0.0,
        risk_free_rate: float = 0.025,
        trailing_params: Optional[Dict] = None,
        freq_day: Optional[any] = None,
        max_total_investment: float = 0.0,
    ) -> Dict:
        """
        Run a smart DCA backtest with valuation-based or averaging strategies.

        Args:
            code: Single ETF/index code
            monthly_investment: Base investment amount
            start_date: Start date (YYYYMMDD)
            end_date: End date (YYYYMMDD)
            strategy_type: 'plain', 'smart_pe', 'smart_pb', 'value_averaging'
            smart_params: Dict with 'low_multiple', 'high_multiple', 'lookback_days'
            rebalance_freq: D/W/M
            commission_rate: Brokerage commission rate
            min_commission: Minimum commission per trade
            slippage: Execution slippage (%)
            initial_capital: Initial account capital for lump-sum comparison
            risk_free_rate: Annual return rate for idle cash
            trailing_params: Dict for take-profit settings
            freq_day: Specific day for weekly/monthly investing
            max_total_investment: Maximum cumulative investment (0 = unlimited)

        Returns:
            Dict with backtest results, metrics, and strategy analysis
        """
        # Fetch price data
        price_series = self.fetch_etf_close(code, start_date, end_date)
        if price_series is None or price_series.empty:
            raise ValueError(f"Failed to fetch price data for {code}")

        # Keep a dataframe copy for consumers that rely on price_df
        try:
            self.price_df = price_series.to_frame(name=code)
        except Exception:
            self.price_df = None

        diagnostics = {
            "price_rows": len(price_series),
            "price_start": price_series.index.min(),
            "price_end": price_series.index.max(),
            "price_cache_hit": self.last_price_cache_hit,
        }

        # Fetch valuation data if needed
        valuation_df = None
        if strategy_type in ["smart_pe", "smart_pb"]:
            valuation_df = self.fetch_valuation_data(code, start_date, end_date)
            diagnostics["valuation_rows"] = 0 if valuation_df is None else len(valuation_df)
            diagnostics["valuation_cache_hit"] = self.last_valuation_cache_hit
        else:
            diagnostics["valuation_rows"] = 0
            diagnostics["valuation_cache_hit"] = False

        # Set default smart parameters
        if smart_params is None:
            smart_params = {
                "low_multiple": 2.0,
                "high_multiple": 0.5,
                "lookback_days": 252 * 5,  # 5 years
            }

        # Initialize tracking
        portfolio_value = []
        portfolio_dates = []
        holdings = 0
        cost_basis = 0.0  # Total capital invested in current round (external only)
        holdings_cost = 0.0  # Cost basis for holdings only (for take-profit calculation)
        cash = initial_capital  # Track idle cash for current round
        transactions = []
        strategy_metrics = []

        # Track lifetime external capital (初始本金 + 所有外部追加), 不因止盈重置
        initial_external_capital = initial_capital
        external_contributions = 0.0
        
        # Take-profit tracking
        is_stopped_out = False
        stop_out_date = None
        stop_out_price = None
        highest_value = 0
        take_profit_activated = False

        # Generate investment dates based on frequency
        investment_dates = self._generate_investment_dates(
            price_series.index, rebalance_freq, freq_day
        )
        diagnostics["investment_dates"] = len(investment_dates)

        # Simulate daily portfolio value
        for trade_date in price_series.index:
            current_price = price_series.loc[trade_date]
            
            # Apply idle cash return (daily compounding)
            if cash > 0 and risk_free_rate > 0:
                daily_rate = risk_free_rate / 365
                cash *= (1 + daily_rate)

            # Check if it's an investment date and not stopped out
            if trade_date in investment_dates and not is_stopped_out:
                # Calculate investment amount based on strategy
                investment_amount = self._calculate_investment_amount(
                    trade_date,
                    monthly_investment,
                    price_series,
                    valuation_df,
                    strategy_type,
                    smart_params,
                )

                if investment_amount > 0 and current_price > 0:
                    # Allow reinvesting existing cash even after hitting external cap.
                    # External cap only limits new external contributions, not recycled cash/profits.
                    external_cap = max_total_investment if max_total_investment > 0 else float("inf")
                    external_spent = initial_external_capital + external_contributions
                    remaining_external = max(external_cap - external_spent, 0)

                    # If no external room and no cash, skip this date
                    if remaining_external <= 0 and cash <= 0:
                        pass
                    else:
                        # How much external capital is needed for this order?
                        external_needed = max(investment_amount - cash, 0)
                        external_used = min(external_needed, remaining_external)

                        # Final investable amount = current cash + allowed external
                        investable_amount = cash + external_used
                        actual_investment = min(investment_amount, investable_amount)

                        if actual_investment > 0:
                            # Add only the external portion to cost_basis
                            if external_used > 0:
                                cash += external_used
                                cost_basis += external_used
                                external_contributions += external_used

                            commission = max(
                                actual_investment * commission_rate, min_commission
                            )
                            execution_price = current_price * (1 + slippage)

                            shares = (actual_investment - commission) / execution_price
                            holdings += shares
                            holdings_cost += actual_investment
                            cash -= actual_investment
                            net_invested = actual_investment - commission

                            transactions.append(
                                {
                                    "date": trade_date,
                                    "action": "BUY",
                                    "price": current_price,
                                    "execution_price": execution_price,
                                    "investment": actual_investment,
                                    "commission": commission,
                                    "net_invested": net_invested,
                                    "shares": shares,
                                }
                            )

                    # Record strategy metric for this date
                    if strategy_type == "smart_pe" and valuation_df is not None:
                        if trade_date in valuation_df.index:
                            pe = valuation_df.loc[trade_date, "pe_ttm"]
                            strategy_metrics.append(
                                {"date": trade_date, "pe": pe, "action": "BUY"}
                            )
                    elif strategy_type == "smart_pb" and valuation_df is not None:
                        if trade_date in valuation_df.index:
                            pb = valuation_df.loc[trade_date, "pb"]
                            strategy_metrics.append(
                                {"date": trade_date, "pb": pb, "action": "BUY"}
                            )

            # Calculate daily portfolio value (holdings + cash)
            portfolio_val = (holdings * current_price if current_price > 0 else 0) + cash
            
            # Take-profit logic
            if trailing_params and holdings > 0:
                # Calculate return based on selected method
                return_calc_method = trailing_params.get("return_calc_method", "holdings_only")
                holdings_value = holdings * current_price if current_price > 0 else 0
                
                if return_calc_method == "holdings_only":
                    # Holdings-only return (default)
                    current_return = (holdings_value - holdings_cost) / holdings_cost if holdings_cost > 0 else 0
                else:
                    # Total portfolio return (including cash)
                    total_invested = cost_basis + initial_capital
                    current_return = (portfolio_val - total_invested) / total_invested if total_invested > 0 else 0
                
                # Update highest value
                if portfolio_val > highest_value:
                    highest_value = portfolio_val
                
                # Check take-profit conditions
                if trailing_params.get("mode") == "target":
                    target_return = trailing_params.get("target_return", 0.5)
                    if current_return >= target_return:
                        # Sell all
                        sell_value = holdings * current_price
                        commission = max(sell_value * commission_rate, min_commission)
                        cash += (sell_value - commission)
                        transactions.append({
                            "date": trade_date,
                            "action": "SELL_ALL",
                            "price": current_price,
                            "shares": holdings,
                            "proceeds": sell_value - commission,
                            "commission": commission,
                        })
                        holdings = 0
                        holdings_cost = 0.0  # Reset holdings cost
                        is_stopped_out = True
                        stop_out_date = trade_date
                        stop_out_price = current_price

                        # Reset baseline: treat realized cash as the new round capital ceiling
                        cost_basis = 0.0
                        initial_capital = cash
                
                elif trailing_params.get("mode") == "trailing":
                    activation_return = trailing_params.get("activation_return", 0.3)
                    drawdown_threshold = trailing_params.get("drawdown_threshold", 0.08)
                    
                    if current_return >= activation_return:
                        take_profit_activated = True
                    
                    if take_profit_activated:
                        drawdown_from_peak = (highest_value - portfolio_val) / highest_value
                        if drawdown_from_peak >= drawdown_threshold:
                            # Sell all
                            sell_value = holdings * current_price
                            commission = max(sell_value * commission_rate, min_commission)
                            cash += (sell_value - commission)
                            transactions.append({
                                "date": trade_date,
                                "action": "SELL_TRAILING",
                                "price": current_price,
                                "shares": holdings,
                                "proceeds": sell_value - commission,
                                "commission": commission,
                            })
                            holdings = 0
                            holdings_cost = 0.0  # Reset holdings cost
                            is_stopped_out = True
                            stop_out_date = trade_date
                            stop_out_price = current_price

                            # Reset baseline: realized cash becomes the new round capital ceiling
                            cost_basis = 0.0
                            initial_capital = cash
            
            # Check re-entry conditions
            if is_stopped_out and trailing_params:
                reentry_mode = trailing_params.get("reentry_mode", "time")
                
                if reentry_mode == "time":
                    days_waited = (trade_date - stop_out_date).days
                    reentry_days = trailing_params.get("reentry_days", 30)
                    if days_waited >= reentry_days:
                        is_stopped_out = False
                        take_profit_activated = False
                        highest_value = portfolio_val
                        
                elif reentry_mode == "price":
                    reentry_drop = trailing_params.get("reentry_drop", 0.15)
                    if current_price <= stop_out_price * (1 - reentry_drop):
                        is_stopped_out = False
                        take_profit_activated = False
                        highest_value = portfolio_val
            
            portfolio_value.append(portfolio_val)
            portfolio_dates.append(trade_date)

        # Build equity curve
        equity_curve = pd.Series(portfolio_value, index=portfolio_dates)

        # Calculate comprehensive metrics using lifetime external capital as denominator
        external_spent = initial_external_capital + external_contributions
        metrics = self._compute_extended_metrics(
            equity_curve, external_spent, benchmark_strategy="plain"
        )

        # Prepare final position (including cash)
        final_price = price_series.iloc[-1]
        final_holdings_value = holdings * final_price if final_price > 0 else 0
        final_total_value = final_holdings_value + cash
        investment_baseline = external_spent
        gain = final_total_value - investment_baseline
        gain_pct = (gain / investment_baseline * 100) if investment_baseline > 0 else 0

        final_position = {
            "code": code,
            "shares": holdings,
            "price": final_price,
            "holdings_value": final_holdings_value,
            "cash": cash,
            "total_value": final_total_value,
            "cost_basis": cost_basis,
            "initial_capital": initial_capital,
            "total_invested": investment_baseline,
            "gain": gain,
            "gain_pct": gain_pct,
        }

        return {
            "equity_curve": equity_curve,
            "metrics": metrics,
            "final_position": final_position,
            "transactions": pd.DataFrame(transactions),
            "strategy_metrics": pd.DataFrame(strategy_metrics) if strategy_metrics else None,
            "strategy_type": strategy_type,
            "total_invested": external_spent,
            "final_value": final_total_value,
            "cash_balance": cash,
            "diagnostics": diagnostics,
            "price_series": price_series,  # Add price data for benchmark comparison
        }

    @staticmethod
    def _calculate_investment_amount(
        trade_date: pd.Timestamp,
        base_amount: float,
        price_series: pd.Series,
        valuation_df: Optional[pd.DataFrame],
        strategy_type: str,
        smart_params: Dict,
    ) -> float:
        """Calculate investment amount based on strategy."""
        if strategy_type == "plain":
            return base_amount

        elif strategy_type == "smart_pe" and valuation_df is not None:
            if trade_date not in valuation_df.index:
                return base_amount

            pe = valuation_df.loc[trade_date, "pe_ttm"]
            if pd.isna(pe) or pe <= 0:
                return base_amount

            # Calculate PE percentile over lookback period
            lookback_days = smart_params.get("lookback_days", 252 * 5)
            lookback_start = trade_date - timedelta(days=lookback_days)
            historical_pe = valuation_df.loc[
                (valuation_df.index >= lookback_start) & (valuation_df.index <= trade_date),
                "pe_ttm",
            ]
            historical_pe = historical_pe.dropna()

            if len(historical_pe) > 0:
                pe_pct = (historical_pe <= pe).sum() / len(historical_pe)
            else:
                pe_pct = 0.5

            # Map percentile to multiplier (lower PE = higher multiplier)
            multiplier = 1.0
            if pe_pct < 0.33:  # Bottom 33% (undervalued)
                multiplier = smart_params.get("low_multiple", 2.0)
            elif pe_pct > 0.67:  # Top 33% (overvalued)
                multiplier = smart_params.get("high_multiple", 0.5)

            return base_amount * multiplier

        elif strategy_type == "smart_pb" and valuation_df is not None:
            if trade_date not in valuation_df.index:
                return base_amount

            pb = valuation_df.loc[trade_date, "pb"]
            if pd.isna(pb) or pb <= 0:
                return base_amount

            # Calculate PB percentile
            lookback_days = smart_params.get("lookback_days", 252 * 5)
            lookback_start = trade_date - timedelta(days=lookback_days)
            historical_pb = valuation_df.loc[
                (valuation_df.index >= lookback_start) & (valuation_df.index <= trade_date),
                "pb",
            ]
            historical_pb = historical_pb.dropna()

            if len(historical_pb) > 0:
                pb_pct = (historical_pb <= pb).sum() / len(historical_pb)
            else:
                pb_pct = 0.5

            multiplier = 1.0
            if pb_pct < 0.33:
                multiplier = smart_params.get("low_multiple", 2.0)
            elif pb_pct > 0.67:
                multiplier = smart_params.get("high_multiple", 0.5)

            return base_amount * multiplier

        elif strategy_type == "value_averaging":
            # Simple value averaging: maintain target growth rate
            target_growth = smart_params.get("target_growth_rate", 0.05)  # 5% annual
            # This is simplified; full implementation would need more logic
            return base_amount

        return base_amount

    @staticmethod
    def _compute_extended_metrics(
        equity_curve: pd.Series,
        total_invested: float,
        benchmark_strategy: str = "plain",
    ) -> Dict:
        """
        Compute extended performance metrics aligned with
        Common Portfolio Evaluation Metrics.
        """
        if len(equity_curve) < 2:
            return {}

        final_value = equity_curve.iloc[-1]
        total_return = (final_value - total_invested) / total_invested if total_invested > 0 else 0

        # Time period
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25

        # CAGR
        cagr = (final_value / total_invested) ** (1 / years) - 1 if years > 0 and total_invested > 0 else 0

        # Volatility
        returns = equity_curve.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # Sharpe Ratio (rf = 0)
        sharpe = (cagr - 0) / volatility * np.sqrt(1) if volatility > 0 else 0

        # Sortino Ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_volatility = (
            downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        )
        sortino = (cagr - 0) / downside_volatility if downside_volatility > 0 else 0

        # Max Drawdown
        cummax = equity_curve.expanding().max()
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win Rate (monthly)
        monthly_returns = equity_curve.resample("ME").last().pct_change().dropna()
        win_rate = (monthly_returns > 0).sum() / len(monthly_returns) if len(monthly_returns) > 0 else 0

        return {
            "total_invested": total_invested,
            "final_value": final_value,
            "total_return_pct": total_return * 100,
            "total_days": days,
            "years": years,
            "cagr_pct": cagr * 100,
            "volatility_pct": volatility * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "max_drawdown_pct": max_drawdown * 100,
            "win_rate_pct": win_rate * 100,
        }
