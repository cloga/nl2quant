"""
Red Chip ETF Grid Trading + Dividend Strategy
==============================================
Strategy: 
- Base position: 50W buy and hold Red Chip ETF (earn 4-6% dividend)
- Grid position: 50W grid trading (buy when -2%, sell when +2%)
- Target: Base dividend 4-6% + Grid profit 4-8% = 8-14% annually

Grid Logic:
- Initial grid: place buy orders at -2%, -4%, -6%, ... from entry price
- Place sell orders at +2%, +4%, +6%, ... from each buy price
- Each grid unit: 5W (10 grids total for 50W)
- Reinvest grid profits back into grid

Note: This is a simplified simulation. Real grid trading requires:
1. Continuous monitoring and order placement
2. Transaction costs (0.03% commission + 0.1% stamp tax on sell)
3. Slippage and liquidity considerations
4. Dynamic grid adjustment based on volatility
"""

import datetime as dt
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

load_dotenv()

START_DATE = "2018-01-01"  # Use recent 7 years for grid effectiveness
TICKER = "h00922.CSI"  # CSI Dividend Total Return
INITIAL_CASH = 1_000_000.0
BASE_POSITION = 500_000.0  # 50% for base holding
GRID_CASH = 500_000.0  # 50% for grid trading
GRID_SPACING = 0.02  # 2% grid spacing
GRID_UNIT = 50_000.0  # 5W per grid
COMMISSION_RATE = 0.0003  # 0.03% commission
STAMP_TAX_RATE = 0.001  # 0.1% stamp tax (sell only)


@dataclass
class GridOrder:
    price: float
    shares: float
    type: str  # 'buy' or 'sell'


def fetch_price(ticker: str, start: str) -> pd.Series:
    """Fetch daily close price via Tushare."""
    start_str = start.replace("-", "")
    end_str = dt.date.today().strftime("%Y%m%d")

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN not set in .env")

    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.index_daily(ts_code=ticker, start_date=start_str, end_date=end_str)

    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    return df["close"].astype(float)


def calculate_cost(amount: float, is_buy: bool) -> float:
    """Calculate transaction cost."""
    commission = amount * COMMISSION_RATE
    stamp_tax = amount * STAMP_TAX_RATE if not is_buy else 0.0
    return commission + stamp_tax


def run_grid_strategy(price: pd.Series) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Run grid trading strategy.
    Returns: (base_equity, grid_equity, stats)
    """
    # Base position: buy and hold
    base_shares = BASE_POSITION / price.iloc[0]
    base_equity = price * base_shares

    # Grid position: initialize
    grid_cash = GRID_CASH
    grid_shares = 0.0
    grid_equity_list = []
    
    # Grid state tracking
    buy_orders: deque = deque()  # Queue of buy prices
    pending_sells: List[GridOrder] = []  # Pending sell orders
    trades: List[dict] = []
    
    # Initialize first grid center at entry price
    entry_price = price.iloc[0]
    
    for date, current_price in price.items():
        # Check if any pending sell orders are triggered
        executed_sells = []
        for sell_order in pending_sells:
            if current_price >= sell_order.price:
                # Execute sell
                proceeds = sell_order.shares * current_price
                cost = calculate_cost(proceeds, is_buy=False)
                grid_cash += proceeds - cost
                grid_shares -= sell_order.shares
                executed_sells.append(sell_order)
                
                # Record profit
                trades.append({
                    'date': date,
                    'type': 'sell',
                    'price': current_price,
                    'shares': sell_order.shares,
                    'profit': proceeds - cost - (sell_order.shares * sell_order.price * (1 + COMMISSION_RATE))
                })
        
        # Remove executed sells
        for sell_order in executed_sells:
            pending_sells.remove(sell_order)
        
        # Check grid buy levels
        # Calculate grid levels below current price
        grid_levels_down = []
        for i in range(1, 11):  # Max 10 levels down
            level_price = entry_price * (1 - GRID_SPACING * i)
            if level_price > current_price * 0.5:  # Don't go below 50% of current
                grid_levels_down.append(level_price)
        
        # Try to buy if price hits a grid level
        for buy_level in grid_levels_down:
            if current_price <= buy_level and grid_cash >= GRID_UNIT:
                # Check if we already bought at this level (avoid duplicate)
                already_bought = any(abs(order.price - buy_level) < 0.01 for order in buy_orders)
                if not already_bought:
                    # Execute buy
                    buy_amount = min(GRID_UNIT, grid_cash)
                    cost = calculate_cost(buy_amount, is_buy=True)
                    shares_bought = (buy_amount - cost) / current_price
                    
                    grid_cash -= buy_amount
                    grid_shares += shares_bought
                    
                    # Record buy
                    buy_orders.append(GridOrder(current_price, shares_bought, 'buy'))
                    trades.append({
                        'date': date,
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares_bought,
                        'amount': buy_amount
                    })
                    
                    # Create corresponding sell order at +2%
                    sell_price = current_price * (1 + GRID_SPACING)
                    pending_sells.append(GridOrder(sell_price, shares_bought, 'sell'))
                    
                    break  # Only one buy per day
        
        # Calculate grid equity
        grid_equity_value = grid_cash + grid_shares * current_price
        grid_equity_list.append((date, grid_equity_value))
    
    grid_equity = pd.Series({d: v for d, v in grid_equity_list})
    
    # Calculate stats
    buy_trades = [t for t in trades if t['type'] == 'buy']
    sell_trades = [t for t in trades if t['type'] == 'sell']
    total_profit = sum(t.get('profit', 0) for t in sell_trades)
    
    stats = {
        'total_trades': len(trades),
        'buy_count': len(buy_trades),
        'sell_count': len(sell_trades),
        'total_profit': total_profit,
        'avg_profit_per_trade': total_profit / len(sell_trades) if sell_trades else 0,
    }
    
    return base_equity, grid_equity, stats


def max_drawdown(series: pd.Series) -> float:
    roll_max = series.cummax()
    drawdown = series / roll_max - 1.0
    return drawdown.min()


def annualized_return(final_value: float, initial: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).days, 1)
    return (final_value / initial) ** (365.0 / days) - 1.0


def main():
    print("Fetching Red Chip ETF data...")
    price = fetch_price(TICKER, START_DATE)
    
    print("Running grid strategy simulation...")
    base_equity, grid_equity, grid_stats = run_grid_strategy(price)
    
    # Combined equity
    total_equity = base_equity + grid_equity
    
    # Pure buy & hold benchmark
    bh_shares = INITIAL_CASH / price.iloc[0]
    bh_equity = price * bh_shares
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("RED CHIP ETF GRID TRADING + DIVIDEND STRATEGY")
    print("=" * 70)
    print(f"Period: {price.index[0].date()} -> {price.index[-1].date()} ({len(price)} days)")
    print(f"Initial Capital: {INITIAL_CASH:,.0f}")
    print(f"Base Position: {BASE_POSITION:,.0f} (Buy & Hold)")
    print(f"Grid Cash: {GRID_CASH:,.0f} (Grid Trading)")
    print()
    
    # Base position metrics
    base_final = base_equity.iloc[-1]
    base_return = base_final / BASE_POSITION - 1.0
    base_cagr = annualized_return(base_final, BASE_POSITION, base_equity.index[0], base_equity.index[-1])
    print("--- Base Position (Buy & Hold 50W) ---")
    print(f"Final Value:       {base_final:>12,.0f}")
    print(f"Total Return:      {base_return:>12.2%}")
    print(f"CAGR:              {base_cagr:>12.2%}")
    print()
    
    # Grid position metrics
    grid_final = grid_equity.iloc[-1]
    grid_return = grid_final / GRID_CASH - 1.0
    grid_cagr = annualized_return(grid_final, GRID_CASH, grid_equity.index[0], grid_equity.index[-1])
    print("--- Grid Position (50W Grid Trading) ---")
    print(f"Final Value:       {grid_final:>12,.0f}")
    print(f"Total Return:      {grid_return:>12.2%}")
    print(f"CAGR:              {grid_cagr:>12.2%}")
    print(f"Total Trades:      {grid_stats['total_trades']:>12}")
    print(f"  - Buys:          {grid_stats['buy_count']:>12}")
    print(f"  - Sells:         {grid_stats['sell_count']:>12}")
    print(f"Total Profit:      {grid_stats['total_profit']:>12,.0f}")
    print(f"Avg Profit/Trade:  {grid_stats['avg_profit_per_trade']:>12,.0f}")
    print()
    
    # Combined metrics
    total_final = total_equity.iloc[-1]
    total_return = total_final / INITIAL_CASH - 1.0
    total_cagr = annualized_return(total_final, INITIAL_CASH, total_equity.index[0], total_equity.index[-1])
    total_mdd = max_drawdown(total_equity)
    print("--- Combined Strategy (Base + Grid) ---")
    print(f"Final Value:       {total_final:>12,.0f}")
    print(f"Total Return:      {total_return:>12.2%}")
    print(f"CAGR:              {total_cagr:>12.2%}")
    print(f"Max Drawdown:      {total_mdd:>12.2%}")
    print()
    
    # Benchmark
    bh_final = bh_equity.iloc[-1]
    bh_return = bh_final / INITIAL_CASH - 1.0
    bh_cagr = annualized_return(bh_final, INITIAL_CASH, bh_equity.index[0], bh_equity.index[-1])
    bh_mdd = max_drawdown(bh_equity)
    print("--- Benchmark (Pure Buy & Hold 100W) ---")
    print(f"Final Value:       {bh_final:>12,.0f}")
    print(f"Total Return:      {bh_return:>12.2%}")
    print(f"CAGR:              {bh_cagr:>12.2%}")
    print(f"Max Drawdown:      {bh_mdd:>12.2%}")
    print()
    
    # Comparison
    edge = total_cagr - bh_cagr
    print("--- Comparison ---")
    print(f"CAGR Edge:         {edge:>12.2%}")
    print(f"MDD Improvement:   {(total_mdd - bh_mdd):>12.2%}")
    print()
    
    # Export
    out = pd.DataFrame({
        'base_position': base_equity,
        'grid_position': grid_equity,
        'combined': total_equity,
        'benchmark': bh_equity,
    })
    out.to_csv("analysis_output_grid_strategy.csv")
    print("Saved equity curves to analysis_output_grid_strategy.csv")
    print("=" * 70)
    
    # Analysis notes
    print("\nNOTES:")
    print("- Grid spacing: 2% (buy at -2%, sell at +2% from each buy)")
    print("- Grid unit: 5W per level (10 levels max)")
    print("- Transaction cost: 0.03% commission + 0.1% stamp tax (sell)")
    print("- Expected dividend from base position: 4-6% annually (not simulated)")
    print("- Grid profit is reinvested back into grid cash")
    print("- Grid works best in ranging/choppy markets")
    print("- In strong trends (up/down), grid may underperform")


if __name__ == "__main__":
    main()
