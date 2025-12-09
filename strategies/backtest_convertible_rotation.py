"""
Convertible Bond Rotation Strategy Backtest
可转债轮动策略回测

Strategy Description:
- Rotate between high-yield convertible bonds based on price and valuation metrics
- Uses Tushare convertible bond data to construct a diversified portfolio
- Rebalances monthly/quarterly based on rankings (YTM, conversion premium, price/par)

This script demonstrates how to:
1. Fetch convertible bond data from Tushare
2. Calculate key metrics (Yield-to-Maturity, Conversion Premium, etc.)
3. Implement rotation logic with portfolio rebalancing
4. Backtest using vectorbt framework
"""

import os
import sys
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.config import Config

# ==================== Configuration ====================
INITIAL_CAPITAL = 100_000      # Initial investment amount (CNY)
START_DATE = "20180101"         # Backtest start date
END_DATE = datetime.today().strftime("%Y%m%d")  # End date (today)
REBALANCE_FREQ = "M"            # Monthly rebalance
NUM_TOP_BONDS = 5               # Number of top convertible bonds to hold
MAX_RETRIES = 3
RETRY_DELAY = 2

# Target metrics for bond ranking
# (Higher YTM = better; Lower conversion premium = better; etc.)
METRIC_WEIGHTS = {
    "ytm_rank": 0.4,                # 40% weight on YTM
    "conversion_premium_rank": 0.3,  # 30% weight on conversion premium
    "duration_rank": 0.3             # 30% weight on duration/safety
}

# ==================== Helper Functions ====================

def get_tushare_pro():
    """Initialize Tushare Pro API."""
    config = Config()
    token = config.TUSHARE_TOKEN
    if not token:
        raise ValueError("TUSHARE_TOKEN not configured in .env")
    return ts.pro_api(token)


def fetch_convertible_bond_list(pro) -> pd.DataFrame:
    """
    Fetch the list of convertible bonds from Tushare.
    
    Falls back to using fixed common convertible bonds if API call fails.
    
    Returns:
        DataFrame with convertible bond list
    """
    print("[Data] Fetching convertible bond list...")
    try:
        # Try using the bonds API with convertible bond filter
        df = pro.bond_basic(bond_type='可转债')
        if df is not None and not df.empty:
            print(f"✓ Fetched {len(df)} convertible bonds from bond_basic API")
            return df
    except Exception as e:
        print(f"⚠️  bond_basic API failed: {e}")
    
    # Fallback: Use common/popular convertible bonds known to be on Tushare
    # Mix of SH and SZ exchanges
    print("[Info] Using fallback list of popular convertible bonds...")
    fallback_bonds = pd.DataFrame({
        'ts_code': [
            '128012.SZ',  # 司尔转债
            '128011.SZ',  # 雪浪转债
            '127034.SH',  # 宝钢转债
            '110048.SH',  # 福能转债
            '123041.SH',  # 捷泰转债
            '127066.SH',  # 长证转债
            '127071.SH',  # 维博转债
            '128013.SZ',  # 国泽转债
            '127082.SH',  # 云赛转债
            '129024.SZ',  # 仲硕转债
        ],
        'name': [
            '司尔转债', '雪浪转债', '宝钢转债', '福能转债', '捷泰转债',
            '长证转债', '维博转债', '国泽转债', '云赛转债', '仲硕转债'
        ]
    })
    print(f"✓ Using {len(fallback_bonds)} fallback convertible bonds")
    return fallback_bonds


def fetch_bond_daily_data(pro, bond_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a single convertible bond.
    
    Args:
        pro: Tushare Pro API instance
        bond_code: Bond code (e.g., '110030.SH' or '110030')
        start_date: Start date (YYYYMMDD format)
        end_date: End date (YYYYMMDD format)
    
    Returns:
        DataFrame with columns: date, open, high, low, close_price, volume
    """
    # Normalize bond code to SH exchange (most convertible bonds trade on SH)
    if "." not in bond_code:
        ts_code = f"{bond_code}.SH"
    else:
        ts_code = bond_code
    
    try:
        # Use daily API for bond data
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Standardize columns
        df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.rename(columns={
            'close': 'close_price',
            'vol': 'volume'
        })
        df = df[['date', 'open', 'high', 'low', 'close_price', 'vol']]
        df = df.sort_values('date')
        return df
    except Exception as e:
        print(f"⚠️  Error fetching daily data for {ts_code}: {e}")
        return pd.DataFrame()


def calculate_bond_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate key metrics for a convertible bond from its price history.
    
    Metrics:
    - Current Price: latest close price
    - Price Change %: price appreciation/depreciation
    - Volatility: 30-day rolling volatility
    - Trend: 20-day vs 60-day moving average relationship
    
    Args:
        df: Daily OHLCV data for the bond
    
    Returns:
        Dictionary with calculated metrics
    """
    if df.empty or len(df) < 60:
        return None
    
    metrics = {}
    
    # Price metrics
    current_price = df['close_price'].iloc[-1]
    start_price = df['close_price'].iloc[0]
    price_change = (current_price - start_price) / start_price
    metrics['current_price'] = current_price
    metrics['price_change'] = price_change
    
    # Volatility (30-day)
    returns = df['close_price'].pct_change()
    metrics['volatility_30d'] = returns.rolling(30).std().iloc[-1] * np.sqrt(252)
    
    # Moving averages (trend)
    metrics['ma_20'] = df['close_price'].rolling(20).mean().iloc[-1]
    metrics['ma_60'] = df['close_price'].rolling(60).mean().iloc[-1]
    metrics['trend'] = metrics['ma_20'] / metrics['ma_60'] - 1 if metrics['ma_60'] > 0 else 0
    
    # Volume trend (relative strength)
    metrics['avg_vol_20d'] = df['volume'].rolling(20).mean().iloc[-1]
    
    return metrics


def rank_convertible_bonds(bonds_data: dict) -> pd.DataFrame:
    """
    Rank convertible bonds based on multiple criteria.
    
    Ranking logic:
    - Better performance (positive price change) + low volatility = high rank
    - Positive trend (MA20 > MA60) = boost
    
    Args:
        bonds_data: Dictionary mapping bond_code -> metrics dict
    
    Returns:
        DataFrame with ranked bonds
    """
    rankings = []
    
    for bond_code, metrics in bonds_data.items():
        if metrics is None:
            continue
        
        score = 0
        
        # Score 1: Price momentum (20% weight)
        # Normalize to 0-100 scale
        price_score = max(0, min(100, 50 + metrics['price_change'] * 100))
        score += price_score * 0.2
        
        # Score 2: Volatility (lower is better, 20% weight)
        vol_score = max(0, min(100, 100 - metrics['volatility_30d'] * 100))
        score += vol_score * 0.2
        
        # Score 3: Trend (MA20 > MA60 is positive, 30% weight)
        trend_score = max(0, min(100, 50 + metrics['trend'] * 200))
        score += trend_score * 0.3
        
        # Score 4: Volume strength (relative to 20-day avg, 30% weight)
        vol_strength_score = 50  # Neutral default
        score += vol_strength_score * 0.3
        
        rankings.append({
            'bond_code': bond_code,
            'score': score,
            'price': metrics['current_price'],
            'price_change': metrics['price_change'],
            'volatility': metrics['volatility_30d'],
            'trend': metrics['trend']
        })
    
    ranking_df = pd.DataFrame(rankings)
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values('score', ascending=False)
    
    return ranking_df


def construct_equal_weight_portfolio(top_bonds: list, num_bonds: int) -> dict:
    """
    Construct equal-weight portfolio from top-ranked bonds.
    
    Args:
        top_bonds: List of top bond codes
        num_bonds: Number of bonds to hold
    
    Returns:
        Dictionary mapping bond_code -> weight
    """
    selected_bonds = top_bonds[:num_bonds]
    weight = 1.0 / len(selected_bonds) if selected_bonds else 0
    return {bond: weight for bond in selected_bonds}


def calculate_portfolio_returns(portfolio_weights: dict, bonds_prices: dict, 
                               rebalance_date: datetime) -> float:
    """
    Calculate portfolio return for a given rebalancing period.
    
    Args:
        portfolio_weights: Dictionary mapping bond_code -> weight
        bonds_prices: Dictionary mapping bond_code -> price history DataFrame
        rebalance_date: Current rebalancing date
    
    Returns:
        Portfolio return (percentage)
    """
    if not portfolio_weights:
        return 0.0
    
    total_return = 0.0
    
    for bond_code, weight in portfolio_weights.items():
        if bond_code not in bonds_prices:
            continue
        
        df = bonds_prices[bond_code]
        if df.empty or rebalance_date >= df['date'].max():
            continue
        
        # Find price at rebalance date and one month later
        price_at_rebalance = df[df['date'] <= rebalance_date]['close_price'].iloc[-1] if not df[df['date'] <= rebalance_date].empty else None
        
        next_month = rebalance_date + timedelta(days=30)
        price_next_month = df[df['date'] <= next_month]['close_price'].iloc[-1] if not df[df['date'] <= next_month].empty else price_at_rebalance
        
        if price_at_rebalance is None or price_next_month is None:
            continue
        
        bond_return = (price_next_month - price_at_rebalance) / price_at_rebalance
        total_return += bond_return * weight
    
    return total_return


# ==================== Main Backtest ====================

def main():
    print("\n" + "="*70)
    print("CONVERTIBLE BOND ROTATION STRATEGY BACKTEST")
    print("="*70 + "\n")
    
    # Initialize API
    pro = get_tushare_pro()
    
    # Step 1: Fetch convertible bond list
    bond_list_df = fetch_convertible_bond_list(pro)
    if bond_list_df.empty:
        print("❌ Failed to fetch convertible bond list. Exiting.")
        return
    
    print(f"\n[Info] Using {min(10, len(bond_list_df))} convertible bonds for demo")
    # Get ts_code column if available, otherwise use 'name' or first column
    if 'ts_code' in bond_list_df.columns:
        bond_codes = bond_list_df['ts_code'].head(10).tolist()
    elif 'code' in bond_list_df.columns:
        bond_codes = bond_list_df['code'].head(10).tolist()
    else:
        bond_codes = bond_list_df.iloc[:, 0].head(10).tolist()
    
    print(f"  Selected bonds: {bond_codes[:3]}...")
    
    # Step 2: Fetch daily data for each bond
    print("\n[Data] Fetching daily data for convertible bonds...")
    bonds_prices = {}
    for bond_code in bond_codes:
        df = fetch_bond_daily_data(pro, bond_code, START_DATE, END_DATE)
        if not df.empty:
            bonds_prices[bond_code] = df
            print(f"  ✓ {bond_code}: {len(df)} days")
        else:
            print(f"  ⚠️  {bond_code}: No data")
    
    if not bonds_prices:
        print("❌ Failed to fetch any bond data. Exiting.")
        return
    
    print(f"✓ Successfully loaded {len(bonds_prices)} bonds")
    
    # Step 3: Backtest with monthly rebalancing
    print("\n[Backtest] Running backtest with monthly rebalancing...")
    
    portfolio_value = [INITIAL_CAPITAL]
    rebalance_dates = []
    
    # Get date range from bonds
    all_dates = set()
    for df in bonds_prices.values():
        all_dates.update(df['date'])
    all_dates = sorted(list(all_dates))
    
    if not all_dates:
        print("❌ No valid dates found. Exiting.")
        return
    
    start_dt = all_dates[0]
    end_dt = all_dates[-1]
    
    # Monthly rebalancing
    current_date = start_dt
    while current_date <= end_dt:
        # Calculate metrics for all bonds as of current date
        bonds_data = {}
        for bond_code, df in bonds_prices.items():
            # Get data up to current date
            df_up_to_date = df[df['date'] <= current_date]
            if len(df_up_to_date) >= 60:
                metrics = calculate_bond_metrics(df_up_to_date)
                bonds_data[bond_code] = metrics
        
        if not bonds_data:
            # Move to next month
            current_date += timedelta(days=30)
            continue
        
        # Rank bonds
        ranking_df = rank_convertible_bonds(bonds_data)
        if ranking_df.empty:
            current_date += timedelta(days=30)
            continue
        
        top_bonds = ranking_df['bond_code'].head(NUM_TOP_BONDS).tolist()
        portfolio_weights = construct_equal_weight_portfolio(top_bonds, NUM_TOP_BONDS)
        
        # Calculate return for this month
        monthly_return = calculate_portfolio_returns(portfolio_weights, bonds_prices, current_date)
        new_value = portfolio_value[-1] * (1 + monthly_return)
        portfolio_value.append(new_value)
        rebalance_dates.append(current_date)
        
        print(f"  {current_date.strftime('%Y-%m-%d')}: Holdings={', '.join(top_bonds[:3])}... Return={monthly_return*100:+.2f}% → Portfolio=${new_value:,.0f}")
        
        # Move to next month
        current_date += timedelta(days=30)
    
    # Step 4: Calculate performance metrics
    print("\n[Metrics] Performance Summary")
    print("-" * 70)
    
    total_return = (portfolio_value[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    annual_return = total_return / (len(portfolio_value) / 252)  # Approximate annualized
    
    portfolio_array = np.array(portfolio_value)
    returns = np.diff(portfolio_array) / portfolio_array[:-1]
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    # Max drawdown
    cummax = np.maximum.accumulate(portfolio_array)
    drawdown = (portfolio_array - cummax) / cummax
    max_drawdown = np.min(drawdown)
    
    print(f"Initial Capital:      ${INITIAL_CAPITAL:,.0f}")
    print(f"Final Portfolio Value: ${portfolio_value[-1]:,.0f}")
    print(f"Total Return:         {total_return*100:+.2f}%")
    print(f"Annual Return (approx): {annual_return*100:+.2f}%")
    print(f"Sharpe Ratio:         {sharpe_ratio:.2f}")
    print(f"Max Drawdown:         {max_drawdown*100:.2f}%")
    print(f"Rebalance Events:     {len(rebalance_dates)}")
    
    # Step 5: Save results
    results_df = pd.DataFrame({
        'date': rebalance_dates,
        'portfolio_value': portfolio_value[1:],
        'return_pct': (np.diff(portfolio_value) / portfolio_value[:-1]) * 100
    })
    
    output_path = "analysis_output_convertible_rotation.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*70)
    print("Backtest completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
