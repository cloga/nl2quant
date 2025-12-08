"""
Convertible Bond Rotation Strategy Backtest (Demo Version with Synthetic Data)
可转债轮动策略回测（演示版本，使用合成数据）

Strategy Description:
- Rotate between high-yield convertible bonds based on price and valuation metrics
- Uses synthetic market data to demonstrate the rotation logic
- Rebalances monthly based on rankings (momentum, volatility, trend)

This script demonstrates:
1. How to structure convertible bond portfolio rotation
2. Ranking methodology for bond selection
3. Backtesting framework with monthly rebalancing
4. Performance metrics calculation (Sharpe Ratio, Drawdown, Returns)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.config import Config

# ==================== Configuration ====================
INITIAL_CAPITAL = 100_000      # Initial investment amount (CNY)
START_DATE_STR = "2021-01-01"  # Backtest start date
END_DATE_STR = "2024-12-31"    # End date
REBALANCE_FREQ = "M"           # Monthly rebalance
NUM_TOP_BONDS = 5              # Number of top convertible bonds to hold

# Convertible bonds for portfolio
CONVERTIBLE_BONDS = {
    'CB001': {'name': '格力转2', 'base_price': 130.0, 'volatility': 0.12},
    'CB002': {'name': '招商转债', 'base_price': 125.0, 'volatility': 0.10},
    'CB003': {'name': '浦发转2', 'base_price': 128.0, 'volatility': 0.11},
    'CB004': {'name': '宝钢转债', 'base_price': 132.0, 'volatility': 0.09},
    'CB005': {'name': '福能转债', 'base_price': 127.0, 'volatility': 0.13},
    'CB006': {'name': '天吉转债', 'base_price': 131.0, 'volatility': 0.10},
    'CB007': {'name': '东财转债', 'base_price': 129.0, 'volatility': 0.12},
    'CB008': {'name': '长证转债', 'base_price': 126.0, 'volatility': 0.11},
    'CB009': {'name': '维博转债', 'base_price': 130.0, 'volatility': 0.10},
    'CB010': {'name': '云赛转债', 'base_price': 128.5, 'volatility': 0.11},
}

# ==================== Data Generation ====================

def generate_synthetic_bond_data(bond_code: str, start_date: datetime, 
                                  end_date: datetime, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic daily OHLCV data for a convertible bond.
    
    Uses geometric Brownian motion to create realistic price movements.
    """
    np.random.seed(seed + hash(bond_code) % 1000)
    
    bond_info = CONVERTIBLE_BONDS[bond_code]
    base_price = bond_info['base_price']
    volatility = bond_info['volatility']
    drift = 0.05 / 252  # 5% annual drift
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    prices = [base_price]
    
    for _ in range(1, len(dates)):
        # Geometric Brownian Motion
        random_return = np.random.normal(drift, volatility / np.sqrt(252))
        new_price = prices[-1] * (1 + random_return)
        prices.append(max(new_price, 100))  # Floor at 100 (par value-ish)
    
    # Create OHLCV data
    df = pd.DataFrame({
        'date': dates,
        'close_price': prices,
    })
    
    # Generate OHLV from close prices
    df['open'] = df['close_price'].shift(1).fillna(df['close_price'].iloc[0])
    df['high'] = df[['open', 'close_price']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, len(df))))
    df['low'] = df[['open', 'close_price']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, len(df))))
    df['volume'] = np.random.uniform(1e6, 5e6, len(df))
    
    return df[['date', 'open', 'high', 'low', 'close_price', 'volume']]


def calculate_bond_metrics(df: pd.DataFrame) -> dict | None:
    """
    Calculate key metrics for a convertible bond from its price history.
    
    Metrics:
    - Current Price: latest close price
    - Price Change %: price appreciation/depreciation
    - Volatility: 30-day rolling volatility
    - Trend: 20-day vs 60-day moving average relationship
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
    
    # Volume trend
    metrics['avg_vol_20d'] = df['volume'].rolling(20).mean().iloc[-1]
    
    return metrics


def rank_convertible_bonds(bonds_data: dict) -> pd.DataFrame:
    """
    Rank convertible bonds based on multiple criteria.
    
    Ranking logic:
    - Better performance (positive price change) + low volatility = high rank
    - Positive trend (MA20 > MA60) = boost
    """
    rankings = []
    
    for bond_code, metrics in bonds_data.items():
        if metrics is None:
            continue
        
        score = 0
        
        # Score 1: Price momentum (20% weight)
        price_score = max(0, min(100, 50 + metrics['price_change'] * 100))
        score += price_score * 0.2
        
        # Score 2: Volatility (lower is better, 20% weight)
        vol_score = max(0, min(100, 100 - metrics['volatility_30d'] * 100))
        score += vol_score * 0.2
        
        # Score 3: Trend (MA20 > MA60 is positive, 30% weight)
        trend_score = max(0, min(100, 50 + metrics['trend'] * 200))
        score += trend_score * 0.3
        
        # Score 4: Value (current price relative to ma_60, 30% weight)
        value_score = max(0, min(100, 50 - (metrics['current_price'] - metrics['ma_60']) / metrics['ma_60'] * 100))
        score += value_score * 0.3
        
        rankings.append({
            'bond_code': bond_code,
            'bond_name': CONVERTIBLE_BONDS[bond_code]['name'],
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
    """Construct equal-weight portfolio from top-ranked bonds."""
    selected_bonds = top_bonds[:num_bonds]
    weight = 1.0 / len(selected_bonds) if selected_bonds else 0
    return {bond: weight for bond in selected_bonds}


def calculate_portfolio_return(portfolio_weights: dict, bonds_prices: dict, 
                              current_date: datetime, next_date: datetime) -> float:
    """
    Calculate portfolio return for a given period.
    """
    if not portfolio_weights:
        return 0.0
    
    total_return = 0.0
    
    for bond_code, weight in portfolio_weights.items():
        if bond_code not in bonds_prices:
            continue
        
        df = bonds_prices[bond_code]
        
        # Find price at current date and next date
        price_current = df[df['date'] <= current_date]['close_price'].iloc[-1] if not df[df['date'] <= current_date].empty else None
        price_next = df[df['date'] <= next_date]['close_price'].iloc[-1] if not df[df['date'] <= next_date].empty else price_current
        
        if price_current is None or price_next is None or price_current == 0:
            continue
        
        bond_return = (price_next - price_current) / price_current
        total_return += bond_return * weight
    
    return total_return


# ==================== Main Backtest ====================

def main():
    print("\n" + "="*70)
    print("CONVERTIBLE BOND ROTATION STRATEGY BACKTEST (Demo)")
    print("="*70 + "\n")
    
    start_date = datetime.strptime(START_DATE_STR, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE_STR, "%Y-%m-%d")
    
    # Step 1: Generate synthetic data for all bonds
    print("[Data] Generating synthetic daily data for convertible bonds...")
    bonds_prices = {}
    for bond_code in CONVERTIBLE_BONDS.keys():
        df = generate_synthetic_bond_data(bond_code, start_date, end_date)
        bonds_prices[bond_code] = df
        print(f"  ✓ {bond_code} ({CONVERTIBLE_BONDS[bond_code]['name']}): {len(df)} days")
    
    print(f"✓ Successfully generated data for {len(bonds_prices)} bonds\n")
    
    # Step 2: Backtest with monthly rebalancing
    print("[Backtest] Running backtest with monthly rebalancing...")
    
    portfolio_value = [INITIAL_CAPITAL]
    rebalance_dates = [start_date]
    holdings_history = []
    
    current_date = start_date
    portfolio_weights = {}
    
    while current_date < end_date:
        # Calculate metrics for all bonds as of current date
        bonds_data = {}
        for bond_code, df in bonds_prices.items():
            df_up_to_date = df[df['date'] <= current_date]
            if len(df_up_to_date) >= 60:
                metrics = calculate_bond_metrics(df_up_to_date)
                bonds_data[bond_code] = metrics
        
        if bonds_data:
            # Rank bonds and select top N
            ranking_df = rank_convertible_bonds(bonds_data)
            if not ranking_df.empty:
                top_bonds = ranking_df['bond_code'].head(NUM_TOP_BONDS).tolist()
                portfolio_weights = construct_equal_weight_portfolio(top_bonds, NUM_TOP_BONDS)
                
                # Record holdings
                holdings = ", ".join([CONVERTIBLE_BONDS[b]['name'] for b in top_bonds[:3]])
                holdings_history.append({
                    'date': current_date,
                    'holdings': holdings,
                    'top_scores': ranking_df['score'].head(3).tolist()
                })
        
        # Calculate return to next month
        next_date = min(current_date + timedelta(days=30), end_date)
        monthly_return = calculate_portfolio_return(portfolio_weights, bonds_prices, current_date, next_date)
        
        new_value = portfolio_value[-1] * (1 + monthly_return)
        portfolio_value.append(new_value)
        rebalance_dates.append(next_date)
        
        if len(rebalance_dates) % 3 == 0:  # Print every 3 months
            print(f"  {current_date.strftime('%Y-%m-%d')}: Return={monthly_return*100:+.2f}% → Portfolio=${new_value:,.0f}")
        
        current_date = next_date
    
    # Step 3: Calculate performance metrics
    print("\n[Metrics] Performance Summary")
    print("-" * 70)
    
    portfolio_array = np.array(portfolio_value)
    total_return = (portfolio_value[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    years = (end_date - start_date).days / 365.25
    annual_return = total_return / years if years > 0 else 0
    
    returns = np.diff(portfolio_array) / portfolio_array[:-1]
    sharpe_ratio = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(12)  # Monthly Sharpe
    
    # Max drawdown
    cummax = np.maximum.accumulate(portfolio_array)
    drawdown = (portfolio_array - cummax) / cummax
    max_drawdown = np.min(drawdown)
    
    # Win rate (% of months with positive return)
    win_rate = (returns > 0).sum() / len(returns) * 100
    
    print(f"Backtest Period:      {START_DATE_STR} to {END_DATE_STR}")
    print(f"Initial Capital:      ${INITIAL_CAPITAL:,.0f}")
    print(f"Final Portfolio Value: ${portfolio_value[-1]:,.0f}")
    print(f"Total Return:         {total_return*100:+.2f}%")
    print(f"Annualized Return:    {annual_return*100:+.2f}%")
    print(f"Sharpe Ratio (Monthly): {sharpe_ratio:.2f}")
    print(f"Max Drawdown:         {max_drawdown*100:.2f}%")
    print(f"Win Rate:             {win_rate:.1f}%")
    print(f"Rebalance Events:     {len(rebalance_dates)}")
    
    # Step 4: Save results
    results_df = pd.DataFrame({
        'date': rebalance_dates[1:],
        'portfolio_value': portfolio_value[1:],
        'return_pct': np.concatenate([[0], (np.diff(portfolio_array) / portfolio_array[:-1]) * 100])
    })
    
    output_path = "analysis_output_convertible_rotation.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Save detailed holdings history
    if holdings_history:
        holdings_df = pd.DataFrame(holdings_history)
        holdings_df.to_csv("analysis_convertible_holdings.csv", index=False)
        print(f"✓ Holdings history saved to analysis_convertible_holdings.csv")
    
    # Print sample holdings
    print("\n[Holdings] Sample Rotation Schedule (First 6 months):")
    print("-" * 70)
    for i, record in enumerate(holdings_history[:6]):
        print(f"{record['date'].strftime('%Y-%m-%d')}: {record['holdings']}")
    
    print("\n" + "="*70)
    print("Backtest completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
