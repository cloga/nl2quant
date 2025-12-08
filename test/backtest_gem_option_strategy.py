"""
创业板ETF期权卖方策略完整回测
策略：卖出虚值认沽期权 + 动态移仓 + 双卖（认购+认沽）备兑策略
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config

class OptionStrategyBacktest:
    """期权卖方策略回测引擎"""
    
    def __init__(self, price_data, initial_capital=100000, contracts_per_trade=10):
        """
        初始化回测引擎
        
        Args:
            price_data: DataFrame，包含日度价格数据（Open, High, Low, Close）
            initial_capital: 初始资金
            contracts_per_trade: 每次卖出合约张数（默认10张 = 1000股，改为100张 = 10000股）
        """
        self.price_data = price_data
        self.initial_capital = initial_capital
        
        # 策略参数
        self.contract_size = 100  # 每张合约对应股数
        self.contracts_per_trade = contracts_per_trade  # 每次卖出合约张数
        self.otm_put_ratio = 0.10  # 认沽虚值度10%
        self.otm_call_ratio = 0.10  # 认购虚值度10%
        self.roll_days_threshold = 15  # 移仓时间阈值（天）
        self.roll_otm_ratio = 0.05  # 移仓后虚值度5%
        
        # 权利金估算参数（简化模型，实际应使用BS模型）
        # 注意：这里是每股权利金占标的价格的比例，不是名义价值的比例
        self.premium_rate_10otm = 0.001  # 10%虚值期权月度权利金约0.1%（约0.002元/股，假设标的2元）
        self.premium_rate_5otm = 0.002   # 5%虚值期权月度权利金约0.2%
        
        # 状态变量
        self.cash = initial_capital
        self.stock_position = 0  # 持有ETF股数
        self.stock_cost_basis = 0  # 持仓成本价
        self.total_premium_received = 0  # 累计收取权利金
        
        # 期权仓位
        self.put_position = 0  # 卖出认沽张数（负数表示short）
        self.put_strike = 0    # 认沽行权价
        self.put_open_date = None
        self.put_expiry_date = None
        
        self.call_position = 0  # 卖出认购张数
        self.call_strike = 0
        self.call_open_date = None
        self.call_expiry_date = None
        
        # 交易记录
        self.trades = []
        self.daily_equity = []
        self.max_drawdown = 0
        self.max_capital_usage = 0
        
    def estimate_premium(self, spot_price, strike_price, days_to_expiry, option_type='put'):
        """
        简化的权利金估算（实际应使用Black-Scholes模型）
        基于虚值度和剩余时间估算
        
        返回：每股期权权利金（元）
        """
        if days_to_expiry <= 0 or spot_price <= 0 or strike_price <= 0:
            return 0
        
        try:
            if option_type == 'put':
                moneyness = (strike_price - spot_price) / spot_price  # 负值表示虚值
            else:  # call
                moneyness = (strike_price - spot_price) / spot_price  # 正值表示虚值
            
            # 虚值度越大，权利金越低
            # 这里的rate是权利金占标的价格的比例
            if abs(moneyness) >= 0.10:
                base_rate = 0.001  # 10%虚值：每股权利金约为标的价格的0.1%
            elif abs(moneyness) >= 0.05:
                base_rate = 0.002  # 5%虚值：每股权利金约为标的价格的0.2%
            else:
                base_rate = 0.003  # 平值：每股权利金约为标的价格的0.3%
            
            # 时间价值衰减（线性简化）
            time_factor = days_to_expiry / 30
            premium_rate = base_rate * time_factor
            
            result = spot_price * premium_rate
            # 保证结果不是NaN
            if not isinstance(result, (int, float)) or result != result:  # NaN check
                return 0
            return result
        except:
            return 0
    
    def get_next_month_expiry(self, current_date):
        """获取下月到期日（简化为每月最后一个交易日）"""
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1, day=28)
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=28)
        return next_month
    
    def days_to_expiry(self, current_date, expiry_date):
        """计算到期天数"""
        if expiry_date is None:
            return 0
        return max(0, (expiry_date - current_date).days)
    
    def sell_put_option(self, current_date, spot_price, otm_ratio=None):
        """卖出认沽期权"""
        if otm_ratio is None:
            otm_ratio = self.otm_put_ratio
        
        strike = spot_price * (1 - otm_ratio)
        expiry = self.get_next_month_expiry(current_date)
        days = self.days_to_expiry(current_date, expiry)
        
        premium_per_share = self.estimate_premium(spot_price, strike, days, 'put')
        total_premium = premium_per_share * self.contract_size * self.contracts_per_trade
        
        # 保证premium是有效的数字
        if not isinstance(total_premium, (int, float)) or total_premium != total_premium:
            total_premium = 0
        
        self.put_position = -self.contracts_per_trade
        self.put_strike = strike
        self.put_open_date = current_date
        self.put_expiry_date = expiry
        
        self.cash += total_premium
        self.total_premium_received += total_premium
        
        self.trades.append({
            'date': current_date,
            'action': 'SELL_PUT',
            'contracts': self.contracts_per_trade,
            'strike': strike,
            'spot': spot_price,
            'premium': total_premium,
            'otm_ratio': otm_ratio
        })
        
        return total_premium
    
    def sell_call_option(self, current_date, spot_price, strike_price, contracts=None):
        """卖出认购期权（备兑开仓）
        
        Args:
            contracts: 卖出合约张数，默认为 contracts_per_trade，可指定为持仓全部
        """
        if contracts is None:
            contracts = self.contracts_per_trade
            
        expiry = self.get_next_month_expiry(current_date)
        days = self.days_to_expiry(current_date, expiry)
        
        premium_per_share = self.estimate_premium(spot_price, strike_price, days, 'call')
        total_premium = premium_per_share * self.contract_size * contracts
        
        # 保证premium是有效的数字
        if not isinstance(total_premium, (int, float)) or total_premium != total_premium:
            total_premium = 0
        
        self.call_position = -contracts
        self.call_strike = strike_price
        self.call_open_date = current_date
        self.call_expiry_date = expiry
        
        self.cash += total_premium
        self.total_premium_received += total_premium
        
        self.trades.append({
            'date': current_date,
            'action': 'SELL_CALL',
            'contracts': contracts,
            'strike': strike_price,
            'spot': spot_price,
            'premium': total_premium
        })
        
        return total_premium
    
    def check_put_exercise(self, current_date, spot_price):
        """检查认沽期权是否被行权"""
        if self.put_position >= 0:
            return False
        
        if current_date < self.put_expiry_date:
            return False
        
        # 到期日检查
        if spot_price < self.put_strike:
            # 被行权，接入股票
            shares = abs(self.put_position) * self.contract_size
            cost = self.put_strike * shares
            
            # 更新持仓
            old_position = self.stock_position
            old_cost_basis = self.stock_cost_basis
            
            self.stock_position += shares
            self.stock_cost_basis = (old_position * old_cost_basis + shares * self.put_strike) / self.stock_position
            
            self.cash -= cost
            
            self.trades.append({
                'date': current_date,
                'action': 'PUT_EXERCISED',
                'shares': shares,
                'price': self.put_strike,
                'cost': cost,
                'new_position': self.stock_position,
                'avg_cost': self.stock_cost_basis
            })
            
            # 清空认沽仓位
            self.put_position = 0
            self.put_strike = 0
            self.put_open_date = None
            self.put_expiry_date = None
            
            return True
        else:
            # 未被行权，期权作废
            self.trades.append({
                'date': current_date,
                'action': 'PUT_EXPIRED',
                'strike': self.put_strike,
                'spot': spot_price
            })
            
            self.put_position = 0
            self.put_strike = 0
            self.put_open_date = None
            self.put_expiry_date = None
            
            return False
    
    def check_call_exercise(self, current_date, spot_price):
        """检查认购期权是否被行权"""
        if self.call_position >= 0:
            return False
        
        if current_date < self.call_expiry_date:
            return False
        
        # 到期日检查
        if spot_price > self.call_strike:
            # 被行权，卖出股票
            shares = abs(self.call_position) * self.contract_size
            revenue = self.call_strike * shares
            
            profit = (self.call_strike - self.stock_cost_basis) * shares
            
            self.stock_position -= shares
            self.cash += revenue
            
            self.trades.append({
                'date': current_date,
                'action': 'CALL_EXERCISED',
                'shares': shares,
                'price': self.call_strike,
                'cost_basis': self.stock_cost_basis,
                'profit': profit,
                'remaining_position': self.stock_position
            })
            
            # 如果全部卖出，重置成本价
            if self.stock_position == 0:
                self.stock_cost_basis = 0
            
            # 清空认购仓位
            self.call_position = 0
            self.call_strike = 0
            self.call_open_date = None
            self.call_expiry_date = None
            
            return True
        else:
            # 未被行权
            self.trades.append({
                'date': current_date,
                'action': 'CALL_EXPIRED',
                'strike': self.call_strike,
                'spot': spot_price
            })
            
            self.call_position = 0
            self.call_strike = 0
            self.call_open_date = None
            self.call_expiry_date = None
            
            return False
    
    def check_roll_opportunity(self, current_date, spot_price, open_price):
        """检查是否满足移仓条件"""
        if self.put_position >= 0:
            return False
        
        days = self.days_to_expiry(current_date, self.put_expiry_date)
        
        # 条件1：距离到期≤15天
        # 条件2：标的上涨（当前价 > 开仓时价格）
        if days <= self.roll_days_threshold and spot_price > open_price:
            return True
        
        return False
    
    def roll_put_option(self, current_date, spot_price):
        """执行移仓：平掉当前认沽，开新的5%虚值认沽"""
        # 平掉当前认沽（买入平仓，支付权利金）
        days = self.days_to_expiry(current_date, self.put_expiry_date)
        close_premium_per_share = self.estimate_premium(spot_price, self.put_strike, days, 'put')
        close_cost = close_premium_per_share * self.contract_size * abs(self.put_position)
        
        self.cash -= close_cost
        
        self.trades.append({
            'date': current_date,
            'action': 'ROLL_CLOSE_PUT',
            'contracts': abs(self.put_position),
            'strike': self.put_strike,
            'cost': close_cost
        })
        
        # 清空旧仓位
        self.put_position = 0
        self.put_strike = 0
        
        # 开新的5%虚值认沽
        premium = self.sell_put_option(current_date, spot_price, otm_ratio=self.roll_otm_ratio)
        
        self.trades.append({
            'date': current_date,
            'action': 'ROLL_OPEN_PUT',
            'contracts': self.contracts_per_trade,
            'strike': self.put_strike,
            'premium': premium,
            'net_premium': premium - close_cost
        })
    
    def run(self):
        """运行回测 - 改进版：卖认沽全部卖出，认沽被行权后卖认购也全部卖出"""
        print("开始回测...")
        print(f"初始资金: {self.initial_capital:,.2f}")
        print(f"回测期间: {self.price_data.index[0]} 至 {self.price_data.index[-1]}")
        print("="*60)
        
        # 按月分组
        monthly_data = self.price_data.resample('M').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        
        # 记录初始状态
        first_date = monthly_data.index[0]
        first_price = monthly_data.loc[first_date, 'Close']
        
        # 第一个月：卖出10张10%虚值认沽
        premium = self.sell_put_option(first_date, first_price)
        print(f"{first_date.strftime('%Y-%m')}: 初始卖出10张认沽，权利金 {premium:,.2f}")
        
        open_price_at_sell = first_price  # 记录开仓时价格用于判断移仓
        
        # 逐月迭代
        for i in range(1, len(monthly_data)):
            current_date = monthly_data.index[i]
            current_price = monthly_data.loc[current_date, 'Close']
            current_open = monthly_data.loc[current_date, 'Open']
            
            # 1. 检查认沽期权到期/行权
            if self.put_position < 0 and current_date >= self.put_expiry_date:
                exercised = self.check_put_exercise(current_date, current_price)
                
                if exercised:
                    shares_bought = self.stock_position
                    avg_cost = self.stock_cost_basis
                    call_strike = avg_cost * (1 + self.otm_call_ratio)
                    
                    print(f"{current_date.strftime('%Y-%m')}: 认沽被行权，接入 {shares_bought} 股，成本价 {avg_cost:.2f}")
                    
                    # 立即卖出全部认购
                    call_contracts = shares_bought // self.contract_size
                    if call_contracts > 0:
                        call_premium = self.sell_call_option(current_date, current_price, call_strike, contracts=call_contracts)
                        print(f"  → 立即卖出 {call_contracts} 张认购，行权价 {call_strike:.2f}，权利金 {call_premium:,.2f}")
                    
                    open_price_at_sell = current_price
                else:
                    print(f"{current_date.strftime('%Y-%m')}: 认沽到期作废，继续卖出下月认沽")
                    premium = self.sell_put_option(current_date, current_price)
                    print(f"  → 权利金 {premium:,.2f}")
                    open_price_at_sell = current_price
            
            # 2. 检查认购期权到期/行权
            if self.call_position < 0 and current_date >= self.call_expiry_date:
                exercised = self.check_call_exercise(current_date, current_price)
                
                if exercised:
                    print(f"{current_date.strftime('%Y-%m')}: 认购被行权，全部出清")
                    print(f"  → 重新卖出认沽")
                    premium = self.sell_put_option(current_date, current_price)
                    print(f"  → 权利金 {premium:,.2f}")
                    open_price_at_sell = current_price
                else:
                    # 认购未被行权，但仍持有股票，继续卖认购
                    if self.stock_position > 0:
                        call_strike = self.stock_cost_basis * (1 + self.otm_call_ratio)
                        call_premium = self.sell_call_option(current_date, current_price, call_strike)
                        print(f"{current_date.strftime('%Y-%m')}: 认购到期作废，继续卖认购，权利金 {call_premium:,.2f}")
            
            # 2b. 如果有持仓但没有认购头寸，卖认购
            elif self.call_position >= 0 and self.stock_position > 0:
                # 检查是否已经有未到期的认购
                if self.call_position == 0:
                    call_strike = self.stock_cost_basis * (1 + self.otm_call_ratio)
                    call_premium = self.sell_call_option(current_date, current_price, call_strike)
                    print(f"{current_date.strftime('%Y-%m')}: 继续卖认购，权利金 {call_premium:,.2f}")
            
            # 3. 检查移仓机会（当无任何持仓时）
            if self.put_position < 0 and self.call_position >= 0 and self.stock_position == 0:
                if self.check_roll_opportunity(current_date, current_price, open_price_at_sell):
                    print(f"{current_date.strftime('%Y-%m')}: 触发移仓条件，执行移仓")
                    self.roll_put_option(current_date, current_price)
                    open_price_at_sell = current_price
            
            # 4. 计算每日权益
            stock_value = self.stock_position * current_price
            total_equity = self.cash + stock_value
            
            # 计算资金占用（用于评估需要准备的资金）
            if self.stock_position > 0:
                capital_usage = self.stock_position * self.stock_cost_basis
                self.max_capital_usage = max(self.max_capital_usage, capital_usage)
            
            self.daily_equity.append({
                'date': current_date,
                'cash': self.cash,
                'stock_value': stock_value,
                'total_equity': total_equity,
                'stock_position': self.stock_position,
                'premium_received': self.total_premium_received
            })
        
        # 生成回测报告
        self.generate_report()
    
    def generate_report(self):
        """生成回测报告"""
        print("\n" + "="*60)
        print("回测报告")
        print("="*60)
        
        equity_df = pd.DataFrame(self.daily_equity)
        
        final_equity = equity_df.iloc[-1]['total_equity']
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率
        years = (equity_df.iloc[-1]['date'] - equity_df.iloc[0]['date']).days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # 最大回撤
        equity_series = equity_df['total_equity']
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_dd = drawdown.min()
        
        # 夏普比率（简化，假设无风险利率3%）
        monthly_returns = equity_series.pct_change().dropna()
        sharpe = (monthly_returns.mean() - 0.03/12) / monthly_returns.std() * np.sqrt(12)
        
        print(f"\n总收益率: {total_return*100:.2f}%")
        print(f"年化收益率: {annual_return*100:.2f}%")
        print(f"最大回撤: {max_dd*100:.2f}%")
        print(f"夏普比率: {sharpe:.2f}")
        print(f"\n累计权利金收入: {self.total_premium_received:,.2f}")
        print(f"最终权益: {final_equity:,.2f}")
        print(f"最大资金占用: {self.max_capital_usage:,.2f}")
        print(f"建议准备资金倍数: {self.max_capital_usage/self.initial_capital:.1f}x")
        
        # 交易统计
        trades_df = pd.DataFrame(self.trades)
        
        put_exercised = len(trades_df[trades_df['action'] == 'PUT_EXERCISED'])
        put_expired = len(trades_df[trades_df['action'] == 'PUT_EXPIRED'])
        call_exercised = len(trades_df[trades_df['action'] == 'CALL_EXERCISED'])
        roll_count = len(trades_df[trades_df['action'] == 'ROLL_OPEN_PUT'])
        
        print(f"\n交易统计:")
        print(f"  认沽被行权: {put_exercised} 次")
        print(f"  认沽到期作废: {put_expired} 次")
        print(f"  认购被行权: {call_exercised} 次")
        print(f"  移仓次数: {roll_count} 次")
        print(f"  胜率（收权利金）: {put_expired/(put_exercised+put_expired)*100:.1f}%")
        
        # 保存详细数据
        equity_df.to_csv('d:/project/nl2quant/test/option_strategy_equity.csv', index=False, encoding='utf-8-sig')
        trades_df.to_csv('d:/project/nl2quant/test/option_strategy_trades.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n详细数据已保存:")
        print(f"  权益曲线: d:/project/nl2quant/test/option_strategy_equity.csv")
        print(f"  交易记录: d:/project/nl2quant/test/option_strategy_trades.csv")


def fetch_gem_data():
    """获取创业板指数全部历史数据"""
    provider = Config.DATA_PROVIDER
    
    if provider == "akshare":
        import akshare as ak
        print("使用 AkShare 获取创业板指数数据...")
        df = ak.index_zh_a_hist(symbol="399006", period="daily", start_date="20100101", end_date="20251231", adjust="")
        if df.empty:
            raise ValueError("AkShare 未获取到数据")
        df = df.rename(columns={"日期": "date", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[["open", "high", "low", "close"]]
        df.columns = ["Open", "High", "Low", "Close"]
    else:
        import tushare as ts
        print("使用 Tushare 获取创业板指数数据...")
        if not Config.TUSHARE_TOKEN:
            raise ValueError("Tushare token 未配置")
        ts.set_token(Config.TUSHARE_TOKEN)
        pro = ts.pro_api()
        
        df_list = []
        start_year = 2010
        end_year = datetime.now().year
        
        for year in range(start_year, end_year + 1):
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            df_chunk = pro.index_daily(ts_code='399006.SZ', start_date=start_date, end_date=end_date)
            if not df_chunk.empty:
                df_list.append(df_chunk)
        
        if not df_list:
            raise ValueError("Tushare 未获取到数据")
        
        df = pd.concat(df_list, ignore_index=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        df = df.set_index('trade_date')
        df = df[['open', 'high', 'low', 'close']]
        df.columns = ['Open', 'High', 'Low', 'Close']
    
    return df


def main():
    # 获取创业板指数历史数据
    print("正在获取创业板指数历史数据...\n")
    price_data = fetch_gem_data()
    
    print(f"数据范围: {price_data.index[0].strftime('%Y-%m-%d')} 至 {price_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"总交易日: {len(price_data)} 天\n")
    
    # 初始化回测引擎（10万初始资金）
    backtest = OptionStrategyBacktest(price_data, initial_capital=100000)
    
    # 运行回测
    backtest.run()


if __name__ == "__main__":
    main()
