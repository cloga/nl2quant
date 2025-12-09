# 通用网格策略架构设计

## 1. 总体设计思路

### 1.1 核心抽象
将网格策略拆分为 **5 个可独立配置的模块**：

```
┌─────────────────────────────────────────────────────┐
│           Grid Strategy Framework                    │
├─────────────────────────────────────────────────────┤
│  1. DataProvider    ─┐                              │
│  2. GridEngine      ─┼──> Strategy Config           │
│  3. PositionManager ─┤                              │
│  4. RiskManager     ─┤                              │
│  5. Backtester      ─┘                              │
└─────────────────────────────────────────────────────┘
```

### 1.2 设计原则
- **标的无关性**：支持股票、ETF、指数、期货、加密货币
- **策略可配置**：网格间距、资金分配、风控规则均可参数化
- **数据源解耦**：统一接口，支持 Tushare/AKShare/CSV/API
- **扩展性**：易于添加新策略变体（动态网格、趋势过滤等）
- **可测试性**：核心逻辑与回测引擎分离

---

## 2. 模块设计详解

### 2.1 DataProvider（数据提供者）

**职责**：统一不同数据源的接口

```python
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class DataProvider(ABC):
    """抽象数据提供者基类"""
    
    @abstractmethod
    def fetch_price(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        """获取价格序列（收盘价）"""
        pass
    
    @abstractmethod
    def fetch_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取 OHLCV 数据（可选，用于高级策略）"""
        pass
    
    @abstractmethod
    def get_ticker_info(self, ticker: str) -> dict:
        """获取标的元信息（名称、交易所、最小交易单位等）"""
        pass


class TushareProvider(DataProvider):
    """Tushare 数据源"""
    def __init__(self, token: str):
        self.token = token
        # ... 初始化逻辑
    
    def fetch_price(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        # 实现 Tushare API 调用
        pass


class AKShareProvider(DataProvider):
    """AKShare 数据源"""
    def fetch_price(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        # 实现 AKShare API 调用
        pass


class CSVProvider(DataProvider):
    """CSV 文件数据源（用于自定义数据）"""
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
    
    def fetch_price(self, ticker: str, start_date: str, end_date: str) -> pd.Series:
        # 从 CSV 读取并过滤日期
        pass
```

**使用示例**：
```python
# 自动选择数据源
provider = DataProviderFactory.create(
    source='tushare',  # 或 'akshare', 'csv', 'api'
    config={'token': 'your_token'}
)
price_data = provider.fetch_price('h00922.CSI', '2018-01-01', '2025-12-31')
```

---

### 2.2 GridEngine（网格引擎）

**职责**：网格逻辑的核心，管理买卖订单

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum

class GridMode(Enum):
    """网格模式"""
    FIXED = "fixed"              # 固定间距
    DYNAMIC_ATR = "dynamic_atr"  # 基于 ATR 动态间距
    DYNAMIC_VOL = "dynamic_vol"  # 基于波动率动态间距
    GEOMETRIC = "geometric"      # 几何间距（适合加密货币）


@dataclass
class GridConfig:
    """网格配置"""
    # 基础参数
    mode: GridMode = GridMode.FIXED
    spacing: float = 0.02                    # 间距（2% 或 ATR 的倍数）
    num_grids: int = 10                      # 网格数量
    grid_unit: float = 50_000.0              # 每格金额
    
    # 动态参数
    spacing_multiplier: float = 1.0          # 间距调整倍数
    spacing_func: Optional[Callable] = None  # 自定义间距函数
    
    # 风控参数
    max_grids_per_day: int = 1               # 每日最大网格买入数
    price_floor: float = 0.5                 # 相对入场价的最低价格（50%）
    price_ceiling: Optional[float] = None    # 相对入场价的最高价格
    
    # 高级功能
    enable_trailing: bool = False            # 启用移动止盈
    enable_pyramid: bool = False             # 启用金字塔加仓


@dataclass
class GridOrder:
    """网格订单"""
    price: float                             # 订单价格
    shares: float                            # 股数
    order_type: str                          # 'buy' or 'sell'
    grid_level: int                          # 网格层级
    timestamp: Optional[pd.Timestamp] = None
    status: str = 'pending'                  # 'pending', 'filled', 'cancelled'
    fill_price: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'price': self.price,
            'shares': self.shares,
            'type': self.order_type,
            'level': self.grid_level,
            'status': self.status
        }


class GridEngine:
    """网格交易引擎"""
    
    def __init__(self, config: GridConfig, initial_price: float):
        self.config = config
        self.entry_price = initial_price
        self.pending_buys: List[GridOrder] = []
        self.pending_sells: List[GridOrder] = []
        self.filled_orders: List[GridOrder] = []
        
        # 初始化网格层级
        self._initialize_grid_levels()
    
    def _initialize_grid_levels(self):
        """初始化网格买入价格层级"""
        if self.config.mode == GridMode.FIXED:
            for i in range(1, self.config.num_grids + 1):
                buy_price = self.entry_price * (1 - self.config.spacing * i)
                if buy_price >= self.entry_price * self.config.price_floor:
                    self.pending_buys.append(
                        GridOrder(buy_price, 0, 'buy', -i)
                    )
        
        elif self.config.mode == GridMode.GEOMETRIC:
            # 几何级数间距（适合波动大的市场）
            for i in range(1, self.config.num_grids + 1):
                buy_price = self.entry_price * ((1 - self.config.spacing) ** i)
                if buy_price >= self.entry_price * self.config.price_floor:
                    self.pending_buys.append(
                        GridOrder(buy_price, 0, 'buy', -i)
                    )
    
    def update(self, current_price: float, current_date: pd.Timestamp,
               available_cash: float) -> tuple[List[GridOrder], List[GridOrder]]:
        """
        更新网格状态，返回待执行的买卖订单
        
        Returns:
            (filled_buys, filled_sells): 本轮成交的买卖订单
        """
        filled_buys = []
        filled_sells = []
        
        # 检查卖出订单
        for sell_order in self.pending_sells[:]:
            if current_price >= sell_order.price:
                sell_order.status = 'filled'
                sell_order.fill_price = current_price
                sell_order.timestamp = current_date
                filled_sells.append(sell_order)
                self.pending_sells.remove(sell_order)
                self.filled_orders.append(sell_order)
        
        # 检查买入订单
        buys_today = 0
        for buy_order in self.pending_buys[:]:
            if current_price <= buy_order.price and buys_today < self.config.max_grids_per_day:
                # 检查是否有足够资金
                required_cash = self.config.grid_unit
                if available_cash >= required_cash:
                    # 计算实际买入股数
                    shares = (self.config.grid_unit * 0.997) / current_price  # 扣除手续费
                    buy_order.shares = shares
                    buy_order.status = 'filled'
                    buy_order.fill_price = current_price
                    buy_order.timestamp = current_date
                    filled_buys.append(buy_order)
                    self.pending_buys.remove(buy_order)
                    self.filled_orders.append(buy_order)
                    
                    # 创建对应卖出订单
                    sell_price = current_price * (1 + self.config.spacing)
                    sell_order = GridOrder(
                        sell_price, shares, 'sell',
                        buy_order.grid_level + 1,
                        current_date
                    )
                    self.pending_sells.append(sell_order)
                    
                    buys_today += 1
        
        return filled_buys, filled_sells
    
    def adjust_spacing(self, new_spacing: float):
        """动态调整网格间距"""
        self.config.spacing = new_spacing
        # 重新计算未成交订单的价格
        # ...
    
    def get_status(self) -> dict:
        """获取网格状态"""
        return {
            'pending_buys': len(self.pending_buys),
            'pending_sells': len(self.pending_sells),
            'filled_orders': len(self.filled_orders),
            'avg_cost': self._calculate_avg_cost(),
        }
    
    def _calculate_avg_cost(self) -> float:
        """计算持仓均价"""
        buy_orders = [o for o in self.filled_orders if o.order_type == 'buy' and o.status == 'filled']
        if not buy_orders:
            return 0.0
        total_cost = sum(o.fill_price * o.shares for o in buy_orders)
        total_shares = sum(o.shares for o in buy_orders)
        return total_cost / total_shares if total_shares > 0 else 0.0
```

---

### 2.3 PositionManager（持仓管理器）

**职责**：管理基础仓位 + 网格仓位，计算总净值

```python
@dataclass
class Position:
    """持仓"""
    shares: float = 0.0
    avg_cost: float = 0.0
    cash: float = 0.0
    
    def add(self, shares: float, price: float, cost: float):
        """加仓"""
        total_cost = self.shares * self.avg_cost + shares * price + cost
        self.shares += shares
        self.avg_cost = total_cost / self.shares if self.shares > 0 else 0.0
        self.cash -= (shares * price + cost)
    
    def reduce(self, shares: float, price: float, cost: float):
        """减仓"""
        self.shares -= shares
        self.cash += (shares * price - cost)
        if self.shares <= 0:
            self.shares = 0.0
            self.avg_cost = 0.0
    
    def market_value(self, current_price: float) -> float:
        """市值"""
        return self.shares * current_price + self.cash
    
    def unrealized_pnl(self, current_price: float) -> float:
        """浮动盈亏"""
        return (current_price - self.avg_cost) * self.shares


class PositionManager:
    """持仓管理器"""
    
    def __init__(self, initial_cash: float, base_ratio: float = 0.5):
        """
        Args:
            initial_cash: 初始资金
            base_ratio: 基础仓位比例（0-1）
        """
        self.base_position = Position(cash=initial_cash * base_ratio)
        self.grid_position = Position(cash=initial_cash * (1 - base_ratio))
        self.initial_cash = initial_cash
        
        # 交易历史
        self.trade_history: List[dict] = []
    
    def initialize_base_position(self, price: float):
        """初始化基础仓位（买入并持有）"""
        shares = (self.base_position.cash * 0.997) / price  # 扣除手续费
        cost = self.base_position.cash * 0.003
        self.base_position.add(shares, price, cost)
        
        self.trade_history.append({
            'date': pd.Timestamp.now(),
            'type': 'base_buy',
            'price': price,
            'shares': shares,
            'cost': cost
        })
    
    def execute_grid_buy(self, order: GridOrder, commission_rate: float = 0.0003):
        """执行网格买入"""
        cost = order.shares * order.fill_price * commission_rate
        self.grid_position.add(order.shares, order.fill_price, cost)
        
        self.trade_history.append({
            'date': order.timestamp,
            'type': 'grid_buy',
            'price': order.fill_price,
            'shares': order.shares,
            'cost': cost,
            'level': order.grid_level
        })
    
    def execute_grid_sell(self, order: GridOrder, 
                          commission_rate: float = 0.0003,
                          stamp_tax_rate: float = 0.001):
        """执行网格卖出"""
        proceeds = order.shares * order.fill_price
        cost = proceeds * (commission_rate + stamp_tax_rate)
        self.grid_position.reduce(order.shares, order.fill_price, cost)
        
        self.trade_history.append({
            'date': order.timestamp,
            'type': 'grid_sell',
            'price': order.fill_price,
            'shares': order.shares,
            'cost': cost,
            'level': order.grid_level
        })
    
    def total_value(self, current_price: float) -> float:
        """总净值"""
        return (self.base_position.market_value(current_price) +
                self.grid_position.market_value(current_price))
    
    def get_status(self, current_price: float) -> dict:
        """获取持仓状态"""
        total_val = self.total_value(current_price)
        return {
            'total_value': total_val,
            'base_value': self.base_position.market_value(current_price),
            'grid_value': self.grid_position.market_value(current_price),
            'base_shares': self.base_position.shares,
            'grid_shares': self.grid_position.shares,
            'grid_cash': self.grid_position.cash,
            'total_return': (total_val / self.initial_cash - 1) * 100,
        }
```

---

### 2.4 RiskManager（风险管理器）

**职责**：实施风控规则，防止过度交易或极端情况

```python
@dataclass
class RiskConfig:
    """风险配置"""
    # 资金风控
    max_position_ratio: float = 0.95        # 最大持仓比例
    min_cash_reserve: float = 0.05          # 最小现金储备
    
    # 价格风控
    max_drawdown_stop: Optional[float] = None  # 最大回撤止损（如 -0.30）
    circuit_breaker: Optional[float] = None    # 单日跌幅熔断（如 -0.10）
    
    # 交易风控
    max_trades_per_day: int = 10            # 每日最大交易数
    min_trade_interval: int = 0             # 最小交易间隔（分钟）
    
    # 趋势风控
    enable_trend_filter: bool = False       # 启用趋势过滤
    trend_ma_period: int = 120              # 趋势均线周期
    only_sell_in_uptrend: bool = False      # 仅在上升趋势卖出


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.trades_today = 0
        self.last_trade_time: Optional[pd.Timestamp] = None
        self.peak_value = 0.0
        self.ma_series: Optional[pd.Series] = None
    
    def check_can_trade(self, current_date: pd.Timestamp, 
                       current_value: float,
                       current_price: float) -> tuple[bool, str]:
        """
        检查是否允许交易
        
        Returns:
            (can_trade, reason): 是否允许交易及原因
        """
        # 检查每日交易次数
        if self.trades_today >= self.config.max_trades_per_day:
            return False, "Exceeded max trades per day"
        
        # 检查最大回撤
        if self.config.max_drawdown_stop:
            self.peak_value = max(self.peak_value, current_value)
            current_drawdown = (current_value / self.peak_value - 1)
            if current_drawdown < self.config.max_drawdown_stop:
                return False, f"Max drawdown reached: {current_drawdown:.2%}"
        
        # 检查趋势过滤
        if self.config.enable_trend_filter and self.ma_series is not None:
            if current_date in self.ma_series.index:
                ma_value = self.ma_series[current_date]
                is_uptrend = current_price > ma_value
                if self.config.only_sell_in_uptrend and not is_uptrend:
                    return False, "Not in uptrend, sell blocked"
        
        return True, "OK"
    
    def update_ma(self, price_series: pd.Series):
        """更新趋势均线"""
        if self.config.enable_trend_filter:
            self.ma_series = price_series.rolling(
                self.config.trend_ma_period, min_periods=1
            ).mean()
    
    def record_trade(self, trade_time: pd.Timestamp):
        """记录交易"""
        self.trades_today += 1
        self.last_trade_time = trade_time
    
    def reset_daily(self):
        """每日重置"""
        self.trades_today = 0
```

---

### 2.5 Backtester（回测引擎）

**职责**：串联所有模块，执行回测逻辑

```python
@dataclass
class BacktestConfig:
    """回测配置"""
    ticker: str
    start_date: str
    end_date: str
    initial_cash: float = 1_000_000.0
    base_ratio: float = 0.5
    
    # 费率
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    
    # 模块配置
    grid_config: GridConfig = field(default_factory=GridConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)


class GridBacktester:
    """网格策略回测器"""
    
    def __init__(self, config: BacktestConfig, data_provider: DataProvider):
        self.config = config
        self.provider = data_provider
        
        # 初始化模块
        self.position_manager: Optional[PositionManager] = None
        self.grid_engine: Optional[GridEngine] = None
        self.risk_manager = RiskManager(config.risk_config)
        
        # 回测结果
        self.equity_curve: List[dict] = []
        self.metrics: dict = {}
    
    def run(self) -> pd.DataFrame:
        """运行回测"""
        print(f"Fetching data for {self.config.ticker}...")
        price_series = self.provider.fetch_price(
            self.config.ticker,
            self.config.start_date,
            self.config.end_date
        )
        
        if price_series.empty:
            raise ValueError(f"No data for {self.config.ticker}")
        
        print(f"Backtest period: {price_series.index[0]} to {price_series.index[-1]}")
        print(f"Total days: {len(price_series)}")
        
        # 初始化
        initial_price = price_series.iloc[0]
        self.position_manager = PositionManager(
            self.config.initial_cash,
            self.config.base_ratio
        )
        self.position_manager.initialize_base_position(initial_price)
        
        self.grid_engine = GridEngine(self.config.grid_config, initial_price)
        self.risk_manager.peak_value = self.config.initial_cash
        
        # 更新趋势均线
        if self.config.risk_config.enable_trend_filter:
            self.risk_manager.update_ma(price_series)
        
        # 逐日模拟
        last_date = None
        for date, current_price in price_series.items():
            # 每日重置风控计数
            if last_date and date.date() != last_date.date():
                self.risk_manager.reset_daily()
            last_date = date
            
            # 获取当前状态
            current_value = self.position_manager.total_value(current_price)
            available_cash = self.position_manager.grid_position.cash
            
            # 风控检查
            can_trade, reason = self.risk_manager.check_can_trade(
                date, current_value, current_price
            )
            
            if can_trade:
                # 更新网格引擎
                filled_buys, filled_sells = self.grid_engine.update(
                    current_price, date, available_cash
                )
                
                # 执行买入
                for buy_order in filled_buys:
                    self.position_manager.execute_grid_buy(
                        buy_order, self.config.commission_rate
                    )
                    self.risk_manager.record_trade(date)
                
                # 执行卖出
                for sell_order in filled_sells:
                    self.position_manager.execute_grid_sell(
                        sell_order,
                        self.config.commission_rate,
                        self.config.stamp_tax_rate
                    )
                    self.risk_manager.record_trade(date)
            
            # 记录净值曲线
            status = self.position_manager.get_status(current_price)
            self.equity_curve.append({
                'date': date,
                'price': current_price,
                **status
            })
        
        # 计算指标
        self._calculate_metrics(price_series)
        
        return pd.DataFrame(self.equity_curve).set_index('date')
    
    def _calculate_metrics(self, price_series: pd.Series):
        """计算回测指标"""
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        
        # 基准（买入并持有）
        bh_shares = self.config.initial_cash / price_series.iloc[0]
        bh_value = price_series * bh_shares
        
        # 计算指标
        final_value = equity_df['total_value'].iloc[-1]
        total_return = (final_value / self.config.initial_cash - 1) * 100
        
        days = (equity_df.index[-1] - equity_df.index[0]).days
        cagr = ((final_value / self.config.initial_cash) ** (365.0 / days) - 1) * 100
        
        # 最大回撤
        rolling_max = equity_df['total_value'].cummax()
        drawdown = equity_df['total_value'] / rolling_max - 1
        max_dd = drawdown.min() * 100
        
        # 基准对比
        bh_final = bh_value.iloc[-1]
        bh_cagr = ((bh_final / self.config.initial_cash) ** (365.0 / days) - 1) * 100
        
        self.metrics = {
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'benchmark_cagr': bh_cagr,
            'alpha': cagr - bh_cagr,
            'total_trades': len(self.position_manager.trade_history),
        }
    
    def print_summary(self):
        """打印回测摘要"""
        print("\n" + "="*70)
        print(f"Backtest Summary: {self.config.ticker}")
        print("="*70)
        print(f"Initial Capital:    {self.config.initial_cash:>15,.0f}")
        print(f"Final Value:        {self.metrics['final_value']:>15,.0f}")
        print(f"Total Return:       {self.metrics['total_return']:>14.2f}%")
        print(f"CAGR:               {self.metrics['cagr']:>14.2f}%")
        print(f"Max Drawdown:       {self.metrics['max_drawdown']:>14.2f}%")
        print(f"Benchmark CAGR:     {self.metrics['benchmark_cagr']:>14.2f}%")
        print(f"Alpha:              {self.metrics['alpha']:>14.2f}%")
        print(f"Total Trades:       {self.metrics['total_trades']:>15}")
        print("="*70)
```

---

## 3. 使用示例

### 3.1 基础使用（红利指数）

```python
from grid_framework import *

# 1. 配置数据源
provider = TushareProvider(token=os.getenv('TUSHARE_TOKEN'))

# 2. 配置策略
config = BacktestConfig(
    ticker='h00922.CSI',
    start_date='2018-01-01',
    end_date='2025-12-31',
    initial_cash=1_000_000,
    base_ratio=0.5,
    grid_config=GridConfig(
        mode=GridMode.FIXED,
        spacing=0.02,
        num_grids=10,
        grid_unit=50_000
    ),
    risk_config=RiskConfig(
        max_drawdown_stop=-0.30,
        enable_trend_filter=False
    )
)

# 3. 运行回测
backtester = GridBacktester(config, provider)
results = backtester.run()
backtester.print_summary()

# 4. 导出结果
results.to_csv('grid_backtest_results.csv')
```

### 3.2 多标的对比

```python
tickers = ['h00922.CSI', '510300.SH', '515080.SH', '159915.SZ']
results_comparison = {}

for ticker in tickers:
    config.ticker = ticker
    backtester = GridBacktester(config, provider)
    results_comparison[ticker] = backtester.run()

# 生成对比报告
compare_results(results_comparison)
```

### 3.3 动态网格（基于 ATR）

```python
config = BacktestConfig(
    ticker='159915.SZ',  # 创业板 ETF（波动较大）
    grid_config=GridConfig(
        mode=GridMode.DYNAMIC_ATR,
        spacing=2.0,  # 2 倍 ATR
        num_grids=15,
        spacing_func=lambda atr, price: 2 * atr / price  # 自定义间距函数
    )
)
```

### 3.4 趋势过滤网格

```python
config = BacktestConfig(
    ticker='h00922.CSI',
    grid_config=GridConfig(
        mode=GridMode.FIXED,
        spacing=0.02
    ),
    risk_config=RiskConfig(
        enable_trend_filter=True,
        trend_ma_period=120,
        only_sell_in_uptrend=True  # 仅在上升趋势卖出
    )
)
```

---

## 4. 文件结构

```
nl2quant/
├── strategies/
│   ├── __init__.py
│   ├── grid/
│   │   ├── __init__.py
│   │   ├── engine.py          # GridEngine
│   │   ├── config.py          # GridConfig, GridMode
│   │   └── orders.py          # GridOrder
│   ├── position.py            # PositionManager, Position
│   └── risk.py                # RiskManager, RiskConfig
├── data/
│   ├── __init__.py
│   ├── provider.py            # DataProvider 基类
│   ├── tushare_provider.py    # TushareProvider
│   ├── akshare_provider.py    # AKShareProvider
│   └── csv_provider.py        # CSVProvider
├── backtest/
│   ├── __init__.py
│   ├── engine.py              # GridBacktester
│   ├── metrics.py             # 指标计算
│   └── report.py              # 报告生成
├── test/
│   ├── test_grid_engine.py
│   ├── test_position_manager.py
│   └── backtest_examples.py   # 各种使用示例
└── examples/
    ├── example_basic.py       # 基础网格示例
    ├── example_dynamic.py     # 动态网格示例
    └── example_multi_asset.py # 多资产对比
```

---

## 5. 实施路线图

### Phase 1: 核心框架（1-2 周）
- [x] 定义抽象接口
- [ ] 实现 `GridEngine`（固定间距）
- [ ] 实现 `PositionManager`
- [ ] 实现基础 `Backtester`
- [ ] 单元测试

### Phase 2: 数据层（1 周）
- [ ] 实现 `TushareProvider`
- [ ] 实现 `AKShareProvider`
- [ ] 实现 `CSVProvider`
- [ ] 数据缓存机制

### Phase 3: 风控层（1 周）
- [ ] 实现 `RiskManager`
- [ ] 趋势过滤
- [ ] 回撤止损
- [ ] 交易频率控制

### Phase 4: 高级功能（2 周）
- [ ] 动态网格（ATR/波动率）
- [ ] 移动止盈
- [ ] 金字塔加仓
- [ ] 多标的轮动

### Phase 5: 工具与文档（1 周）
- [ ] 参数优化工具
- [ ] 可视化报告
- [ ] 完整文档
- [ ] 使用示例集

---

## 6. 关键优势

| 特性 | 说明 | 好处 |
|------|------|------|
| **标的无关** | 统一接口支持任意交易品种 | 一套代码适配股票/ETF/期货/币圈 |
| **策略解耦** | 网格逻辑、持仓、风控独立 | 易于测试、调试、扩展 |
| **高度可配** | 所有参数外部配置 | 快速验证不同策略变体 |
| **数据独立** | 多数据源适配器 | 不依赖单一数据供应商 |
| **可扩展性** | 模块化设计 | 轻松添加新策略类型 |

---

## 7. 未来扩展方向

### 7.1 策略增强
- **马丁格尔网格**：递增式加仓（风险较高）
- **无限网格**：不设上下界限，追随趋势
- **配对网格**：同时做多/做空相关资产
- **期权增强网格**：卖出虚值期权增加收益

### 7.2 智能化
- **机器学习动态参数**：基于市场状态自动调整间距
- **强化学习优化**：RL agent 学习最优网格配置
- **情绪指标**：结合 VIX、Put/Call Ratio 等市场情绪

### 7.3 实盘对接
- **交易所 API 集成**：直接下单执行
- **实时监控**：WebSocket 实时价格推送
- **钉钉/微信通知**：交易提醒
- **风险预警**：达到阈值自动暂停

---

*设计版本：v1.0*  
*最后更新：2025-12-09*  
*作者：nl2quant Team*
