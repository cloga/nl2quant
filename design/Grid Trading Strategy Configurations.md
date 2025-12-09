# 网格交易策略配置项清单 (Grid Trading Strategy Configurations)

设计一个完善的网格交易策略，通常涵盖 **基础设置、网格逻辑、资金管理** 和 **风控保护** 四大模块。

### 1. 基础标的与方向 (Basic Settings)
决定策略的最基本属性，“在哪里跑”以及“怎么跑”。

* **标的代码 (Symbol):**
    * 示例：`159915.SZ` (创业板ETF) 或 `BTC/USDT`。
* **网格方向 (Grid Direction):**
    * **Neutral (中性震荡):** 适合横盘。预先持有部分底仓，涨卖跌买，双向套利。
    * **Long (做多网格):** 适合震荡上行。只买入不卖空（或使用合约做多）。
    * **Short (做空网格):** 适合震荡下行。只卖出不开多（通常用于合约）。

### 2. 价格区间与密度 (Range & Density)
网格的“骨架”，决定了覆盖范围和交易频率。

* **价格区间上限 (Upper Limit Price):**
    * 价格高于此数值时，停止买入（甚至可能清仓）。
* **价格区间下限 (Lower Limit Price):**
    * 价格低于此数值时，停止卖出（防止低位割肉或无限补仓）。
* **网格数量 (Grid Quantity / Count):**
    * 将区间分为多少格（例如：100格）。
    * *权衡：* 格数多 = 成交频次高、单格利润低；格数少 = 成交频次低、单格利润高。
* **网格类型 (Grid Type):**
    * **Arithmetic (等差网格):** 每格价格差固定（如每跌 0.1 元买入）。适合价格较低或波动较小的标的。
    * **Geometric (等比网格):** 每格百分比固定（如每跌 1% 买入）。适合价格跨度大或波动剧烈的标的。

### 3. 资金与仓位管理 (Money Management)
决定“投入多少钱”以及“每笔买多少”。

* **总投入资金 (Total Investment):**
    * 策略账户的总本金（如 100,000 元）。
* **单格买卖数量 (Amount Per Grid):**
    * **Fixed Quantity:** 每次买卖固定股数/币数（如每次 1000 股）。
    * **Fixed Notional:** 每次买卖固定金额（如每次买 2000 元的货）。
* **初始持仓 (Initial Position):**
    * 中性网格启动时，通常需先买入约 50% 的仓位（底仓），以便上涨时有货可卖。
* **杠杆倍数 (Leverage):**
    * (仅限期货/期权) 设置 1x, 2x, 5x 等。

### 4. 触发与停止条件 (Trigger & Stop)
决定策略何时“上班”和何时“下班”。

* **触发启动价格 (Trigger Price):**
    * (可选) 策略不立即启动，等待价格达到某一点位（如突破压力位）才激活。
* **止损价格 (Stop Loss Price):**
    * **关键配置**。当价格跌破此线，全部卖出止损，防止从“震荡”变成“单边暴跌”。
* **止盈价格 (Take Profit Price):**
    * (可选) 当价格涨破此线，全部清仓止盈。

### 5. 进阶/动态配置 (Advanced Features)
解决普通网格“破网”问题的高级功能。

* **移动/无限网格 (Trailing / Infinite Grid):**
    * **Trailing Up:** 当价格突破上限时，网格区间整体向上平移，实现“让利润奔跑”。
    * **Trailing Down:** (高风险) 当价格跌破下限时，网格整体下移。
* **单格利润率 (Profit per Grid):**
    * 需监控此指标。公式：`单格涨幅 - (买入费率 + 卖出费率)`。建议 > 0.3% 以覆盖滑点。
* **挂单模式 (Order Type):**
    * **Maker (挂单):** 挂在盘口等待成交，费率低，易踏空。
    * **Taker (吃单):** 市价立即成交，费率高，保证成交。

---

### 配置示例 (JSON Structure)

```json
{
  "strategy_name": "ChiNext_ETF_Grid_v1",
  "symbol": "159915.SZ",
  "capital": {
    "total_investment": 100000,
    "currency": "CNY"
  },
  "grid_settings": {
    "direction": "NEUTRAL",
    "type": "GEOMETRIC",
    "upper_limit": 2.500,
    "lower_limit": 1.800,
    "grid_count": 50,
    "amount_per_grid": 2000
  },
  "risk_control": {
    "trigger_price": null,
    "stop_loss": 1.700,
    "take_profit": 2.800
  },
  "advanced": {
    "trailing_up": true,
    "slippage_tolerance": 0.002
  }
}