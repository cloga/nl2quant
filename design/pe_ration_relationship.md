这是一个非常扎实的总结。为了方便你在代码、Excel 或量化系统中直接落地执行，我将上述所有讨论内容整合成一份\*\*《全口径市盈率计算与应用标准文档》\*\*。

这份文档涵盖了数据源选取、计算公式、修正算法以及异常值处理，可以直接作为开发或分析的“需求说明书”。

-----

# 全口径市盈率 (P/E) 计算标准手册

### 0. 核心定义与数据清洗原则

在开始计算任何 PE 之前，必须统一“分子”和“分母”的标准，以确保数据纯净。

  * **分子 (P - Price)**：统一使用**总市值 (Total Market Value)**。
      * *注意：* 不建议使用单股股价，因为涉及除权除息（复权）问题，直接用总市值更简单准确。
  * **分母 (E - Earnings)**：建议优先使用**扣非归母净利润**。
      * *逻辑：* 只有剔除了一次性损益（卖地、政府补助等）的利润，外推才有意义。
      * *处理：* 若净利润 $\le 0$，PE 标记为 `NaN` 或 `Null`（负值无估值意义）。

-----

### 1. 过去视角的计算 (The Past)

**数据属性：** 100% 事实，无主观成分。

| 指标名称 | 英文缩写 | 计算公式 | 核心用途 |
| :--- | :--- | :--- | :--- |
| **静态市盈率** | PE (LYR) | $$\frac{\text{当前总市值}}{\text{上一年年报净利润}}$$ | **保底估值**<br>用于查看公司在最保守情况下的估值水平。 |
| **滚动市盈率** | PE (TTM) | $$\frac{\text{当前总市值}}{\sum(\text{最近4个季度净利润})}$$ | **绝对锚点**<br>价值投资的核心基准，包含了四季度的完整周期。 |

-----

### 2. 现在视角的计算 (The Present / Extrapolation)

**数据属性：** 基于最新财报的数学推导，需警惕季节性陷阱。

这里有三种算法，从“粗糙”到“精细”，建议根据行业属性自动切换：

#### 算法 A：简单年化法 (Simple Annualization)

  * **适用：** 业务极其平稳、无季节性的行业（水电、高速、银行）。
  * **公式（以Q1为例）：**
    $$E_{pred} = \text{Q1净利} \times 4$$
    $$\text{PE}_{\text{Linear}} = \frac{\text{市值}}{E_{pred}}$$
  * **风险：** 遇到季节性行业（如白酒Q1、电商Q4）会产生剧烈误差。

#### 算法 B：历史权重法 (Historical Weighting) —— **推荐默认使用**

  * **适用：** 绝大多数有季节性波动的传统行业。
  * **步骤：**
    1.  计算过去3年该季度（如Q1）占全年利润的平均比例 $R_{avg}$。
    2.  外推全年利润：$E_{pred} = \text{本期净利} \div R_{avg}$
  * **公式：**
    $$\text{PE}_{\text{Weighted}} = \frac{\text{市值}}{(\text{本期净利} / \text{历史平均占比})}$$

#### 算法 C：同比增速叠加法 (Growth Overlay)

  * **适用：** 高成长股，且去年基数正常的公司。
  * **步骤：**
    1.  计算本期同比增速：$g = (\text{本期} - \text{去年同期}) / |\text{去年同期}|$
    2.  叠加到去年全年：$E_{pred} = \text{去年全年净利} \times (1 + g)$
  * **约束：** 必须设置阈值（如 $|g| > 50\%$ 时失效），防止基数效应导致的数值爆炸。

-----

### 3. 未来视角的计算 (The Future)

**数据属性：** 机构的一致预期，代表市场共识。

| 指标名称 | 数据源 | 计算公式 | 核心用途 |
| :--- | :--- | :--- | :--- |
| **预测市盈率 (均值)** | 研报数据 | $$\frac{\text{当前总市值}}{\text{分析师预测净利润之平均值}}$$ | **情绪水位**<br>代表市场整体看多/看空的程度。 |
| **预测市盈率 (中位数)** | 研报数据 | $$\frac{\text{当前总市值}}{\text{分析师预测净利润之中位数}}$$ | **目标定价**<br>剔除了极端值的最靠谱预期，用于判断未来空间。 |

-----

### 4. 综合分析系统 (The Dashboard Logic)

将上述指标整合到一个表格或看板中时，建议采用以下 **“三色预警逻辑”** 来辅助决策：

**逻辑一：成长加速侦测 (戴维斯双击前兆)**

  * **条件：** $\text{线性外推PE} \ll \text{TTM PE}$ （例如低 30% 以上）
  * **解读：** 公司最新业绩大爆发，且该爆发甚至超过了过去一年的平均水平。
  * **操作：** 重点关注，确认是否为基本面反转。

**逻辑二：成长伪证/季节性陷阱**

  * **条件：** $\text{线性外推PE} \ll \text{TTM PE}$ **但** $\text{历史权重PE} \approx \text{TTM PE}$
  * **解读：** 看起来是业绩爆发，其实只是到了旺季（如空调到了夏天）。
  * **操作：** 忽略波动，继续持有。

**逻辑三：预期差套利**

  * **条件：** $\text{机构预测PE (中位数)} \ll \text{TTM PE}$
  * **解读：** 虽然现在看起来不便宜（TTM高），但机构一致认为明年会赚大钱。
  * **操作：** 研究机构逻辑是否成立（如新产能投产、新产品上市）。

### 附：Python 代码实现结构参考

```python
class PECalculator:
    def __init__(self, current_mv, financial_data, forecasts):
        self.mv = current_mv
        self.fin = financial_data # 包含历史分季度数据
        self.forecasts = forecasts # 包含机构预测数据

    def calc_ttm(self):
        # 计算过去4个季度之和
        return self.mv / self.fin.last_4q_sum()

    def calc_linear_extrapolation(self, method='weighted'):
        current_profit = self.fin.current_period_profit()
        
        if method == 'simple':
            # 简单年化
            annualized = current_profit * (4 / self.fin.current_quarter_num())
            
        elif method == 'weighted':
            # 历史权重法
            ratio = self.fin.get_historical_quarter_ratio()
            annualized = current_profit / ratio
            
        elif method == 'growth':
            # 增速叠加法
            growth = self.fin.get_yoy_growth()
            last_year_total = self.fin.get_last_year_total()
            annualized = last_year_total * (1 + growth)
            
        return self.mv / annualized

    def calc_forward_consensus(self):
        # 计算机构预测中位数
        median_profit = self.forecasts.median()
        return self.mv / median_profit
```