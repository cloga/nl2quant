# 相关性分析 - 快速开始 (Quick Start)

## 30秒了解这个工具

你有两个股票/基金/指数，想知道它们是否相关？用这个工具！

```bash
python scripts/correlation_cli.py --code1 600036.SH --code2 601166.SH
```

**会得到什么？**
- ✓ 相关系数 (是否同向)
- ✓ 是否适合配对交易 (协整检验)
- ✓ 价差分析 (套利机会)
- ✓ Granger因果 (谁领先谁)
- ✓ 滚动相关性 (关系是否稳定)
- ✓ 尾部依赖 (危机中如何联动)

## 常用命令

### 1. 最简单用法 (综合分析)

```bash
# 分析招商银行 vs 兴业银行
python scripts/correlation_cli.py --code1 600036.SH --code2 601166.SH

# 分析黄金现货 vs 黄金股
python scripts/correlation_cli.py --code1 518880.SH --code2 159562.SZ

# 分析某股票 vs 某指数
python scripts/correlation_cli.py --code1 000001.SZ --code2 399300.SZ
```

### 2. 保存报告

```bash
python scripts/correlation_cli.py \
  --code1 600036.SH \
  --code2 601166.SH \
  --output my_analysis.md
```

报告会保存到 `my_analysis.md`，可用任何编辑器打开。

### 3. 快速判断 (单维度)

**只想看相关系数？**
```bash
python scripts/correlation_cli.py --code1 600036.SH --code2 601166.SH --dimension linear
```

**只想检查是否协整？**
```bash
python scripts/correlation_cli.py --code1 600036.SH --code2 601166.SH --dimension cointegration
```

**只想看是否有极端事件？**
```bash
python scripts/correlation_cli.py --code1 600036.SH --code2 601166.SH --dimension tail
```

## 快速决策指南

### 我想做配对交易，这两个标的行吗？

运行分析，看**维度2 (协整)** 的结果：

```
Engle-Granger协整检验:
  - 结论: 存在协整关系 ✓ (可以)
或
  - 结论: 不存在协整关系 ✗ (不行)
```

**同时检查价差分析:**
```
极端事件频率: 11.09% ✓ (频率高，机会多)
或
极端事件频率: 0.5% ✗ (机会太少)
```

**+Z-Score:**
```
当前Z-Score: -1.53 
→ |z| < 2 = 目前无交易信号
→ |z| > 2 = 可以考虑入场
```

### 这两个标的能否作为组合对冲？

运行分析，看**维度5 (尾部依赖)**：

```
左尾依赖: 58.6%
→ 58.6% > 50% = 暴跌时联动强，对冲效果有限
→ 20% < 左尾依赖 < 50% = 对冲效果中等
→ 左尾依赖 < 20% = 对冲效果好 (危机中不同时下跌)
```

### 两个标的哪个更强？

看**维度1 (Beta系数)**：

```
Beta系数: 0.82
→ β < 1 = 标的B波动比标的A小，更稳定
→ β > 1 = 标的B波动比标的A大，更敏感
```

### 这两个标的的关系稳定吗？

看**维度4 (滚动相关性)**：

```
标准差: 0.1585
→ σ < 0.1 = 很稳定
→ 0.1 < σ < 0.2 = 中等稳定 ✓
→ σ > 0.3 = 不稳定，关系经常变化
```

## 工具识别逻辑

工具能自动识别你输入的是什么类型：

| 代码前缀 | 识别为 | 例子 |
|---------|------|------|
| 1xxxxx, 5xxxxx, 16xxxx | **基金** | 110022.SH (易方达), 510050.SH (50ETF) |
| 0xxxxx, 3xxxxx, 6xxxx | **股票** | 000001.SZ (平安), 300750.SZ (宁德时代), 600036.SH (招商银行) |
| 4xxxxx, 9xxxx | **指数** | 399300.SZ (沪深300), 000001.SH (上证指数) |

## 三种常见用法

### 用法A: 配对交易筛选

**问题:** "我想找两个协整的股票做配对交易"

**解决方案:**
```bash
# 对多对股票批量分析
for pair in "600036.SH 601166.SH" "601398.SH 601939.SH" "000858.SZ 000568.SZ"
do
  codes=(${pair})
  python scripts/correlation_cli.py --code1 ${codes[0]} --code2 ${codes[1]} --dimension cointegration
done
```

**筛选标准:**
- ✓ Engle-Granger P值 < 0.05
- ✓ 极端事件频率 > 2%
- ✓ 当前Z-Score |z| > 1.5

### 用法B: 风险管理 (组合对冲)

**问题:** "我持有A股，想找一个不相关的B来对冲"

**解决方案:**
```bash
# 检查尾部依赖
python scripts/correlation_cli.py --code1 600036.SH --code2 110022.SH --dimension tail
```

**筛选标准:**
- ✓ 左尾依赖 < 0.4 (危机中不会同时下跌)
- ✓ 波动特性互补 (β值不同)

### 用法C: 市场分析

**问题:** "黄金现货和黄金股最近是否还联动？"

**解决方案:**
```bash
python scripts/correlation_cli.py --code1 518880.SH --code2 159562.SZ --dimension rolling
```

**查看:**
- 当前滚动相关系数
- 是否发现解耦事件
- 历史波动趋势

## 输出解读速查表

| 指标 | 说明 | 值范围 | 好的值 | 坏的值 |
|------|------|--------|--------|--------|
| **Pearson相关系数** | 线性相关 | [-1, 1] | 接近0.7-0.9 (配对) 或接近0 (对冲) | N/A |
| **P值** | 统计显著性 | [0, 1] | < 0.05 | > 0.05 |
| **Beta系数** | 相对波动 | > 0 | 取决于目标 | 只看绝对值意义 |
| **Engle-Granger P值** | 协整性 | [0, 1] | < 0.05 (配对) | > 0.05 |
| **Z-Score** | 价差离均值 | [-∞, +∞] | |z| > 2 (入场) | |z| < 1 (无信号) |
| **滚动相关系数σ** | 稳定性 | [0, ∞) | < 0.15 | > 0.3 |
| **左尾依赖** | 危机联动 | [0, 1] | < 0.4 (对冲) | > 0.7 |
| **极端事件频率** | 机会频率 | [0%, 100%] | > 5% (配对) | < 1% |

## 命令速查

```bash
# 基础用法
python scripts/correlation_cli.py --code1 A --code2 B

# 指定资产类型
python scripts/correlation_cli.py --code1 A --type1 stock --code2 B --type2 fund

# 保存报告
python scripts/correlation_cli.py --code1 A --code2 B --output report.md

# 线性分析
python scripts/correlation_cli.py --code1 A --code2 B --dimension linear

# 协整分析
python scripts/correlation_cli.py --code1 A --code2 B --dimension cointegration

# Granger因果
python scripts/correlation_cli.py --code1 A --code2 B --dimension granger

# 滚动相关（指定窗口）
python scripts/correlation_cli.py --code1 A --code2 B --dimension rolling --window 20

# 尾部依赖
python scripts/correlation_cli.py --code1 A --code2 B --dimension tail
```

## 更多帮助

```bash
python scripts/correlation_cli.py --help
```

## 相关文档

- **详细指南**: `doc/CORRELATION_ANALYZER_GUIDE.md`
- **方法论**: `design/correlation_analyisi.md`
- **源代码**: `app/correlation_analyzer.py`

---

Happy Analyzing! 🚀
