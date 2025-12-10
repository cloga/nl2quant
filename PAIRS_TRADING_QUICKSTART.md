# A股配对交易筛选器 - 快速启动指南

## ✅ 环境设置完成

已在虚拟环境中安装所有依赖：
- `scikit-learn` - PCA 降维和 DBSCAN 聚类
- `statsmodels` - 协整检验 (Cointegration Test)
- `plotly` - 交互式可视化

## 📋 可用的启动方式

### 1️⃣ 使用 Streamlit UI（推荐新手）

```bash
cd d:\project\nl2quant
.venv\Scripts\Activate.ps1
streamlit run main.py
```

然后在菜单中选择 **"配对交易筛选器"** (4_Pairs_Screener.py)

**特点**：
- 交互式参数调整
- 实时可视化聚类结果和配对列表
- 一键导出 CSV/JSON 结果

### 2️⃣ 使用命令行工具（快速批量处理）

```bash
cd d:\project\nl2quant
.venv\Scripts\Activate.ps1

# 快速测试（5只银行股，180天）
python pairs_screener.py --pool banks --days 180 --csv result_banks.csv

# 深度分析（沪深300，2年）
python pairs_screener.py --pool hs300 --days 750 --n-components 18 --output result_hs300.json

# 自定义代码
python pairs_screener.py --codes 601398,601939,601288,600519,000858 --days 365
```

### 3️⃣ 使用菜单式快速启动

```bash
cd d:\project\nl2quant
.venv\Scripts\Activate.ps1
python run_screener.py
```

菜单选项：
1. 预设方案（快速测试、深度分析、板块专项等）
2. 自定义配置
3. 查看帮助信息

### 4️⃣ Python API 调用

```python
from app.pairs_screener import PairsScreener
from datetime import datetime, timedelta

# 初始化
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

screener = PairsScreener(
    start_date.strftime("%Y%m%d"),
    end_date.strftime("%Y%m%d")
)

# 运行筛选（5只银行股）
codes = ['601398', '601939', '601288', '601166', '601328']
results = screener.run(codes, eps=0.5, n_components=15)

# 获取结果
pairs_df = results['pairs']  # 协整配对
cluster_fig = results['cluster_fig']  # 聚类可视化
labels = results['labels']  # 聚类标签

print(pairs_df)
cluster_fig.show()
```

## 🔍 参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--days` | 回溯天数 | 180-750 |
| `--eps` | DBSCAN 邻域半径 | 0.3-0.7 |
| `--n-components` | PCA 主成分数 | 12-20 |
| `--pool` | 预定义股票池 | hs300/banks/liquor |

## 📊 预定义股票池

- **hs300**: 沪深300 成分股（50只）
- **banks**: A股主要银行股（16只）
- **liquor**: A股主要白酒股（6只）

自定义: `--codes 601398,601939,...`

## 🎯 快速示例

### 示例 1: 快速找银行股的配对

```bash
python pairs_screener.py --pool banks --days 180
```

**预期结果**: 找到 2-5 对强协整配对（2-3分钟）

### 示例 2: 跨行业隐形关联

```bash
python pairs_screener.py --pool hs300 --eps 0.4 --n-components 18
```

**预期结果**: 发现可能跨行业的相似走势股票（3-5分钟）

### 示例 3: 长期稳定性分析

```bash
python pairs_screener.py --pool banks --days 750 --output banks_2year.json
```

**预期结果**: 验证配对在长期内是否保持协整（5-8分钟）

## 📈 输出文件

- **pairs_XXX.csv**: 配对结果（可用 Excel 打开）
- **pairs_XXX.json**: 完整结果（包含参数、聚类统计）

CSV 格式:
```
stock_a,stock_b,correlation,coint_pvalue,coint_score
601398,601939,0.9876,0.00234,-3.2145
```

## ❓ 常见问题

**Q: 没有找到配对怎么办？**

A: 尝试以下方案（按优先级）：
1. 增加股票数量或选择同行业股票
2. 增加回溯天数（`--days 365` → `--days 750`）
3. 调整 eps 参数（从 0.5 → 0.4 或 0.6）
4. 增加 PCA 成分数（`--n-components 20`）

**Q: Streamlit 版本更新是否兼容？**

A: 是的，使用的都是稳定 API，兼容最新版 Streamlit

**Q: 能否用其他行业的股票？**

A: 可以，直接使用 `--codes` 传入自定义代码列表

## 📖 详细文档

完整的原理、参数解释和高级用法见：

`doc/PAIRS_TRADING_GUIDE.md`

## 🧪 验证安装

```bash
python test_pairs_screener.py
```

所有测试通过表示环境正确配置。

---

**现在您可以开始使用A股配对交易筛选器了！**

推荐从 Streamlit UI 开始探索，然后根据需要切换到命令行工具。
