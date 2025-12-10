# A股配对交易标的筛选程序 - 使用指南

## 概述

本程序基于 **Pairs Trading.md** 中的"用无监督学习聚类选择配对交易的标的完整方案"开发，实现了以下功能：

1. **PCA 降维** - 从高维收益率数据中提取核心特征
2. **DBSCAN 聚类** - 自动发现相似走势的股票
3. **协整检验** - 在聚类内寻找具有协整关系的股票对
4. **可视化分析** - t-SNE 聚类可视化和配对结果展示

---

## 快速开始

### 方式一：使用 Streamlit UI

```bash
cd d:\project\nl2quant
streamlit run main.py
```

然后在浏览器中点击左侧菜单 `配对交易筛选器` (4_Pairs_Screener.py)

**优势**：
- 交互式参数调整
- 实时可视化
- 结果即时导出

### 方式二：使用命令行工具

```bash
# 使用预定义的沪深300池子
python pairs_screener.py --pool hs300 --days 365

# 使用自定义股票代码
python pairs_screener.py --codes 601398,601939,601288,600519 --days 180

# 保存结果为JSON
python pairs_screener.py --pool banks --output results.json --csv pairs_result.csv
```

**优势**：
- 快速批量处理
- 便于脚本集成
- 支持定时任务

---

## 算法流程详解

### 第一阶段：数据准备

```python
screener = PairsScreener(
    start_date="20240101",  # 开始日期
    end_date="20241231"     # 结束日期
)

# 获取股票价格数据
price_df = screener.fetch_stock_data(codes=['601398', '601939', ...])
# 输出: DataFrame (日期 × 股票), 自动处理缺失值和复权
```

**数据说明**：
- 使用 `DCABacktestEngine.fetch_etf_close()` 获取数据
- 自动从 Tushare 获取复权收盘价
- 过滤掉数据不足的股票（<50天）

### 第二阶段：收益率计算

```python
returns_df = screener.compute_returns(price_df)
# 使用对数收益率: log(P_t / P_t-1)
# 输出: DataFrame (日期 × 股票)
```

**为什么用对数收益率？**
- 更符合金融数据的统计性质
- 方便计算复合收益
- 对极端值更稳健

### 第三阶段：PCA 降维

```python
X_pca, pca = screener.perform_pca(
    returns_df,
    n_components=15  # 保留15个主成分
)
# 输出: 
#   - X_pca: (股票数 × 15) 的降维矩阵
#   - pca: PCA模型对象（包含解释方差等信息）
```

**参数说明**：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `n_components` | 保留的主成分个数 | 10-20 |
| 解释方差比 | 累计解释方差占比 | 80%-90% |

**直观理解**：
- 原始数据：每只股票由 250 个交易日的收益率定义（250维）
- PCA降维后：每只股票由 15 个"因子"定义（15维）
- 这15个因子捕捉了原始数据 85%+ 的信息量

### 第四阶段：DBSCAN 聚类

```python
labels = screener.perform_dbscan(
    X_pca,
    eps=0.5,        # 邻域半径
    min_samples=2   # 成为核心点的最小邻居数
)
# 输出: 长度为股票数的标签数组
#      -1 表示噪音点，0,1,2... 表示聚类ID
```

**参数调优指南**：

| eps值 | 簇数 | 噪音 | 场景 |
|-------|-----|------|------|
| 0.2-0.3 | 很多 | 少 | 过度分割（可能太细） |
| 0.4-0.6 | 适中 | 适中 | **推荐** |
| 0.8-1.0 | 很少 | 多 | 欠分割（可能太粗） |

**调参建议**：
```bash
# 如果想要更多小簇（比如行业分割更细）
python pairs_screener.py --pool hs300 --eps 0.3

# 如果想要更少大簇（比如跨行业关联）
python pairs_screener.py --pool hs300 --eps 0.8
```

### 第五阶段：协整检验

在每个聚类内部，进行两两的协整检验：

```python
pairs_df = screener.find_cointegrated_pairs(returns_df, labels)
# 输出: DataFrame
#   stock_a: 股票A代码
#   stock_b: 股票B代码
#   correlation: Pearson相关系数 (0.85-1.0)
#   coint_pvalue: 协整检验的p值 (<0.05为显著)
#   coint_score: Engle-Granger协整统计量
```

**关键指标**：

| 指标 | 含义 | 筛选标准 |
|------|------|---------|
| `correlation` | 两只股票的线性相关程度 | ≥ 0.85 |
| `coint_pvalue` | 协整检验的p值 | < 0.05（拒绝不相关原假设） |
| `coint_score` | Engle-Granger统计量 | 越负越好（越显著） |

**为什么需要协整检验？**
- **相关性高 ≠ 协整**
  - 相关系数只衡量线性关系强度
  - 两只股票可能"一起涨但差距越来越大"
- **协整才是配对交易的灵魂**
  - 证明两只股票的价差存在均值回归性质
  - 价差偏离时才有交易机会

### 第六阶段：可视化

```python
# t-SNE 可视化聚类结果
cluster_fig = screener.visualize_clusters(X_pca, labels, stock_codes)
```

**图表解读**：
- 每个点 = 一只股票
- 相同颜色 = 同一聚类
- 距离近 = 走势相似
- 孤立点（通常为 -1） = 噪音（独特的走势）

---

## 使用场景

### 场景 1：快速扫描沪深300寻找配对机会

```bash
python pairs_screener.py --pool hs300 --days 365 --csv pairs.csv
```

**预期结果**：
- 找到 5-30 对协整配对（取决于市场状况）
- 运行时间：2-5 分钟（取决于网络）

### 场景 2：专注某个行业（如银行股）

```bash
python pairs_screener.py --pool banks --days 180 --output banks.json
```

**特点**：
- 行业内股票较为同质
- 通常能找到更多强协整关系
- 更适合实盘交易（避免基本面不匹配）

### 场景 3：调优参数以发现隐形关联

```bash
# 增加PCA成分以保留更多细节
python pairs_screener.py --pool hs300 --n-components 20

# 减小eps以产生更多细小聚类
python pairs_screener.py --pool hs300 --eps 0.3

# 结合使用
python pairs_screener.py --pool hs300 --n-components 20 --eps 0.35
```

### 场景 4：长期回测（2年数据）

```bash
python pairs_screener.py --pool hs300 --days 750 --csv pairs_2year.csv
```

---

## 结果解释

### 配对结果示例

```
     stock_a  stock_b  correlation  coint_pvalue  coint_score
0     601398   601939       0.9876       0.00234      -3.2145
1     600519   000858       0.9654       0.01289      -2.8932
2     601166   601328       0.9543       0.03412      -2.5678
```

**这表示**：
1. 工商银行(601398)和建设银行(601939)
   - 相关系数 0.9876（极高的走势相似度）
   - p值 0.00234（非常显著，<0.05）
   - **结论**：强协整关系，是很好的配对机会

2. 贵州茅台(600519)和五粮液(000858)
   - 相关系数 0.9654
   - p值 0.01289
   - **结论**：白酒行业，协整显著

### 聚类详情

```
聚类0 (5只股票): 601398, 601939, 601288, 601169, 601658  → 银行股
聚类1 (4只股票): 600519, 000858, 000568, 601633          → 白酒股
聚类2 (6只股票): 600030, 601688, 601211, ...             → 证券股
...
噪音点 (-1):  [特殊股票，独特走势]
```

---

## 进阶：手动调用 API

### 基础用法

```python
from app.pairs_screener import PairsScreener

# 初始化
screener = PairsScreener(
    start_date="20240101",
    end_date="20241231"
)

# 运行完整筛选流程
results = screener.run(
    codes=['601398', '601939', '601288', ...],
    eps=0.5,
    n_components=15
)

# 获取结果
pairs_df = results['pairs']      # 协整配对
cluster_fig = results['cluster_fig']  # 可视化
labels = results['labels']       # 聚类标签
X_pca = results['X_pca']         # PCA降维后的数据
```

### 分步运行（用于自定义处理）

```python
# 1. 数据获取
price_df = screener.fetch_stock_data(codes)

# 2. 收益率计算
returns_df = screener.compute_returns(price_df)

# 3. PCA降维
X_pca, pca = screener.perform_pca(returns_df, n_components=15)

# 4. 聚类
labels = screener.perform_dbscan(X_pca, eps=0.5)

# 5. 协整检验
pairs_df = screener.find_cointegrated_pairs(returns_df, labels)

# 6. 可视化
fig = screener.visualize_clusters(X_pca, labels, screener.stock_codes)

# 现在可以自定义处理 pairs_df、fig等
```

---

## 常见问题

### Q1: 没找到协整配对怎么办？

**原因分析**：
1. **股票池太小或相似度低** → 增加股票数量或选择同行业股票
2. **时间窗口太短** → 增加 `--days` 参数（建议≥180）
3. **eps参数不合适** → 尝试调整 eps（建议从0.4-0.6开始）
4. **数据质量问题** → 检查股票是否停牌或上市不足一年

**调试步骤**：
```bash
# 步骤1：先用小范围测试
python pairs_screener.py --pool banks --days 180

# 步骤2：如果成功，扩大范围
python pairs_screener.py --pool hs300 --days 365

# 步骤3：如果仍无结果，调整参数
python pairs_screener.py --pool hs300 --eps 0.6 --n-components 18
```

### Q2: 找到的配对太多了，怎么筛选最好的？

配对结果已按 `correlation` 降序排序，Top5通常是最强的：

```python
pairs_df = pairs_df.sort_values('correlation', ascending=False)
best_pairs = pairs_df.head(5)  # 最好的5对

# 或按p值筛选（p值越小越显著）
significant_pairs = pairs_df[pairs_df['coint_pvalue'] < 0.01]
```

### Q3: 能否只找某个行业的配对？

**可以！** 使用 `--pool` 参数：

```bash
# 银行股
python pairs_screener.py --pool banks

# 或自定义：
python pairs_screener.py --codes 600519,000858,000568,601633,603198
```

### Q4: 结果的稳定性如何？

**影响因素**：
- ✅ 时间窗口越长越稳定（推荐≥1年）
- ✅ 协整p值越小越稳定（<0.01最佳）
- ❌ DBSCAN对eps参数敏感，小改变可能改变聚类
- ❌ 市场制度变化（融券额度、涨跌停）会破坏协整

**建议**：
- 定期（每月或每季度）重新筛选
- 只交易p值<0.01的配对
- 实盘前用历史数据回测该配对

---

## 输出文件说明

### CSV 格式

```csv
stock_a,stock_b,correlation,coint_pvalue,coint_score
601398,601939,0.9876,0.00234,-3.2145
600519,000858,0.9654,0.01289,-2.8932
```

### JSON 格式

```json
{
  "timestamp": "2024-12-10T10:30:00",
  "parameters": {
    "start_date": "20240101",
    "end_date": "20241231",
    "stocks_count": 50,
    "eps": 0.5,
    "n_components": 15
  },
  "pairs_count": 12,
  "pairs": [
    {
      "stock_a": "601398",
      "stock_b": "601939",
      "correlation": 0.9876,
      "coint_pvalue": 0.00234,
      "coint_score": -3.2145
    }
  ],
  "clusters": {
    "n_clusters": 8,
    "n_noise": 2
  }
}
```

---

## 性能与限制

| 项目 | 说明 |
|------|------|
| 最大股票数 | 300-500（受网络和内存限制） |
| 推荐股票数 | 50-150（平衡速度和覆盖面） |
| 运行时间 | 50只股票约2-3分钟 |
| 内存占用 | 50只股票约500MB |
| 网络请求 | O(n)，每只股票一个API请求 |

**优化建议**：
- 使用预定义的行业池（内置优化）
- 避免同时测试超过200只股票
- 可以分批运行再合并结果

---

## 参考资源

- **原理文档**：`design/Pairs Trading.md`
- **核心代码**：`app/pairs_screener.py`
- **UI界面**：`pages/4_Pairs_Screener.py`
- **CLI工具**：`pairs_screener.py`

---

## 许可与免责

本程序用于研究和教学目的。实际交易前需要：
1. ✅ 通过历史回测验证配对有效性
2. ✅ 考虑A股交易机制限制（T+1、融券难等）
3. ✅ 制定风险管理策略（止损、仓位控制）
4. ✅ 定期监控协整关系是否失效

**免责声明**：过去的协整关系不保证未来继续存在。一旦基本面逻辑改变（重组、暴雷等），配对关系可能瞬间失效。

