# 配对交易 (pairs-trading)
研究A股中的配对交易（Pairs Trading）是一个非常经典的统计套利方向，但在中国市场落地时，**除了数学模型，更要解决交易机制的限制（如融券难、T+1等）**。

以下是为你设计的一套从理论到落地的完整研究路线图：

### 第一阶段：核心概念与A股的特殊性

在开始写代码之前，你必须先明确A股环境下的两大难点，否则模型跑得再好，实盘也无法从容落地。

1.  **核心逻辑：**

      * 寻找两只历史上走势高度一致（**协整关系**）的股票（比如“招商银行”和“兴业银行”）。
      * 当它们的价格差（Spread）偏离历史均值过大时，买入“低估”的那只，做空“高估”的那只。
      * 等待价差回归均值，平仓获利。**这种策略赚的是价差回归的钱，与大盘涨跌无关（市场中性）。**

2.  **A股的特殊挑战（非常重要）：**

      * **做空限制（融券）：** 并非所有A股都能融券（做空），且融券成本高、券源不稳定。
          * *应对方案：* 选择ETF进行配对（ETF融券相对容易），或者采用\*\*“轮动持仓策略”\*\*（只做多，两只股票谁低估就持有谁，高估了就换仓，不加杠杆做空）。
      * **T+1交易：** 当天买入的股票由于价差迅速回归，当天无法卖出。
          * *应对方案：* 必须将交易频率降低，做日线级别的均值回归，而非分钟级的高频套利。

-----

### 第二阶段：研究步骤与方法论

#### 1\. 数据获取 (Data Mining)

你需要历史收盘价数据（复权后）。

  * **工具：** Python (`pandas`, `numpy`)
  * **数据源（推荐免费/开源）：**
      * **AkShare：** 目前最好用的开源财经数据接口。
      * **Tushare Pro：** 老牌接口，基础数据免费，积分制。
      * **Baostock：** 完全免费，适合下载历史数据。

#### 2\. 标的筛选 (Pair Selection)

这是最关键的一步。你不能随便拉两只股票就配对，通常遵循以下漏斗：

  * **行业同质性：** 同一个细分行业（如白酒中的“五粮液 vs 泸州老窖”，银行中的“农行 vs 建行”）。
  * **相关性测试 (Correlation)：** 计算两只股票价格序列的相关系数（Pearson Correlation），通常要求 \> 0.9。
  * **协整性检验 (Cointegration - 核心)：** 相关性高不代表价差会回归（可能一起涨但差值越来越大）。**协整**才是配对交易的灵魂。
      * *数学工具：* 使用 Engle-Granger 两步法或 Johansen Test。
      * *Python实现：* `statsmodels.tsa.stattools.coint`

#### 3\. 策略构建 (Signal Generation)

一旦确定了具有协整关系的股票对（Stock A, Stock B），就可以构建信号：

  * **构建价差 (Spread)：** $Spread = Price_A - (\beta \times Price_B)$ （$\beta$ 是对冲比率，通过线性回归计算得出）。
  * **标准化 (Z-Score)：** 将价差标准化，计算 $Z = \frac{Spread - Mean}{StdDev}$。
  * **交易规则：**
      * **开仓：** 当 Z-Score \> 2（即价差过大），做空 A，做多 B；当 Z-Score \< -2，做多 A，做空 B。
      * **平仓：** 当 Z-Score 回归到 0 附近（均值回归），平掉双边仓位。
      * **止损：** 当 Z-Score 突破 3 或 4，说明两只股票的基本面逻辑断裂（协整关系失效），必须止损。

-----

### 第三阶段：Python 实现简易框架

以下是一个基于 Python 的最小化实现逻辑，帮助你快速上手：

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# 1. 假设你已经获取了两只股票的价格序列 (Series)
stock_a = data['600036.SH'] # 招商银行
stock_b = data['601166.SH'] # 兴业银行

# 2. 协整检验 (p-value < 0.05 代表存在协整关系)
score, pvalue, _ = coint(stock_a, stock_b)
print(f"协整检验 P-value: {pvalue}")

if pvalue < 0.05:
    # 3. 计算对冲比率 (Hedge Ratio - Beta)
    # 使用 OLS 回归: Stock_A = alpha + beta * Stock_B + error
    x = sm.add_constant(stock_b)
    results = sm.OLS(stock_a, x).fit()
    beta = results.params[1]
    
    # 4. 构建价差序列
    spread = stock_a - beta * stock_b
    
    # 5. 计算 Z-Score
    # 使用滚动窗口计算均值和标准差（防止未来函数）
    window = 20
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    z_score = (spread - spread_mean) / spread_std
    
    # 6. 绘图观察交易信号
    import matplotlib.pyplot as plt
    z_score.plot()
    plt.axhline(2.0, color='red', linestyle='--')  # 开空 A / 开多 B
    plt.axhline(-2.0, color='green', linestyle='--') # 开多 A / 开空 B
    plt.axhline(0, color='black') # 平仓
    plt.show()
else:
    print("这两只股票没有协整关系，不适合配对交易。")
```

-----

### 第四阶段：进阶与避坑建议

1.  **基本面确认（Qualitative Check）：**
    不要完全依赖数据挖掘（Data Snooping）。由于A股经常受政策影响，如果一只股票停牌、重组或暴雷，协整关系会瞬间失效。**在机器筛选出 Pair 后，一定要人工确认它们的基本面逻辑是否依旧相似。**

2.  **替代策略——“AH股溢价”：**
    如果你有港股通权限，最完美的配对交易其实是 **“同一家公司的 A股 vs H股”**。

      * *逻辑：* 既然是同一家公司，同股同权，价格走势应该完全一致。
      * *机会：* A股通常比H股贵（溢价），当溢价率过高或过低时，可以进行套利。

3.  **替代策略——“ETF配对”：**
    行业ETF（如 证券ETF vs 银行ETF）比个股更稳定，非系统性风险小，且部分ETF融券更容易。

4.  **回测工具：**
    不要自己手写 `for` 循环回测。建议使用成熟框架：

      * **Backtrader：** Python 社区最流行的回测框架，功能强大。
      * **Zipline / Pyfolio：** 量化分析专用。

### 你可以做的下一步

你可以先安装 `AkShare` 库，下载“银行板块”所有股票近一年的收盘价数据。

**你想让我提供一段代码，演示如何从一个行业板块中自动筛选出“协整性最高”的那一对股票吗？**

# 如何从所有A股中自动筛选配对交易的标的

这是一个非常宏大的工程。如果你直接暴力计算所有 A 股的两两组合，会面临两个巨大的问题：

1.  **计算量爆炸：** A 股约有 5000 多只股票。两两组合的数量是 $C_{5000}^2 \approx 1,250,000,000$ (1250万) 对。对每一对都做协整检验（Cointegration Test）是非常耗时的，单机跑可能需要几天几夜。
2.  **数据窥探偏差（Data Snooping）：** 在 1250 万对组合里，纯粹因为巧合而“看起来相关”的伪回归可能高达几万对，实盘必死。

要解决这个问题，必须采用\*\*“分治法”（Divide and Conquer）**和**“降维打击”\*\*。以下是工业级的处理流程：

-----

### 方法一：行业块内筛选（最稳健、最推荐）

不要全市场乱配（比如不要拿“中石油”去配“片仔癀”）。只在**细分行业内部**进行全排列筛选。

**逻辑：** 同行业的公司受宏观政策、原材料价格、行业周期的影响是一致的，基本面逻辑天然相似。

**操作步骤：**

1.  **数据准备：** 获取所有 A 股的 **申万二级行业** 分类标签。
2.  **分组循环：**
      * 将 5000 只股票按行业分成约 100+ 个组。
      * 只在每个组内（比如“股份制银行”组，约 9 只股票）进行两两配对。
3.  **计算量骤降：** 原本是 $5000^2$，现在变成了 $\sum (行业内股票数^2)$，计算量可以降低 90% 以上。

-----

### 方法二：无监督学习聚类（最极客、机构常用）

如果你不想局限于行业分类（想寻找跨行业的隐形关联），可以使用机器学习中的**聚类算法（Clustering）**。

**核心思路：** 先用算法把“长得像”的股票自动归为一类，然后只在类内做配对。

**操作步骤：**

1.  **特征提取 (PCA)：**
      * 构建一个收益率矩阵（行是时间，列是股票）。
      * 使用 **PCA (主成分分析)** 对 5000 只股票进行降维，提取前 10-20 个主要特征因子。
2.  **聚类 (DBSCAN / OPTICS)：**
      * 使用 **DBSCAN** 算法（它不需要预设类别数量，能自动发现噪点）。
      * 将股票根据 PCA 的特征聚集成若干个簇（Cluster）。
3.  **类内检验：**
      * 只在同一个 Cluster 内部进行协整检验。

-----

### 方法三：两步过滤法（高性能 Python 实现方案）

如果你坚持要跑全市场，必须使用**向量化计算**先过滤掉 99% 的垃圾组合，再对剩下的 1% 做精细检验。

#### 步骤 1：利用矩阵运算秒算相关性 (Pearson Correlation)

`pandas` 或 `numpy` 的矩阵运算极快。计算 5000x5000 的相关系数矩阵只需要几秒钟。

#### 步骤 2：基于相关性粗筛

直接剔除相关系数 \< 0.9 的组合。剩下的组合可能只有几万对。

#### 步骤 3：多进程跑协整检验

对剩下的几万对，开启 Python 的多进程（Multiprocessing），并行计算 P-value。

-----

### 代码实战：基于“行业分块 + 多进程”的高效筛选框架

这是最适合个人电脑运行的方案。

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from multiprocessing import Pool, cpu_count
import itertools
import time

# 假设 df_price 是全市场股票收盘价矩阵 (index=date, columns=stock_code)
# 假设 df_industry 是股票行业映射表 (columns=['stock_code', 'industry'])

def check_cointegration(args):
    """
    单个配对的检测函数（用于多进程）
    """
    s1, s2, code1, code2 = args
    
    # 1. 快速相关性检查 (Numpy计算比Pandas快)
    # 如果相关性太低，直接跳过，不做耗时的协整检验
    corr = np.corrcoef(s1, s2)[0, 1]
    if corr < 0.90:
        return None
        
    # 2. 协整检验 (耗时操作)
    try:
        score, pvalue, _ = coint(s1, s2)
        if pvalue < 0.05:
            return (code1, code2, corr, pvalue)
    except:
        return None
    return None

def run_screener(df_price, df_industry):
    all_results = []
    
    # 按行业分组
    grouped = df_industry.groupby('industry')
    
    for industry_name, group in grouped:
        stocks_in_sector = group['stock_code'].tolist()
        
        # 确保这些股票在价格表中存在
        valid_stocks = [s for s in stocks_in_sector if s in df_price.columns]
        
        if len(valid_stocks) < 2:
            continue
            
        print(f"正在扫描行业: {industry_name}, 包含 {len(valid_stocks)} 只股票")
        
        # 生成该行业内的所有组合
        # itertools.combinations 生成不重复的组合
        combinations = list(itertools.combinations(valid_stocks, 2))
        
        # 准备多进程任务数据
        tasks = []
        for s1_code, s2_code in combinations:
            # 提取价格序列 (丢弃NaN以防报错)
            series1 = df_price[s1_code].dropna().values
            series2 = df_price[s2_code].dropna().values
            
            # 简单的长度对齐 (实际工程中需要更严谨的数据对齐)
            min_len = min(len(series1), len(series2))
            if min_len < 200: # 数据太短不具有统计意义
                continue
                
            tasks.append((series1[-min_len:], series2[-min_len:], s1_code, s2_code))
        
        # 开启多进程池 (利用所有 CPU 核心)
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(check_cointegration, tasks)
        
        # 过滤掉 None 并保存
        for res in results:
            if res:
                all_results.append({
                    'Industry': industry_name,
                    'Stock_A': res[0],
                    'Stock_B': res[1],
                    'Corr': res[2],
                    'P_Value': res[3]
                })

    return pd.DataFrame(all_results)

# --- 使用提示 ---
# 1. 数据必须清洗干净（复权价格）。
# 2. 这里的 df_price 建议使用 Parquet 格式读取，比 CSV 快几十倍。
# 3. 实际运行前，先拿一个行业（如'银行'）测试一下速度。
```

### 技术架构建议

如果你真的要搭建这个系统，请遵循以下架构规范：

1.  **数据存储 (Parquet / HDF5)：**
    千万不要用 CSV 读取 5000 个文件。将所有历史数据合并成一个大的 `.parquet` 文件或存入 `ClickHouse`/`DolphinDB` 数据库。读取速度是瓶颈所在。
2.  **滚动窗口 (Rolling Window)：**
    不要只算过去 3 年的一个静态 P 值。你需要计算**滚动 P 值**。
      * *比如：* 过去 12 个月协整，过去 6 个月也协整，过去 3 个月也协整。
      * 这能证明关系的稳定性，排除掉那种“刚好过去 3 年凑巧协整”的组合。
3.  **黑名单机制：**
    建立一个 `Blacklist`。如果在跑程序时遇到 ST 股、刚上市不满一年的新股、经常长期停牌的“钉子户”，直接剔除，不要浪费算力。

### 你可以做的下一步

虽然上述代码解决了“怎么算”的问题，但在实际 A 股操作中，\*\*“能不能做空”\*\*决定了你用哪种策略。

**你需要我根据“融券受限”的现实情况，为你定制一套只需做多（Long-Only）的“轮动配对策略”逻辑吗？** 这样你就不用担心借不到券的问题了。

# 用无监督学习聚类选择配对交易的标的完整方案

使用无监督学习（Unsupervised Learning）特别是\*\*聚类（Clustering）\*\*来筛选配对交易标的，是量化基金（Hedge Funds）中非常主流且高级的做法。

它的核心理念是：**不要让认为定义的“行业分类”限制了你的想象力，让数据自己告诉我们哪些股票“长得像”。**

这能帮你挖掘出跨行业的隐形关联（比如：一家做锂矿的公司和一家做电动车的公司，行业分类不同，但股价走势可能高度雷同）。

以下是全流程的深度拆解：

-----

### 第一阶段：数据降维与特征提取 (Dimensionality Reduction)

如果你直接拿 5000 只股票过去一年的日收益率去聚类，会有两个问题：

1.  **维度灾难 (Curse of Dimensionality)：** 数据太稀疏，距离计算失效。
2.  **噪音太大：** 个股特有的噪音会掩盖共性。

所以，第一步是用 **PCA（主成分分析）** 把股票的“性格”提取出来。

1.  **构建矩阵：**

      * 创建一个矩阵 $X$，行是时间（例如过去 250 个交易日），列是股票（例如 3000 只成分股）。
      * 填充值为 **日对数收益率 (Log Returns)**。
      * **标准化 (Normalization)：** 这一步至关重要。必须对每只股票做 Z-Score 标准化，让它们的均值为 0，方差为 1。因为聚类是基于“距离”的，如果不标准化，高波动股票会干扰结果。

2.  **PCA 降维：**

      * 我们不需要 250 天的所有数据，我们只需要提取主要特征。
      * 利用 `sklearn.decomposition.PCA`。
      * **保留多少个主成分？** 通常选取前 10-15 个主成分，或者保留解释方差比（Explained Variance Ratio）达到 80%-90% 的部分。
      * **输出结果：** 你会得到一个形状为 `(3000, 15)` 的矩阵。现在，每一只股票不再由 250 天的价格定义，而是由 15 个“特征因子”定义。

-----

### 第二阶段：聚类算法选择 (Clustering Algorithm)

有了这 `(3000, 15)` 的特征矩阵，我们就可以把它们丢进聚类算法里。

**为什么不能用 K-Means？**

  * K-Means 强制要求你指定分成几类（K值）。你根本不知道市场应该分几类。
  * K-Means 会强制把所有点都归类，哪怕有些股票是特立独行的“奇葩”（Outliers），这会污染配对池。

**最佳选择：DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

  * **原理：** 基于密度的聚类。它会寻找高密度的区域划分为簇。
  * **两大优势：**
    1.  **不需要指定 K 值：** 它自动发现有几类。
    2.  **自动剔除噪音：** 那些在这个 15 维空间里孤零零的股票（没有相似标的），会被标记为 `-1`（噪音），直接剔除。这正是我们要的！

-----

### 第三阶段：Python 代码实现详解

这是一个完整的核心逻辑实现 Demo：

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
import matplotlib.pyplot as plt

# 1. 数据准备
# 假设 returns_df 是一个 DataFrame: index=日期, columns=股票代码
# 数据已经清洗过，没有 NaN
returns_df = pd.read_parquet('all_stock_returns.parquet') 

# 2. 标准化 (Standardization)
# 聚类对由于单位不同导致的距离敏感，必须标准化
scaler = StandardScaler()
# 我们需要对"列"（股票）进行特征提取，所以转置一下：行=股票，列=时间特征
X = returns_df.T.values 
X_scaled = scaler.fit_transform(X)

# 3. PCA 降维 (Feature Extraction)
# 提取前 20 个主成分，或者解释度达到 90%
pca = PCA(n_components=15) 
X_pca = pca.fit_transform(X_scaled)

print(f"降维后的形状: {X_pca.shape}") 
# 输出应该是 (股票数量, 15)
# 解释：每只股票现在被压缩成了 15 个数字，代表它的核心波动特征

# 4. DBSCAN 聚类 (Clustering)
# eps: 邻域半径（越小越严格，簇越少）
# min_samples: 成为核心点所需的最小邻居数（设为2，因为我们只要找到一对就行）
clf = DBSCAN(eps=0.5, min_samples=2, metric='euclidean')
# 或者使用 OPTICS，它对参数 eps 不敏感，更稳定
# clf = OPTICS(min_samples=2, metric='euclidean')

clf.fit(X_pca)
labels = clf.labels_

# 5. 结果整理
results = pd.DataFrame(data=labels, index=returns_df.columns, columns=['Cluster_ID'])

# 剔除噪音点 (Cluster_ID == -1)
clustered_stocks = results[results['Cluster_ID'] != -1]

# 统计聚类情况
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f'发现聚类数量: {n_clusters_}')
print(f'被剔除的噪音股票数: {n_noise_}')

# 6. 在聚类内部进行协整检验 (Pair Selection)
# 只在 Cluster_ID 相同的股票之间跑协整
# (此处接之前的协整检验代码)
for cluster_id in clustered_stocks['Cluster_ID'].unique():
    group = clustered_stocks[clustered_stocks['Cluster_ID'] == cluster_id]
    stocks_in_group = group.index.tolist()
    
    if len(stocks_in_group) > 20:
        # 如果一个簇太大（比如几百只），说明聚类参数 eps 设置得太松了
        continue 
        
    print(f"簇 {cluster_id} 包含股票: {stocks_in_group}")
    # Run Cointegration Test inside this list...
```

-----

### 第四阶段：可视化与参数调优 (The Art)

为了直观地看到聚类效果，通常我们会用 **t-SNE** 或 **UMAP** 把那 15 个维度进一步压缩到 2 维，然后画出散点图。

  * **如果聚类效果好：** 你会看到图上有一些紧密的小团块。比如“招行、兴业、平安银行”会紧紧挨在一起。
  * **如果聚类效果差：** 会变成一大坨散乱的点，或者所有点都聚成了一个大球。

**参数调节的关键（难点）：**
DBSCAN 中的 `eps` (Epsilon) 参数非常敏感。

  * **eps 太大：** 所有股票都会被归为一个类（Under-segmentation）。
  * **eps 太小：** 绝大多数股票都会被当成噪音剔除（Over-segmentation）。
  * *技巧：* 可以编写一个循环，遍历 `eps` 从 0.1 到 1.0，寻找一个能产生“适量小簇”（比如每个簇 2-10 只股票）的参数值。

### 这种方法的优缺点总结

| 维度 | 优势 | 劣势 |
| :--- | :--- | :--- |
| **效率** | 极大提升。从 1000万次计算降低到几千次。 | 需要算力进行 PCA 和 聚类计算（但比协整快）。 |
| **发现力** | **极强**。能发现“隐形关联”。比如某只化工股和某只半导体股可能因为原材料关系走势高度一致。 | 解释性差。老板问你为什么这俩配对？你说“算法算的”，这很难通过风控。 |
| **稳定性** | 动态适应市场风格变化。 | **参数敏感**。市场波动率变大时，DBSCAN 的 `eps` 参数可能失效，导致选不出股票。 |

### 下一步的建议

由于聚类算法对数据的“距离”非常敏感，**数据的清洗**比模型更重要。

建议你：

1.  **先跑通流程：** 用 `沪深300` 成分股作为样本池（数量少，算得快），尝试用 PCA + DBSCAN 看能不能把“银行股”和“白酒股”自动分出来。
2.  **可视化验证：** 必须把聚类结果画出来，或者打印出每个簇的股票名称，人工看看合不合理。

**需要我提供一段代码，利用 t-SNE 算法把 300 只股票的聚类结果画成一张二维散点图吗？** 这样你能一眼看出聚类效果是否成功。