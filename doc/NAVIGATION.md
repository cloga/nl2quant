# 📍 项目导航指南

## 🚀 快速启动

```bash
streamlit run main.py
```

应用会在 `localhost:8501` 打开

---

## 📊 DCA 定投回测功能

### 📁 文件位置
```
app/pages/
├── 1_DCA_Backtest.py           # 📊 单策略详细分析
└── 2_Strategy_Comparison.py    # 🎯 多策略对比分析
```

### 🎯 如何访问
1. **启动主应用**: `streamlit run main.py`
2. **进入主界面**后，左侧导航栏选择：
   - **📊 DCA Backtest** - 单个 ETF 的详细回测
   - **🎯 Strategy Comparison** - 三个策略的对比

### 📋 功能说明

#### 页面1: DCA Backtest (454 行)
- **功能**: 单个 ETF 的 DCA 回测分析
- **策略**: 普通定投、智能 PE、智能 PB
- **指标**: 7 个核心指标（CAGR、Sharpe、Sortino 等）
- **图表**: 净值曲线、估值追踪
- **导出**: CSV 数据导出

#### 页面2: Strategy Comparison (350 行)
- **功能**: 三个策略的并行回测对比
- **对比项**: 性能表格、净值曲线、关键指标柱状图
- **分析**: 详细统计展开查看

---

## 📚 文档位置

所有 DCA 相关文档已放在 `doc/` 文件夹中：

```
doc/
├── START_HERE.md                   # ⭐ 快速开始（推荐首先阅读）
├── DCA_PAGES_GUIDE.md              # 📖 多页面应用完整指南
├── DCA_COMPLETE_GUIDE.md           # 📖 功能完整文档（含 10 个 Q&A）
├── DCA_PROJECT_SUMMARY.md          # 📋 项目架构和技术细节
├── QUICKSTART_DCA.md               # ⚡ 5 分钟快速上手
└── README_DCA.md                   # 📝 功能说明
```

---

## 📂 整体项目结构

```
nl2quant/
├── main.py                         # ⭐ 主应用入口
├── app/
│   ├── pages/                      # 多页面应用
│   │   ├── 1_DCA_Backtest.py      # DCA 回测页面
│   │   └── 2_Strategy_Comparison.py # 策略对比页面
│   ├── dca_backtest_engine.py     # 核心回测引擎
│   ├── report_generator.py        # 报告生成模块
│   ├── agents/                    # LangGraph 代理
│   ├── graph.py                   # 工作流定义
│   └── config.py                  # 配置加载
├── doc/                           # 📚 完整文档库
│   ├── START_HERE.md
│   ├── DCA_PAGES_GUIDE.md
│   ├── DCA_COMPLETE_GUIDE.md
│   ├── QUICKSTART_DCA.md
│   ├── README_DCA.md
│   └── DCA_PROJECT_SUMMARY.md
├── README.md                      # 项目概览
├── requirements.txt               # 依赖包
└── .env.example                  # 配置模板
```

---

## 💡 核心功能速览

| 功能 | 位置 | 说明 |
|------|------|------|
| **主应用** | `main.py` | NL-to-Quant 多代理框架 |
| **DCA 回测** | `app/pages/1_DCA_Backtest.py` | 单策略详细分析 |
| **策略对比** | `app/pages/2_Strategy_Comparison.py` | 三策略并行对比 |
| **回测引擎** | `app/dca_backtest_engine.py` | 核心计算引擎 |
| **报告生成** | `app/report_generator.py` | HTML/JSON 报告 |

---

## 🎯 使用流程

### 第一次使用

1. 📖 阅读 `doc/START_HERE.md`（5 分钟）
2. 🚀 运行 `streamlit run main.py`
3. 📊 选择左侧「DCA Backtest」或「Strategy Comparison」
4. ⚙️ 在侧边栏配置参数
5. ▶️ 点击「开始回测」看结果

### 深入学习

1. 📖 阅读 `doc/DCA_PAGES_GUIDE.md`（详细功能说明）
2. 📖 阅读 `doc/DCA_COMPLETE_GUIDE.md`（10 个常见问题）
3. 🔧 调整参数进行自己的分析

---

## 🆚 三个投资策略

| 策略 | 特点 | 适用场景 |
|------|------|--------|
| **普通定投** | 每期固定金额 | 稳定资金流，无法择时 |
| **智能PE定投** | 根据 PE 百分位调整金额 | 相信均值回归 |
| **智能PB定投** | 根据 PB 百分位调整金额 | 关注资产质量 |

---

## ⚡ 快速参考

### 常见问题

**Q: 页面在哪里找？**
A: 启动后左侧导航栏，选择「DCA Backtest」或「Strategy Comparison」

**Q: 怎么导出数据？**
A: 回测完成后，在结果下方点击「下载」按钮（支持 CSV 格式）

**Q: 需要哪些参数？**
A: ETF 代码、投资金额、投资频率、时间范围。智能策略还需配置低估/高估倍数

**Q: 哪个指标最重要？**
A: 优先看 Sharpe 比率（风险调整后的收益）而非单纯总收益

---

## 📞 技术支持

- 📖 完整文档：查看 `doc/` 文件夹
- 🐛 问题排查：检查 `.env` 配置和 Tushare 连接
- 💬 详细说明：见 `doc/DCA_COMPLETE_GUIDE.md`

---

**最后更新**: 2025-12-09
