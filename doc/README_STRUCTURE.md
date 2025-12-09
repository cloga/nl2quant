# 项目目录结构

```
nl2quant/
├── app/                    # 核心应用代码
│   ├── agents/            # LangGraph agents (planner, data, quant, exec, analyst)
│   ├── config.py          # 配置管理
│   ├── state.py           # Agent 状态定义
│   └── ...
│
├── strategies/             # 回测策略脚本
│   ├── backtest_*.py      # 各种策略回测
│   ├── rebalance_*.py     # 再平衡策略
│   └── analyze_*.py       # 分析脚本
│
├── outputs/                # 所有输出结果（已加入 .gitignore）
│   ├── *.csv              # 回测数据输出
│   ├── *.txt              # 分析日志
│   └── *.md               # 总结报告
│
├── test/                   # 测试和文档
│   ├── test_cases.md      # 策略案例文档
│   └── test_*.py          # 单元测试
│
├── history/                # Streamlit 会话历史
│
├── .env                    # 环境变量（不提交）
├── .env.example            # 环境变量模板
├── requirements.txt        # Python 依赖
├── main.py                 # Streamlit 主入口
└── README.md               # 项目说明

```

## 文件组织规则

### strategies/ - 策略脚本
- `backtest_*.py` - 完整回测策略（含数据获取、执行、输出）
- `rebalance_*.py` - 再平衡策略（全天候、国债红利等）
- `analyze_*.py` - 专项分析脚本（波动率、行权细节等）

### outputs/ - 输出结果
- 所有 CSV/TXT/MD 格式的回测输出
- 已自动忽略提交到 Git（避免仓库膨胀）
- 可随时清理重新生成

### test/ - 测试与文档
- `test_cases.md` - 所有策略的设计文档、参数、回测结果
- `test_*.py` - 功能测试脚本

## 常用命令

```bash
# 运行策略回测
python strategies/backtest_dividend_dca.py
python strategies/rebalance_all_weather.py

# 启动 Streamlit 应用
streamlit run main.py

# 清理输出目录
Remove-Item outputs\* -Recurse -Force
```

## 注意事项

1. **不要在 test/ 目录放策略脚本** - 应放在 strategies/
2. **不要在根目录生成输出文件** - 应输出到 outputs/
3. **outputs/ 已被忽略** - 不会提交到 Git，本地保留即可
