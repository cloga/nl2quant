# 🤖 NL-to-Quant Platform

**Natural Language to Quantitative Analysis & Backtesting**

NL-to-Quant is an AI-powered platform that enables users to perform financial analysis and backtesting using natural language. Built with **LangGraph**, **Streamlit**, **Tushare**, and **VectorBT**, it automates the workflow of fetching data, generating strategy code, executing backtests, and visualizing results.

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

## ✨ Features

*   **Natural Language Interface**: Describe your strategy in plain English or Chinese (e.g., "Buy when MA5 crosses MA20").
*   **Multi-Agent Architecture**: Orchestrated by LangGraph, specialized agents handle data, coding, execution, analysis、宏观解读与估值定位。
*   **Automated Data Fetching**: Integrated with **Tushare Pro** for Chinese stock market data.
*   **Fast Backtesting**: Uses **VectorBT** for high-performance vectorized backtesting.
*   **Interactive Visualization**: View equity curves and performance metrics directly in the chat interface.
*   **LLM Agnostic**: Supports OpenAI, DeepSeek, GitHub Models, and other OpenAI-compatible providers.
*   **Macro & Valuation Insights**: 宏观分析 Agent 提供结构化宏观解读，估值 Agent 给出价格分位/相对位置提示（需已有行情数据）。

## 🚀 Getting Started

### Prerequisites

*   Python 3.10 or higher
*   [Tushare Pro](https://tushare.pro/) Token (for market data)
*   LLM API Key (OpenAI, DeepSeek, etc.)

### Installation

1.  **Clone the repository** (if applicable) or navigate to the project folder:
    ```bash
    cd nl-to-quant
    ```

2.  **Create and Activate Virtual Environment**:
    It's recommended to use a virtual environment to manage dependencies.

    *   **Windows**:
        ```powershell
        python -m venv venv
        .\venv\Scripts\Activate.ps1
        ```
    *   **macOS/Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Set up environment variables**:
    Copy the example configuration file:
    ```bash
    cp .env.example .env
    # On Windows PowerShell: copy .env.example .env
    ```

2.  **Edit `.env`**:
    Open `.env` and fill in your API keys.

    *Example for DeepSeek:*
    ```ini
    LLM_PROVIDER=deepseek
    LLM_API_KEY=sk-your-deepseek-api-key
    LLM_BASE_URL=https://api.deepseek.com
    LLM_MODEL_NAME=deepseek-chat
    
    TUSHARE_TOKEN=your-tushare-token-here
    ```

    *Example for OpenAI:*
    ```ini
    LLM_PROVIDER=openai
    LLM_API_KEY=sk-your-openai-key
    LLM_MODEL_NAME=gpt-4o
    
    TUSHARE_TOKEN=your-tushare-token-here
    ```

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run main.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

### Example Prompts

*   "对 600519.SH 进行双均线回测"
*   "Backtest a simple moving average crossover strategy on 600519.SH. Buy when MA10 > MA50, sell when MA10 < MA50."
*   "Fetch data for AAPL and show me the close price." (Note: Requires US data source configuration, currently optimized for Tushare/CN stocks)
*   "给出当前市场的宏观环境解读和风险点"
*   "基于已获取的行情，评估 300750.SZ 的估值相对位置"
*   "/macro 简要点评当前宏观环境" (直接调用宏观 Agent)
*   "/valuation 评估 600519.SH 的估值相对位置" (已获取行情后可用)
*   "/data 获取 000300.SH 的行情" (直接调用数据 Agent)

## 📂 Project Structure

```text
nl-to-quant/
├── app/
│   ├── agents/             # Agent implementations (Data, Quant, Exec, Analyst, Macro, Valuation)
│   ├── config.py           # Configuration loader
│   ├── graph.py            # LangGraph workflow definition
│   ├── llm.py              # LLM factory
│   └── state.py            # Shared state definition
├── main.py                 # Streamlit entry point
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── DESIGN.md               # Technical design document
```

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. The generated trading strategies and backtest results should not be considered financial advice. Always verify code and results before making investment decisions.
