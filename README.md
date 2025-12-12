# 🤖 NL-to-Quant Platform

**Natural Language to Quantitative Analysis & Backtesting**

NL-to-Quant is an AI-powered platform that enables users to perform financial analysis and backtesting using natural language. Built with **LangGraph**, **Streamlit**, **Tushare**, and **VectorBT**, it automates the workflow of fetching data, generating strategy code, executing backtests, and visualizing results.

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

## ✨ Features

*   **Natural Language Interface**: Describe your strategy in plain English or Chinese (e.g., "Buy when MA5 crosses MA20").
*   **Multi-Agent Architecture**: Orchestrated by LangGraph, specialized agents handle data, coding, execution, and analysis.
*   **Automated Data Fetching**: Integrated with **Tushare Pro** for Chinese stock market data.
*   **Fast Backtesting**: Uses **VectorBT** for high-performance vectorized backtesting.
*   **Interactive Visualization**: View equity curves and performance metrics directly in the chat interface.
*   **LLM Agnostic**: Supports OpenAI, DeepSeek, GitHub Models, and other OpenAI-compatible providers.

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

2.  **Create Virtual Environment (Required)**:
    This project is intended to be run **only** inside a local virtual environment named `.venv`.

    *   **Windows (PowerShell)**:
        ```powershell
        python -m venv .venv
        Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
        .\.venv\Scripts\Activate.ps1
        ```

3.  **Install dependencies (inside `.venv`)**:
    ```powershell
    python -m pip install --upgrade pip
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

```powershell
# Recommended: always run using the venv interpreter
.\.venv\Scripts\python.exe -m streamlit run main.py
```

The application will open in your default web browser (usually at `http://localhost:8501`).

### Notes (Windows)

- If PowerShell blocks activation scripts, keep using:
    `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force`
- You can also avoid activation entirely by always using `.\.venv\Scripts\python.exe ...`.

### Example Prompts

*   "对 600519.SH 进行双均线回测"
*   "Backtest a simple moving average crossover strategy on 600519.SH. Buy when MA10 > MA50, sell when MA10 < MA50."
*   "Fetch data for AAPL and show me the close price." (Note: Requires US data source configuration, currently optimized for Tushare/CN stocks)

### Index Valuation (AKShare)

To fetch index PE/PB/dividend yield history with current percentiles via AKShare:

```python
from app.index_api import get_index_valuation

result = get_index_valuation(
    ts_code="000300.SH",
    name="沪深300",
    years=10,
    data_source="akshare",  # omit to use DATA_SOURCE from .env
)

print(result["pe"], result["pb"], result["dividend_yield"])
# result["history"] contains the full time series
```

Set `DATA_SOURCE=akshare` in `.env` to default all index fetches to AKShare. Keep `tushare` for the previous behavior.

## 📂 Project Structure

```text
nl-to-quant/
├── app/
│   ├── agents/             # Agent implementations (Data, Quant, Exec, Analyst)
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
