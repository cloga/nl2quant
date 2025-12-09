"""
Strategy Report Generator
==========================
Generates comprehensive backtest reports with full metric analysis.
References Common Portfolio Evaluation Metrics framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional
import json


class StrategyReportGenerator:
    """Generate comprehensive strategy evaluation reports."""

    def __init__(self):
        """Initialize report generator."""
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_html_report(
        self,
        backtest_result: Dict,
        strategy_config: Dict,
        output_path: str = None,
    ) -> str:
        """
        Generate an HTML report with full analysis.

        Args:
            backtest_result: Result dict from DCABacktestEngine
            strategy_config: Strategy configuration dict
            output_path: Path to save HTML file (optional)

        Returns:
            HTML string
        """
        metrics = backtest_result["metrics"]
        position = backtest_result.get("final_position", {})
        transactions = backtest_result.get("transactions", pd.DataFrame())

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>定投策略回测报告</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 5px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .metric-card.positive {{
            border-left-color: #4caf50;
        }}
        .metric-card.negative {{
            border-left-color: #f44336;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric-unit {{
            font-size: 12px;
            color: #999;
            margin-left: 4px;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th {{
            background: #f5f5f5;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            color: #333;
            border-bottom: 2px solid #ddd;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #fafafa;
        }}
        .analysis {{
            background: #f0f4ff;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }}
        .footer {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 定投策略回测报告</h1>
        <p>生成时间: {self.timestamp}</p>
        <p>策略类型: {strategy_config.get('strategy_type', 'Unknown')}</p>
    </div>

    <div class="metrics-grid">
        {self._generate_metric_cards(metrics)}
    </div>

    {self._generate_summary_section(metrics)}

    {self._generate_indicators_analysis(metrics)}

    {self._generate_position_section(position)}

    {self._generate_transaction_summary(transactions)}

    {self._generate_recommendations(metrics, strategy_config)}

    <div class="footer">
        <p>本报告基于历史数据生成，仅供参考。过去的表现不代表未来的结果。</p>
        <p>投资有风险，请根据个人风险承受能力谨慎决策。</p>
    </div>
</body>
</html>
        """

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

        return html

    @staticmethod
    def _generate_metric_cards(metrics: Dict) -> str:
        """Generate HTML metric cards."""
        cards = []

        metric_configs = [
            ("CAGR", "cagr_pct", "%", "positive"),
            ("年化波动率", "volatility_pct", "%", "negative"),
            ("Sharpe 比率", "sharpe_ratio", "", ""),
            ("Sortino 比率", "sortino_ratio", "", ""),
            ("Calmar 比率", "calmar_ratio", "", ""),
            ("最大回撤", "max_drawdown_pct", "%", "negative"),
            ("月度胜率", "win_rate_pct", "%", "positive"),
            ("总收益率", "total_return_pct", "%", "positive"),
        ]

        for label, key, unit, style in metric_configs:
            value = metrics.get(key, 0)
            if isinstance(value, (int, float)):
                if unit == "%":
                    display_value = f"{value:.2f}"
                else:
                    display_value = f"{value:.2f}"
            else:
                display_value = str(value)

            card_class = f"metric-card {style}" if style else "metric-card"
            cards.append(f"""
        <div class="{card_class}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display_value}<span class="metric-unit">{unit}</span></div>
        </div>
            """)

        return "\n".join(cards)

    @staticmethod
    def _generate_summary_section(metrics: Dict) -> str:
        """Generate summary analysis section."""
        total_return = metrics.get("total_return_pct", 0)
        cagr = metrics.get("cagr_pct", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        volatility = metrics.get("volatility_pct", 0)

        # Generate analysis text
        performance_assessment = "良好" if cagr > 6 else "一般" if cagr > 3 else "有待提升"
        risk_assessment = "低" if volatility < 8 else "中" if volatility < 15 else "高"
        risk_adjusted = "优秀" if sharpe > 1.5 else "良好" if sharpe > 0.8 else "一般"

        return f"""
    <div class="section">
        <h2>📊 总体评估</h2>
        <table>
            <tr>
                <th>评估维度</th>
                <th>指标值</th>
                <th>评级</th>
                <th>说明</th>
            </tr>
            <tr>
                <td>收益能力</td>
                <td>{cagr:.2f}% (CAGR)</td>
                <td>{performance_assessment}</td>
                <td>年化复合增长率，衡量长期增长能力</td>
            </tr>
            <tr>
                <td>风险水平</td>
                <td>{volatility:.2f}% (波动率)</td>
                <td>{risk_assessment}</td>
                <td>年化标准差，反映投资波动程度</td>
            </tr>
            <tr>
                <td>风险调整收益</td>
                <td>{sharpe:.2f} (Sharpe)</td>
                <td>{risk_adjusted}</td>
                <td>每单位风险的收益，越高越好</td>
            </tr>
            <tr>
                <td>抗风险能力</td>
                <td>{abs(max_dd):.2f}% (最大回撤)</td>
                <td>{'稳健' if max_dd > -15 else '需警惕'}</td>
                <td>历史最大跌幅，反映承压能力</td>
            </tr>
        </table>
        <div class="analysis">
            <strong>总体结论:</strong>
            <p>该定投策略在回测期间实现了 {total_return:.2f}% 的总收益，
            年化复合收益率为 {cagr:.2f}%。相对{risk_assessment}的风险水平，
            风险调整收益({sharpe:.2f})处于{risk_adjusted}水平。
            最大回撤{abs(max_dd):.2f}%在可控范围内。</p>
        </div>
    </div>
        """

    @staticmethod
    def _generate_indicators_analysis(metrics: Dict) -> str:
        """Generate detailed indicator analysis."""
        sortino = metrics.get("sortino_ratio", 0)
        calmar = metrics.get("calmar_ratio", 0)
        win_rate = metrics.get("win_rate_pct", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        return f"""
    <div class="section">
        <h2>📈 指标深度分析</h2>
        <table>
            <tr>
                <th>指标名称</th>
                <th>数值</th>
                <th>参考标准</th>
                <th>解读</th>
            </tr>
            <tr>
                <td><strong>Sharpe 比率</strong></td>
                <td>{sharpe:.2f}</td>
                <td>>1.0 良好<br>>2.0 优秀</td>
                <td>衡量每单位总风险所获收益。当前{'处于良好水平' if sharpe > 0.8 else '有改进空间'}</td>
            </tr>
            <tr>
                <td><strong>Sortino 比率</strong></td>
                <td>{sortino:.2f}</td>
                <td>>1.0 良好<br>>2.0 优秀</td>
                <td>仅考虑下行波动，更适合评估实际风险损失。当前{'优于Sharpe表现' if sortino > sharpe else '与Sharpe接近'}</td>
            </tr>
            <tr>
                <td><strong>Calmar 比率</strong></td>
                <td>{calmar:.2f}</td>
                <td>>2.0 稳健<br>>1.0 可接受</td>
                <td>年收益与最大回撤的比值，衡量回撤修复能力。当前{'回撤修复能力强' if calmar > 2 else '需持续观察'}</td>
            </tr>
            <tr>
                <td><strong>月度胜率</strong></td>
                <td>{win_rate:.1f}%</td>
                <td>>50% 盈利<br>>60% 良好</td>
                <td>正收益月份占比。当前{'表现稳定' if win_rate > 55 else '波动性较大'}</td>
            </tr>
        </table>
    </div>
        """

    @staticmethod
    def _generate_position_section(position: Dict) -> str:
        """Generate position analysis section."""
        if not position:
            return ""

        return f"""
    <div class="section">
        <h2>💼 期末持仓</h2>
        <table>
            <tr>
                <th>代码</th>
                <th>持仓数量</th>
                <th>当前价格</th>
                <th>持仓市值</th>
                <th>成本价</th>
                <th>浮动盈亏</th>
                <th>收益率</th>
            </tr>
            <tr>
                <td>{position.get('code', 'N/A')}</td>
                <td>{position.get('shares', 0):,.2f}</td>
                <td>¥{position.get('price', 0):.2f}</td>
                <td>¥{position.get('value', 0):,.2f}</td>
                <td>¥{position.get('cost_basis', 0) / position.get('shares', 1):.2f}</td>
                <td>¥{position.get('gain', 0):,.2f}</td>
                <td>{position.get('gain_pct', 0):.2f}%</td>
            </tr>
        </table>
    </div>
        """

    @staticmethod
    def _generate_transaction_summary(transactions: pd.DataFrame) -> str:
        """Generate transaction summary."""
        if transactions.empty:
            return ""

        total_trades = len(transactions)
        total_invested = transactions["investment"].sum()
        total_commission = transactions["commission"].sum()

        return f"""
    <div class="section">
        <h2>📝 交易统计</h2>
        <table>
            <tr>
                <th>统计项目</th>
                <th>数值</th>
            </tr>
            <tr>
                <td>总交易次数</td>
                <td>{total_trades} 次</td>
            </tr>
            <tr>
                <td>累计投入</td>
                <td>¥{total_invested:,.2f}</td>
            </tr>
            <tr>
                <td>总佣金</td>
                <td>¥{total_commission:,.2f}</td>
            </tr>
            <tr>
                <td>佣金比例</td>
                <td>{(total_commission/total_invested*100):.3f}%</td>
            </tr>
            <tr>
                <td>平均单笔投入</td>
                <td>¥{total_invested/total_trades:,.2f}</td>
            </tr>
        </table>
    </div>
        """

    @staticmethod
    def _generate_recommendations(metrics: Dict, strategy_config: Dict) -> str:
        """Generate recommendations based on metrics."""
        cagr = metrics.get("cagr_pct", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        volatility = metrics.get("volatility_pct", 0)

        recommendations = []

        if cagr < 3:
            recommendations.append("• 年化收益偏低，考虑增加权益资产比例或选择更激进的投资标的")
        elif cagr > 8:
            recommendations.append("• 历史收益表现优异，但需警惕过去的牛市环境可能不会重复")

        if sharpe < 0.5:
            recommendations.append("• Sharpe比率偏低，风险调整收益有限，考虑优化组合结构或投资频率")
        elif sharpe > 1.5:
            recommendations.append("• 风险调整收益优秀，当前配置和策略参数设置合理")

        if abs(max_dd) > 20:
            recommendations.append("• 最大回撤较大，建议确保有足够的心理承受能力和流动性储备")

        if volatility > 15:
            recommendations.append("• 年化波动率较高，可考虑增加债券/避险资产来降低整体风险")
        elif volatility < 5:
            recommendations.append("• 波动率很低，可能缺乏成长动力，考虑增加权益暴露")

        if strategy_config.get("strategy_type") == "smart_pe" or strategy_config.get("strategy_type") == "smart_pb":
            recommendations.append("• 正在使用估值驱动策略，需定期审视估值分位数的有效性")

        recommendation_html = "\n".join([f"<p>{r}</p>" for r in recommendations]) if recommendations else "<p>策略表现良好，继续执行当前计划。</p>"

        return f"""
    <div class="section">
        <h2>💡 建议与展望</h2>
        <div class="analysis">
            {recommendation_html}
        </div>
    </div>
        """

    @staticmethod
    def generate_json_summary(
        backtest_result: Dict,
        strategy_config: Dict,
    ) -> str:
        """Generate JSON summary for easy integration."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_config,
            "metrics": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in backtest_result.get("metrics", {}).items()
            },
            "final_position": backtest_result.get("final_position", {}),
            "total_transactions": len(backtest_result.get("transactions", pd.DataFrame())),
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)
