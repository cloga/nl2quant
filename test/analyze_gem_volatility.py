"""
分析创业板指数历史波动，评估卖出认沽期权策略的胜率
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config

def fetch_gem_data():
    """获取创业板指数全部历史数据"""
    provider = Config.DATA_PROVIDER
    
    if provider == "akshare":
        import akshare as ak
        print("使用 AkShare 获取创业板指数数据...")
        # 创业板指数代码：399006
        df = ak.index_zh_a_hist(symbol="399006", period="daily", start_date="20100101", end_date="20251231", adjust="")
        if df.empty:
            raise ValueError("AkShare 未获取到数据")
        df = df.rename(columns={"日期": "date", "收盘": "close", "开盘": "open", "最高": "high", "最低": "low"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[["open", "high", "low", "close"]]
        df.columns = ["Open", "High", "Low", "Close"]
    else:
        import tushare as ts
        print("使用 Tushare 获取创业板指数数据...")
        if not Config.TUSHARE_TOKEN:
            raise ValueError("Tushare token 未配置")
        ts.set_token(Config.TUSHARE_TOKEN)
        pro = ts.pro_api()
        
        # 分批获取全部历史数据
        df_list = []
        start_year = 2010
        end_year = datetime.now().year
        
        for year in range(start_year, end_year + 1):
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            df_chunk = pro.index_daily(ts_code='399006.SZ', start_date=start_date, end_date=end_date)
            if not df_chunk.empty:
                df_list.append(df_chunk)
                print(f"  已获取 {year} 年数据")
        
        if not df_list:
            raise ValueError("Tushare 未获取到数据")
        
        df = pd.concat(df_list, ignore_index=True)
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date')
        df = df.set_index('trade_date')
        df = df[['open', 'high', 'low', 'close']]
        df.columns = ['Open', 'High', 'Low', 'Close']
    
    return df

def analyze_monthly_returns(df):
    """分析月度收益率分布"""
    # 重采样为月度数据（取每月最后一个交易日）
    monthly = df['Close'].resample('M').last()
    
    # 计算月度收益率
    monthly_returns = monthly.pct_change().dropna()
    
    print("\n" + "="*60)
    print("创业板指数月度收益率统计")
    print("="*60)
    print(f"数据范围: {monthly.index[0].strftime('%Y-%m')} 至 {monthly.index[-1].strftime('%Y-%m')}")
    print(f"总月数: {len(monthly_returns)}")
    print(f"\n月度收益率统计:")
    print(f"  平均收益率: {monthly_returns.mean()*100:.2f}%")
    print(f"  中位数收益率: {monthly_returns.median()*100:.2f}%")
    print(f"  标准差: {monthly_returns.std()*100:.2f}%")
    print(f"  最大月涨幅: {monthly_returns.max()*100:.2f}%")
    print(f"  最大月跌幅: {monthly_returns.min()*100:.2f}%")
    
    # 统计上涨/下跌月份
    up_months = (monthly_returns > 0).sum()
    down_months = (monthly_returns < 0).sum()
    flat_months = (monthly_returns == 0).sum()
    
    print(f"\n月度涨跌分布:")
    print(f"  上涨月份: {up_months} ({up_months/len(monthly_returns)*100:.1f}%)")
    print(f"  下跌月份: {down_months} ({down_months/len(monthly_returns)*100:.1f}%)")
    print(f"  持平月份: {flat_months}")
    
    return monthly_returns

def analyze_option_strategy(monthly_returns):
    """分析卖出认沽期权策略的胜率"""
    print("\n" + "="*60)
    print("卖出认沽期权策略模拟分析")
    print("="*60)
    
    # 不同虚值程度的统计
    otm_levels = [0.05, 0.08, 0.10, 0.12, 0.15]
    
    for otm in otm_levels:
        # 统计下跌超过虚值度的月份（被行权）
        exercised = (monthly_returns < -otm).sum()
        safe = (monthly_returns >= -otm).sum()
        
        print(f"\n虚值度 {otm*100:.0f}% (行权价 = 现价 × {1-otm:.2f}):")
        print(f"  安全月份（收权利金）: {safe} ({safe/len(monthly_returns)*100:.1f}%)")
        print(f"  被行权月份: {exercised} ({exercised/len(monthly_returns)*100:.1f}%)")
        
        # 被行权时的平均跌幅
        if exercised > 0:
            avg_loss = monthly_returns[monthly_returns < -otm].mean()
            max_loss = monthly_returns[monthly_returns < -otm].min()
            print(f"  被行权时平均跌幅: {avg_loss*100:.2f}%")
            print(f"  被行权时最大跌幅: {max_loss*100:.2f}%")
            
            # 分析连续被行权情况
            consecutive_exercise = 0
            max_consecutive_exercise = 0
            exercise_streaks = []
            exercised_months = []
            
            for idx, ret in enumerate(monthly_returns):
                if ret < -otm:
                    consecutive_exercise += 1
                    max_consecutive_exercise = max(max_consecutive_exercise, consecutive_exercise)
                    exercised_months.append(monthly_returns.index[idx].strftime('%Y-%m'))
                else:
                    if consecutive_exercise > 0:
                        exercise_streaks.append(consecutive_exercise)
                    consecutive_exercise = 0
            
            if consecutive_exercise > 0:
                exercise_streaks.append(consecutive_exercise)
            
            consecutive_2plus = sum(1 for s in exercise_streaks if s >= 2)
            consecutive_3plus = sum(1 for s in exercise_streaks if s >= 3)
            
            print(f"  连续被行权分析:")
            print(f"    最长连续行权: {max_consecutive_exercise} 个月")
            print(f"    连续行权2月以上: {consecutive_2plus} 次")
            print(f"    连续行权3月以上: {consecutive_3plus} 次")
            
            if otm == 0.10:  # 只对10%虚值输出详细被行权月份
                print(f"  被行权月份明细: {', '.join(exercised_months)}")
    
    # 分析连续下跌情况（影响移仓策略）
    print("\n" + "="*60)
    print("连续下跌分析（影响移仓决策）")
    print("="*60)
    
    consecutive_down = 0
    max_consecutive_down = 0
    down_streaks = []
    
    for ret in monthly_returns:
        if ret < 0:
            consecutive_down += 1
            max_consecutive_down = max(max_consecutive_down, consecutive_down)
        else:
            if consecutive_down > 0:
                down_streaks.append(consecutive_down)
            consecutive_down = 0
    
    if consecutive_down > 0:
        down_streaks.append(consecutive_down)
    
    print(f"最长连续下跌月数: {max_consecutive_down}")
    print(f"连续下跌2个月以上次数: {sum(1 for s in down_streaks if s >= 2)}")
    print(f"连续下跌3个月以上次数: {sum(1 for s in down_streaks if s >= 3)}")

def analyze_drawdown(df):
    """分析最大回撤"""
    cumulative = (1 + df['Close'].pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    print("\n" + "="*60)
    print("最大回撤分析")
    print("="*60)
    print(f"最大回撤: {max_dd*100:.2f}%")
    print(f"发生时间: {max_dd_date.strftime('%Y-%m-%d')}")

def main():
    print("正在获取创业板指数全部历史数据...")
    df = fetch_gem_data()
    
    print(f"\n成功获取数据: {len(df)} 个交易日")
    print(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
    
    # 月度收益率分析
    monthly_returns = analyze_monthly_returns(df)
    
    # 期权策略分析
    analyze_option_strategy(monthly_returns)
    
    # 回撤分析
    analyze_drawdown(df)
    
    # 输出详细月度数据到CSV供进一步分析
    output_file = "d:/project/nl2quant/test/gem_monthly_analysis.csv"
    monthly_df = pd.DataFrame({
        'month': monthly_returns.index.strftime('%Y-%m'),
        'return': monthly_returns.values * 100
    })
    monthly_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细月度数据已保存至: {output_file}")

if __name__ == "__main__":
    main()
