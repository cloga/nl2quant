#!/usr/bin/env python
"""
根据三个市盈率的关系对A股进行分类
分类逻辑参考：design/pe_ration_relationship.md
"""

import pandas as pd
import tushare as ts
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
from datetime import datetime
import time

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 加载环境变量
load_dotenv()
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
if not TUSHARE_TOKEN:
    raise ValueError("请在.env文件中设置TUSHARE_TOKEN")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def get_all_a_share_codes():
    """获取所有A股代码"""
    df = pro.stock_basic(
        exchange='',
        list_status='L',
        fields='ts_code,symbol,name,area,industry,market,list_date'
    )
    print(f"获取到 {len(df)} 只A股")
    return df


def get_pe_data_batch(ts_codes):
    """
    获取市盈率数据（按日期批量获取）
    
    返回字段：
    - pe: 静态市盈率（基于最近年报净利润）
    - pe_ttm: TTM市盈率（基于最近12个月净利润）
    """
    print(f"获取最新交易日的市盈率数据...")
    
    try:
        # 不指定ts_code，获取所有股票的最新数据
        df = pro.daily_basic(
            trade_date='',  # 留空获取最新交易日
            fields='ts_code,trade_date,close,pe,pe_ttm,pb,total_mv,circ_mv'
        )
        
        if df.empty:
            print("警告: 未获取到数据，尝试获取前一交易日...")
            # 尝试获取最近的交易日数据
            import datetime
            today = datetime.datetime.now()
            for i in range(1, 10):  # 尝试最近10天
                check_date = (today - datetime.timedelta(days=i)).strftime('%Y%m%d')
                df = pro.daily_basic(
                    trade_date=check_date,
                    fields='ts_code,trade_date,close,pe,pe_ttm,pb,total_mv,circ_mv'
                )
                if not df.empty:
                    print(f"使用 {check_date} 的数据")
                    break
        
        if not df.empty:
            # 只保留指定的股票代码
            df = df[df['ts_code'].isin(ts_codes)]
            print(f"成功获取 {len(df)} 只股票的市盈率数据")
            return df
        
    except Exception as e:
        print(f"错误: 获取失败 - {e}")
    
    return pd.DataFrame()


def get_forecast_pe(ts_code, current_price, pe_static):
    """
    计算动态市盈率（模拟同花顺/雪球算法）
    
    算法：动态PE = 当前股价 ÷ 预测下一年度EPS
    
    预测EPS的方法：
    1. 优先使用业绩快报的最新EPS趋势
    2. 计算最近几期的EPS增长率
    3. 外推得到下一年度预测EPS
    4. 如果无法预测，则用历史平均增长率
    
    参数:
        ts_code: 股票代码
        current_price: 当前股价
        pe_static: 静态市盈率
    """
    try:
        # 1. 获取业绩快报数据（包含EPS）
        df_express = pro.express(ts_code=ts_code, fields='ts_code,end_date,eps')
        
        if df_express.empty or len(df_express) < 2:
            # 没有足够的业绩快报数据，用简化方法
            return estimate_dynamic_pe_simple(ts_code, current_price, pe_static)
        
        # 2. 按日期排序，取最近4期
        df_express = df_express.sort_values('end_date', ascending=False).head(4)
        df_express['eps'] = pd.to_numeric(df_express['eps'], errors='coerce')
        df_express = df_express.dropna(subset=['eps'])
        
        if len(df_express) < 2:
            return estimate_dynamic_pe_simple(ts_code, current_price, pe_static)
        
        # 3. 计算EPS增长率
        latest_eps = df_express.iloc[0]['eps']
        previous_eps = df_express.iloc[1]['eps']
        
        if previous_eps <= 0 or latest_eps <= 0:
            return None
        
        growth_rate = (latest_eps - previous_eps) / previous_eps
        
        # 4. 预测下一年度EPS（线性外推）
        forecast_eps = latest_eps * (1 + growth_rate)
        
        # 5. 计算动态PE
        if forecast_eps > 0:
            dynamic_pe = current_price / forecast_eps
            
            # 过滤异常值
            if 0 < dynamic_pe < 200:
                return dynamic_pe
        
    except Exception as e:
        pass
    
    return None


def estimate_dynamic_pe_simple(ts_code, current_price, pe_static):
    """
    简化的动态PE估算（当无业绩快报时使用）
    
    方法：基于财务指标计算历史增长率
    """
    try:
        # 获取最近4期财务指标
        df_fina = pro.fina_indicator(
            ts_code=ts_code,
            fields='ts_code,end_date,eps,roe,dt_roe'
        )
        
        if df_fina.empty or len(df_fina) < 2:
            return None
        
        df_fina = df_fina.sort_values('end_date', ascending=False).head(4)
        df_fina['eps'] = pd.to_numeric(df_fina['eps'], errors='coerce')
        df_fina = df_fina.dropna(subset=['eps'])
        
        if len(df_fina) < 2:
            return None
        
        # 计算平均增长率
        growth_rates = []
        for i in range(len(df_fina) - 1):
            current = df_fina.iloc[i]['eps']
            previous = df_fina.iloc[i + 1]['eps']
            if previous > 0 and current > 0:
                rate = (current - previous) / previous
                growth_rates.append(rate)
        
        if not growth_rates:
            return None
        
        avg_growth = sum(growth_rates) / len(growth_rates)
        
        # 获取最新EPS
        latest_eps = df_fina.iloc[0]['eps']
        
        if latest_eps > 0:
            # 预测下一年EPS
            forecast_eps = latest_eps * (1 + avg_growth)
            
            if forecast_eps > 0:
                dynamic_pe = current_price / forecast_eps
                
                if 0 < dynamic_pe < 200:
                    return dynamic_pe
        
    except Exception as e:
        pass
    
    return None


def calculate_dynamic_pe(forecast_pe):
    """
    直接使用券商预测的动态市盈率
    
    参数:
        forecast_pe: 券商预测的市盈率（已经是动态PE）
    """
    if pd.isna(forecast_pe) or forecast_pe <= 0:
        return None
    
    return forecast_pe


def classify_by_pe_relationship(row):
    """
    根据三个市盈率的关系进行分类
    
    分类逻辑：
    1. 静态 > TTM > 动态：盈利持续增长（成长股）
    2. 静态 < TTM < 动态：盈利持续下滑（衰退股）
    3. 三者接近（差值<20%）：盈利稳定（蓝筹股）
    4. 动态远低于静态/TTM（差值>50%）：盈利反转预期（困境反转）
    """
    pe_static = row['pe']
    pe_ttm = row['pe_ttm']
    pe_dynamic = row.get('pe_dynamic')
    
    # 如果市盈率数据不完整，归为"数据不足"
    if pd.isna(pe_static) or pd.isna(pe_ttm):
        return '数据不足'
    
    # 如果市盈率为负（亏损），归为"亏损"
    if pe_static <= 0 or pe_ttm <= 0:
        return '亏损'
    
    # 计算差异百分比
    diff_static_ttm = abs(pe_static - pe_ttm) / pe_static
    
    # 情况3：三者接近（稳定型）
    if pd.notna(pe_dynamic) and pe_dynamic > 0:
        diff_static_dynamic = abs(pe_static - pe_dynamic) / pe_static
        diff_ttm_dynamic = abs(pe_ttm - pe_dynamic) / pe_ttm
        
        # 三者差值都<20%
        if diff_static_ttm < 0.2 and diff_static_dynamic < 0.2 and diff_ttm_dynamic < 0.2:
            return '盈利稳定（蓝筹）'
        
        # 情况4：动态远低于静态/TTM（困境反转）
        if pe_dynamic < pe_static * 0.5 or pe_dynamic < pe_ttm * 0.5:
            return '盈利反转预期（困境反转）'
        
        # 情况1：静态 > TTM > 动态（成长股）
        if pe_static > pe_ttm > pe_dynamic:
            return '盈利持续增长（成长股）'
        
        # 情况2：静态 < TTM < 动态（衰退股）
        if pe_static < pe_ttm < pe_dynamic:
            return '盈利持续下滑（衰退股）'
    
    # 仅有静态和TTM的情况
    if pe_static > pe_ttm:
        return '近期盈利改善'
    elif pe_static < pe_ttm:
        return '近期盈利下滑'
    else:
        return '盈利平稳'


def main(limit=None, with_forecast=False):
    """
    主函数
    
    参数:
        limit: 限制处理的股票数量（用于测试）
        with_forecast: 是否获取业绩预告计算动态PE（耗时较长）
    """
    print("\n" + "="*60)
    print("A股市盈率分类分析")
    print("="*60)
    
    # 1. 获取所有A股代码
    print("\n[步骤1] 获取A股列表...")
    stock_basic = get_all_a_share_codes()
    
    if limit:
        stock_basic = stock_basic.head(limit)
        print(f"（测试模式：仅处理前{limit}只股票）")
    
    # 2. 批量获取市盈率数据
    print("\n[步骤2] 批量获取静态PE和TTM PE...")
    pe_data = get_pe_data_batch(stock_basic['ts_code'].tolist())
    
    if pe_data.empty:
        print("错误：未能获取市盈率数据")
        return
    
    # 3. 合并数据
    result = stock_basic.merge(pe_data, on='ts_code', how='left')
    
    # 4. 获取动态市盈率（可选）
    if with_forecast:
        print("\n[步骤3] 计算动态PE（基于业绩快报/财务指标预测）...")
        print("（注意：此步骤较慢，可能需要几分钟）")
        
        forecast_pes = []
        for idx, row in result.iterrows():
            if idx % 100 == 0:
                print(f"  进度: {idx}/{len(result)}")
            
            ts_code = row['ts_code']
            current_price = row['close']
            pe_static = row['pe']
            
            forecast_pe = get_forecast_pe(ts_code, current_price, pe_static)
            forecast_pes.append(forecast_pe)
            time.sleep(0.15)  # 避免频繁请求
        
        result['pe_dynamic'] = forecast_pes
    
    # 5. 分类
    print("\n[步骤4] 根据PE关系进行分类...")
    result['category'] = result.apply(classify_by_pe_relationship, axis=1)
    
    # 6. 统计分析
    print("\n" + "="*60)
    print("分类统计结果")
    print("="*60)
    category_stats = result['category'].value_counts()
    print(category_stats)
    
    # 7. 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"a_share_pe_classification_{timestamp}.csv"
    
    # 选择输出字段
    output_cols = [
        'ts_code', 'name', 'industry', 'area', 'market',
        'trade_date', 'close', 'pe', 'pe_ttm',
        'total_mv', 'circ_mv', 'category'
    ]
    
    if with_forecast and 'pe_dynamic' in result.columns:
        output_cols.insert(-1, 'pe_dynamic')
    
    result[output_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n结果已保存到: {output_file}")
    
    # 8. 展示各类别的示例
    print("\n" + "="*60)
    print("各类别示例（前5只）")
    print("="*60)
    
    for category in result['category'].unique():
        if pd.notna(category):
            print(f"\n【{category}】")
            samples = result[result['category'] == category].head(5)
            for _, row in samples.iterrows():
                pe_str = f"静态PE={row['pe']:.2f}, TTM PE={row['pe_ttm']:.2f}"
                if with_forecast and 'pe_dynamic' in row and pd.notna(row['pe_dynamic']):
                    pe_str += f", 动态PE={row['pe_dynamic']:.2f}"
                print(f"  {row['name']}({row['ts_code']}): {pe_str}")
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60 + "\n")
    
    return result


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='A股市盈率分类分析')
    parser.add_argument('--limit', type=int, help='限制处理的股票数量（测试用）')
    parser.add_argument('--with-forecast', action='store_true', 
                       help='获取业绩预告计算动态PE（耗时较长，约需10-20分钟）')
    
    args = parser.parse_args()
    
    result = main(limit=args.limit, with_forecast=args.with_forecast)
