"""PE数据缓存管理模块

功能：
1. 本地缓存 PE 数据（JSON格式）
2. 记录更新日期
3. 增量更新（避免重复计算）
4. 批量获取 daily_basic 数据
"""

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import tushare as ts
from dotenv import load_dotenv
from app.utils.rate_limiter import GLOBAL_LIMITER

from app.data.pe_ratios import PERatios, calculate_all_pe_ratios


class PECache:
    """PE数据缓存管理器"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "pe_cache.json"
        self.metadata_file = self.cache_dir / "pe_cache_metadata.json"
        
    def get_metadata(self) -> Dict:
        """获取缓存元数据（更新日期等）"""
        if not self.metadata_file.exists():
            return {
                "last_update": None,
                "total_stocks": 0,
                "cache_version": "1.0"
            }
        
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def save_metadata(self, metadata: Dict):
        """保存缓存元数据"""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def load_cache(self) -> Dict[str, Dict]:
        """加载缓存数据
        
        返回: {ts_code: {pe_data_dict}}
        """
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"加载缓存失败: {e}")
            return {}
    
    def save_cache(self, cache_data: Dict[str, Dict]):
        """保存缓存数据"""
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        # 更新元数据
        metadata = {
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_stocks": len(cache_data),
            "cache_version": "1.0"
        }
        self.save_metadata(metadata)
    
    def get_cached_pe_ratios(self, ts_code: str) -> Optional[PERatios]:
        """从缓存获取PE数据"""
        cache_data = self.load_cache()
        if ts_code not in cache_data:
            return None
        
        data = cache_data[ts_code]
        return PERatios(**data)
    
    def is_cache_fresh(self, max_age_days: int = 1) -> bool:
        """检查缓存是否新鲜
        
        参数:
            max_age_days: 缓存最大有效天数
        """
        metadata = self.get_metadata()
        if not metadata.get("last_update"):
            return False
        
        last_update = datetime.strptime(metadata["last_update"], "%Y-%m-%d %H:%M:%S")
        age_days = (datetime.now() - last_update).days
        return age_days < max_age_days


def get_daily_basic_batch(pro, trade_date: Optional[str] = None) -> pd.DataFrame:
    """批量获取所有股票的 daily_basic 数据
    
    参数:
        pro: Tushare Pro API 实例
        trade_date: 交易日期（YYYYMMDD），为空则获取最新交易日
    
    返回:
        包含 ts_code, trade_date, close, pe, pe_ttm, total_mv 等字段的 DataFrame
    """
    print("批量获取所有股票的行情数据...", flush=True)
    
    try:
        GLOBAL_LIMITER.acquire()
        df = pro.daily_basic(
            trade_date=trade_date or '',  # 留空获取最新交易日
            fields='ts_code,trade_date,close,pe,pe_ttm,total_mv,total_share'
        )
        
        if df.empty and not trade_date:
            # 尝试获取最近的交易日
            print("未获取到今日数据，尝试获取最近交易日...", flush=True)
            for i in range(1, 10):
                check_date = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y%m%d')
                GLOBAL_LIMITER.acquire()
                df = pro.daily_basic(
                    trade_date=check_date,
                    fields='ts_code,trade_date,close,pe,pe_ttm,total_mv,total_share'
                )
                if not df.empty:
                    print(f"使用 {check_date} 的数据", flush=True)
                    break
        
        print(f"批量获取完成，共 {len(df)} 只股票", flush=True)
        return df
    
    except Exception as e:
        print(f"批量获取失败: {e}", flush=True)
        return pd.DataFrame()


def batch_compute_and_cache(
    ts_codes: List[str],
    force_update: bool = False,
    use_batch_daily: bool = True,
    progress_callback=None,
    skip_same_day: bool = True
) -> Dict[str, Dict]:
    """批量计算PE并缓存
    
    参数:
        ts_codes: 股票代码列表
        force_update: 是否强制更新（忽略缓存）
        use_batch_daily: 是否使用批量获取 daily_basic 优化
        progress_callback: 进度回调函数 callback(current, total, ts_code, status)
        skip_same_day: 是否跳过同一天内的重复计算（True 时使用缓存）
    
    返回:
        {ts_code: pe_data_dict}
    """
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("未设置 TUSHARE_TOKEN")
    ts.set_token(token)
    pro = ts.pro_api()
    
    cache = PECache()
    cache_data = {} if force_update else cache.load_cache()
    
    # 获取当前日期作为时间戳标准
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 批量获取行情数据作为查找表
    daily_basic_dict = {}
    if use_batch_daily:
        df_daily = get_daily_basic_batch(pro)
        if not df_daily.empty:
            daily_basic_dict = {row['ts_code']: row for _, row in df_daily.iterrows()}
            print(f"已加载 {len(daily_basic_dict)} 只股票的行情数据到内存", flush=True)
    
    total = len(ts_codes)
    new_count = 0
    skip_count = 0
    same_day_skip_count = 0
    error_count = 0
    errors = []
    
    for idx, ts_code in enumerate(ts_codes):
        try:
            # 检查缓存
            if not force_update and ts_code in cache_data:
                cached_record = cache_data[ts_code]
                cached_date = cached_record.get("_cache_date")
                
                # 同一天且启用同天跳过模式 → 直接用缓存
                if skip_same_day and cached_date == today:
                    same_day_skip_count += 1
                    if progress_callback:
                        progress_callback(idx + 1, total, ts_code, "same_day_cached")
                    continue
                
                # 跨天但未强制更新 → 也用缓存（但下次可能需要更新）
                skip_count += 1
                if progress_callback:
                    progress_callback(idx + 1, total, ts_code, "cached")
                continue
            
            # 计算PE
            pe_ratios = calculate_all_pe_ratios(ts_code)
            pe_dict = asdict(pe_ratios)
            # 添加时间戳
            pe_dict["_cache_date"] = today
            cache_data[ts_code] = pe_dict
            new_count += 1
            
            if progress_callback:
                progress_callback(idx + 1, total, ts_code, "computed")
            else:
                print(f"进度: {idx+1}/{total}... 已完成 {ts_code}", flush=True)
            
        except Exception as e:
            error_count += 1
            errors.append((ts_code, str(e)))
            if progress_callback:
                progress_callback(idx + 1, total, ts_code, "error")
            continue
    
    # 保存缓存
    cache.save_cache(cache_data)
    
    # 构建进度摘要
    summary_parts = [f"新增 {new_count}"]
    if same_day_skip_count > 0:
        summary_parts.append(f"同天跳过 {same_day_skip_count}")
    if skip_count > 0:
        summary_parts.append(f"缓存跳过 {skip_count}")
    if error_count > 0:
        summary_parts.append(f"失败 {error_count}")
    
    print(f"\n计算完成: {', '.join(summary_parts)}", flush=True)
    
    if errors:
        error_file = cache.cache_dir / f"pe_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            for code, msg in errors:
                f.write(f"{code}: {msg}\n")
        print(f"错误明细: {error_file}", flush=True)
    
    return cache_data


def export_cache_to_csv(output_file: str):
    """将缓存数据导出为CSV"""
    cache = PECache()
    cache_data = cache.load_cache()
    
    if not cache_data:
        print("缓存为空，无数据可导出")
        return
    
    # 转为DataFrame
    rows = []
    for ts_code, data in cache_data.items():
        rows.append(data)
    
    df = pd.DataFrame(rows)
    
    # 确保列顺序
    columns_order = [
        "ts_code", "trade_date", "close_price", "market_cap",
        "static_pe", "ttm_pe", "linear_extrapolate_pe", "latest_quarter",
        "forecast_pe_mean", "forecast_pe_median"
    ]
    
    df = df[[col for col in columns_order if col in df.columns]]
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"导出完成: {output_path}，共 {len(df)} 条记录")
