"""
数据缓存管理模块
实现 Tushare 数据的本地缓存，避免重复下载
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pickle


class DataCache:
    """Tushare 数据缓存管理器"""

    def __init__(self, cache_dir: str = None):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径，默认为 .cache/pairs_screener
        """
        if cache_dir is None:
            cache_dir = str(Path(__file__).parent.parent / ".cache" / "pairs_screener")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存元数据文件（记录每个缓存的时间戳）
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """加载缓存元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_metadata(self):
        """保存缓存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] 保存缓存元数据失败: {e}")

    def _get_cache_key(self, code: str, start_date: str, end_date: str) -> str:
        """
        生成缓存键
        
        Args:
            code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
        
        Returns:
            缓存键（MD5哈希）
        """
        key_str = f"{code}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"

    def is_cache_valid(self, code: str, start_date: str, end_date: str, max_age_hours: int = 24) -> bool:
        """
        检查缓存是否有效
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            max_age_hours: 最大缓存年龄（小时），默认24小时
        
        Returns:
            缓存是否有效
        """
        cache_key = self._get_cache_key(code, start_date, end_date)
        cache_file = self._get_cache_file(cache_key)
        
        # 检查文件是否存在
        if not cache_file.exists():
            return False
        
        # 检查文件是否过期
        if cache_key in self.metadata:
            cache_time = datetime.fromisoformat(self.metadata[cache_key]['timestamp'])
            age = datetime.now() - cache_time
            
            if age > timedelta(hours=max_age_hours):
                return False
            
            return True
        
        return False

    def get(self, code: str, start_date: str, end_date: str) -> pd.Series:
        """
        从缓存获取数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            数据Series或None
        """
        if not self.is_cache_valid(code, start_date, end_date):
            return None
        
        cache_key = self._get_cache_key(code, start_date, end_date)
        cache_file = self._get_cache_file(cache_key)
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"[CACHE] 从缓存读取: {code} ({start_date}~{end_date})")
            return data
        except Exception as e:
            print(f"[WARN] 从缓存读取失败 {code}: {e}")
            return None

    def set(self, code: str, start_date: str, end_date: str, data: pd.Series):
        """
        保存数据到缓存
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data: 数据Series
        """
        cache_key = self._get_cache_key(code, start_date, end_date)
        cache_file = self._get_cache_file(cache_key)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # 更新元数据
            self.metadata[cache_key] = {
                'code': code,
                'start_date': start_date,
                'end_date': end_date,
                'timestamp': datetime.now().isoformat(),
                'data_points': len(data) if data is not None else 0,
            }
            self._save_metadata()
            
            print(f"[CACHE] 已缓存: {code} ({start_date}~{end_date}) - {len(data)}条数据")
        except Exception as e:
            print(f"[WARN] 缓存保存失败 {code}: {e}")

    def clear_expired(self, max_age_hours: int = 24):
        """
        清除过期缓存
        
        Args:
            max_age_hours: 过期时间（小时）
        """
        now = datetime.now()
        expired_keys = []
        
        for cache_key, meta in self.metadata.items():
            try:
                cache_time = datetime.fromisoformat(meta['timestamp'])
                age = now - cache_time
                
                if age > timedelta(hours=max_age_hours):
                    expired_keys.append(cache_key)
            except Exception:
                expired_keys.append(cache_key)
        
        # 删除过期文件
        for key in expired_keys:
            cache_file = self._get_cache_file(key)
            try:
                if cache_file.exists():
                    cache_file.unlink()
                del self.metadata[key]
            except Exception as e:
                print(f"[WARN] 删除过期缓存失败: {e}")
        
        if expired_keys:
            self._save_metadata()
            print(f"[CACHE] 清理了 {len(expired_keys)} 个过期缓存文件")

    def clear_all(self):
        """清除所有缓存"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            self.metadata.clear()
            self._save_metadata()
            print(f"[CACHE] 已清除所有缓存")
        except Exception as e:
            print(f"[WARN] 清除缓存失败: {e}")

    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计信息的字典
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)  # MB
        
        return {
            'total_files': len(cache_files),
            'total_size_mb': round(total_size, 2),
            'cache_entries': len(self.metadata),
            'oldest_entry': min(
                [m['timestamp'] for m in self.metadata.values()],
                default=None
            ),
            'newest_entry': max(
                [m['timestamp'] for m in self.metadata.values()],
                default=None
            ),
        }

    def print_cache_info(self):
        """打印缓存信息"""
        stats = self.get_cache_stats()
        
        print("\n" + "="*60)
        print("数据缓存信息")
        print("="*60)
        print(f"缓存位置: {self.cache_dir}")
        print(f"缓存文件数: {stats['total_files']}")
        print(f"缓存总大小: {stats['total_size_mb']} MB")
        print(f"缓存条目: {stats['cache_entries']}")
        
        if stats['oldest_entry']:
            print(f"最老缓存: {stats['oldest_entry']}")
        if stats['newest_entry']:
            print(f"最新缓存: {stats['newest_entry']}")
        
        print("\n缓存内容:")
        if self.metadata:
            for i, (key, meta) in enumerate(list(self.metadata.items())[:10], 1):
                print(f"  {i}. {meta['code']} ({meta['start_date']}~{meta['end_date']}) "
                      f"- {meta['data_points']}条 - {meta['timestamp']}")
            
            if len(self.metadata) > 10:
                print(f"  ... 还有 {len(self.metadata) - 10} 个缓存")
        else:
            print("  (空)")
        print("="*60 + "\n")
