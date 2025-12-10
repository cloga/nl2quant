#!/usr/bin/env python
"""
数据缓存管理工具
用于查看、清理和管理 Tushare 数据缓存
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.data_cache import DataCache
from datetime import datetime


def interactive_menu():
    """交互式菜单"""
    cache = DataCache()
    
    while True:
        print("\n" + "="*60)
        print("数据缓存管理工具")
        print("="*60)
        
        print("\n0. 退出")
        print("1. 显示缓存信息")
        print("2. 清除过期缓存 (> 24小时)")
        print("3. 清除所有缓存")
        print("4. 设置缓存过期时间")
        print("5. 查看缓存详情")
        
        choice = input("\n请选择 (0-5): ").strip()
        
        if choice == '0':
            print("\n再见！")
            break
        
        elif choice == '1':
            cache.print_cache_info()
        
        elif choice == '2':
            hours = input("输入过期时间（小时，默认24）: ").strip() or "24"
            try:
                hours = int(hours)
                cache.clear_expired(max_age_hours=hours)
                cache.print_cache_info()
            except ValueError:
                print("[ERROR] 请输入有效的数字")
        
        elif choice == '3':
            confirm = input("确认清除所有缓存? (y/n): ").strip().lower()
            if confirm == 'y':
                cache.clear_all()
                print("\n缓存已清空")
            else:
                print("已取消")
        
        elif choice == '4':
            print("\n当前缓存设置: 24小时过期")
            print("注: 过期时间是在检查缓存时使用，修改此值需要编辑代码")
        
        elif choice == '5':
            print("\n缓存位置: " + str(cache.cache_dir))
            print("\n缓存详情:")
            
            if cache.metadata:
                for i, (key, meta) in enumerate(cache.metadata.items(), 1):
                    print(f"\n{i}. {meta['code']}")
                    print(f"   日期范围: {meta['start_date']} ~ {meta['end_date']}")
                    print(f"   数据点: {meta['data_points']}")
                    print(f"   时间戳: {meta['timestamp']}")
            else:
                print("  (无缓存数据)")
        
        else:
            print("[ERROR] 无效选择")


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据缓存管理工具")
    parser.add_argument('--info', action='store_true', help='显示缓存信息并退出')
    parser.add_argument('--clear', action='store_true', help='清除所有缓存')
    parser.add_argument('--clear-expired', type=int, metavar='HOURS',
                       help='清除超过指定小时数的缓存')
    parser.add_argument('--interactive', action='store_true', help='进入交互模式')
    
    args = parser.parse_args()
    
    cache = DataCache()
    
    if args.info:
        cache.print_cache_info()
    elif args.clear:
        confirm = input("确认清除所有缓存? (y/n): ").strip().lower()
        if confirm == 'y':
            cache.clear_all()
        else:
            print("已取消")
    elif args.clear_expired:
        cache.clear_expired(max_age_hours=args.clear_expired)
        cache.print_cache_info()
    elif args.interactive or (not args.info and not args.clear and not args.clear_expired):
        interactive_menu()


if __name__ == '__main__':
    main()
