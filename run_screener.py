#!/usr/bin/env python
"""
快速启动脚本 - 一键运行配对交易筛选
"""

import json
from pathlib import Path
from pairs_screener import main as cli_main
import sys

# 加载配置
CONFIG_PATH = Path(__file__).parent / "app" / "screener_config.json"

def load_config():
    """加载配置文件"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_menu():
    """打印菜单"""
    config = load_config()
    
    print("\n" + "="*60)
    print("A股配对交易标的筛选 - 快速启动")
    print("="*60)
    print("\n📋 预设方案:")
    
    for i, (name, preset) in enumerate(config['screener_presets'].items(), 1):
        print(f"\n{i}. {name}")
        print(f"   描述: {preset['description']}")
        print(f"   参数: pool={preset['pool']}, days={preset['days']}, eps={preset['eps']}")

def run_preset(preset_name):
    """运行预设方案"""
    config = load_config()
    
    if preset_name not in config['screener_presets']:
        print(f"❌ 未找到预设: {preset_name}")
        return False
    
    preset = config['screener_presets'][preset_name]
    
    print(f"\n🚀 启动预设: {preset_name}")
    print(f"📝 {preset['description']}")
    
    # 构造命令行参数
    args = [
        '--pool', preset['pool'],
        '--days', str(preset['days']),
        '--eps', str(preset['eps']),
        '--n-components', str(preset['n_components']),
        '--csv', f"pairs_{preset_name}.csv",
        '--output', f"pairs_{preset_name}.json",
    ]
    
    # 调用CLI
    sys.argv = ['pairs_screener.py'] + args
    cli_main()
    
    print(f"\n✅ 完成！结果已保存:")
    print(f"   - pairs_{preset_name}.csv")
    print(f"   - pairs_{preset_name}.json")
    
    return True

def run_custom():
    """运行自定义配置"""
    config = load_config()
    
    print("\n" + "="*60)
    print("自定义配置")
    print("="*60)
    
    # 选择股票池
    print("\n选择股票池:")
    for i, (name, pool) in enumerate(config['stock_pools'].items(), 1):
        print(f"{i}. {name} ({pool['count']}只) - {pool['description']}")
    
    pool_choice = input("\n选择 (1-3): ").strip()
    pool_names = list(config['stock_pools'].keys())
    
    if pool_choice not in ['1', '2', '3']:
        print("❌ 无效选择")
        return False
    
    selected_pool = pool_names[int(pool_choice) - 1]
    
    # 输入参数
    print(f"\n已选择: {selected_pool}")
    
    days = input("回溯天数 (默认365): ").strip() or "365"
    eps = input("DBSCAN eps (默认0.5): ").strip() or "0.5"
    n_components = input("PCA成分数 (默认15): ").strip() or "15"
    
    # 构造命令行参数
    args = [
        '--pool', selected_pool,
        '--days', days,
        '--eps', eps,
        '--n-components', n_components,
        '--csv', f"pairs_{selected_pool}_custom.csv",
        '--output', f"pairs_{selected_pool}_custom.json",
    ]
    
    print(f"\n🚀 启动自定义筛选...")
    sys.argv = ['pairs_screener.py'] + args
    cli_main()
    
    print(f"\n✅ 完成！")
    return True

def main():
    """主程序"""
    print("\n欢迎使用 A股配对交易标的筛选工具\n")
    
    print("选择运行模式:")
    print("1. 使用预设方案（推荐新手）")
    print("2. 自定义配置")
    print("3. 查看帮助")
    print("0. 退出")
    
    choice = input("\n请选择 (0-3): ").strip()
    
    if choice == '1':
        # 显示预设菜单
        print_menu()
        preset_choice = input("\n请选择预设 (1-6 或输入名称): ").strip()
        
        config = load_config()
        preset_names = list(config['screener_presets'].keys())
        
        if preset_choice.isdigit() and 1 <= int(preset_choice) <= len(preset_names):
            preset_name = preset_names[int(preset_choice) - 1]
        elif preset_choice in preset_names:
            preset_name = preset_choice
        else:
            print("❌ 无效选择")
            return
        
        run_preset(preset_name)
    
    elif choice == '2':
        run_custom()
    
    elif choice == '3':
        # 显示帮助
        print("\n" + "="*60)
        print("帮助信息")
        print("="*60)
        
        config = load_config()
        
        print("\n📚 参数说明:")
        for key, value in config['tips']['parameters'].items():
            print(f"\n{key}:")
            print(f"  {value}")
        
        print("\n📖 解释指标:")
        for key, value in config['tips']['interpretation'].items():
            print(f"\n{key}:")
            print(f"  {value}")
        
        print("\n💡 通用建议:")
        for tip in config['tips']['general']:
            print(f"• {tip}")
    
    elif choice == '0':
        print("👋 再见！")
    
    else:
        print("❌ 无效选择")

if __name__ == '__main__':
    main()
