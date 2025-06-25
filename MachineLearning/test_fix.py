#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单验证脚本 - 测试board.py修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_board_basic():
    """测试棋盘基础功能"""
    try:
        from board import GomokuBoard
        board = GomokuBoard()
        
        # 测试is_valid_move方法
        assert board.is_valid_move(4, 4) == True
        assert board.is_valid_move(-1, 0) == False
        assert board.is_valid_move(9, 9) == False
        
        # 测试落子后的有效性检查
        board.make_move(4, 4)
        assert board.is_valid_move(4, 4) == False
        
        print("✓ board.py 修复验证成功")
        return True
        
    except Exception as e:
        print(f"✗ board.py 修复验证失败: {e}")
        return False

def test_mcts_import():
    """测试MCTS导入"""
    try:
        from mcts import MCTS, MCTSNode
        from board import GomokuBoard
        
        board = GomokuBoard()
        mcts = MCTS()
        
        print("✓ MCTS导入成功")
        return True
        
    except Exception as e:
        print(f"✗ MCTS导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=== 修复验证测试 ===")
    
    tests = [
        ("棋盘基础功能", test_board_basic),
        ("MCTS导入", test_mcts_import),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n测试 {name}...")
        if test_func():
            passed += 1
    
    print(f"\n=== 测试结果: {passed}/{total} 通过 ===")
    
    if passed == total:
        print("🎉 所有测试通过！修复成功！")
    else:
        print("❌ 部分测试失败，需要进一步修复")

if __name__ == "__main__":
    main()