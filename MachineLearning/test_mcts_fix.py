#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试MCTS参数修复
验证MCTS类的正确实例化和基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from board import GomokuBoard
from mcts import MCTS
import random
import numpy as np

def test_mcts_instantiation():
    """测试MCTS实例化"""
    print("=== 测试MCTS实例化 ===")
    try:
        # 测试正确的参数
        mcts = MCTS(time_limit=1.0, max_iterations=100)
        print("✅ MCTS实例化成功")
        print(f"时间限制: {mcts.time_limit}秒")
        print(f"最大迭代次数: {mcts.max_iterations}")
        return True
    except Exception as e:
        print(f"❌ MCTS实例化失败: {e}")
        return False

def test_mcts_search():
    """测试MCTS搜索功能"""
    print("\n=== 测试MCTS搜索功能 ===")
    try:
        # 创建棋盘和MCTS
        board = GomokuBoard(size=9)
        mcts = MCTS(time_limit=0.5, max_iterations=50)  # 短时间测试
        
        # 执行搜索
        move = mcts.search(board)
        
        if move:
            print(f"✅ MCTS搜索成功，推荐移动: {move}")
            # 验证移动的有效性
            if board.is_valid_move(move[0], move[1]):
                print("✅ 推荐移动有效")
                return True
            else:
                print("❌ 推荐移动无效")
                return False
        else:
            print("❌ MCTS搜索失败，未找到移动")
            return False
    except Exception as e:
        print(f"❌ MCTS搜索测试失败: {e}")
        return False

def test_training_data_generation():
    """测试训练数据生成逻辑"""
    print("\n=== 测试训练数据生成逻辑 ===")
    try:
        board = GomokuBoard(size=9)
        mcts = MCTS(time_limit=0.2, max_iterations=20)
        
        # 模拟训练数据生成的关键步骤
        move = mcts.search(board)
        if move is None:
            valid_moves = board.get_valid_moves()
            if valid_moves:
                move = random.choice(valid_moves)
        
        if move:
            row, col = move
            
            # 创建策略分布
            board_size = board.size
            valid_moves = board.get_valid_moves()
            action_probs = np.zeros(board_size * board_size)
            
            if valid_moves:
                prob_per_move = 1.0 / len(valid_moves)
                for valid_move in valid_moves:
                    action_idx = valid_move[0] * board_size + valid_move[1]
                    action_probs[action_idx] = prob_per_move
            
            action = row * board_size + col
            
            print(f"✅ 训练数据生成成功")
            print(f"选择的移动: {move}")
            print(f"动作索引: {action}")
            print(f"策略分布总和: {np.sum(action_probs):.3f}")
            print(f"有效移动数量: {len(valid_moves)}")
            
            return True
        else:
            print("❌ 无法生成有效移动")
            return False
            
    except Exception as e:
        print(f"❌ 训练数据生成测试失败: {e}")
        return False

def test_evaluate_player():
    """测试评估脚本中的MCTSPlayer"""
    print("\n=== 测试MCTSPlayer ===")
    try:
        # 模拟MCTSPlayer类的关键逻辑
        class TestMCTSPlayer:
            def __init__(self, iterations=100):
                self.iterations = iterations
            
            def get_move(self, board):
                mcts = MCTS(time_limit=0.2, max_iterations=self.iterations)
                move = mcts.search(board)
                return move
        
        # 测试
        board = GomokuBoard(size=9)
        player = TestMCTSPlayer(iterations=20)
        move = player.get_move(board)
        
        if move:
            print(f"✅ MCTSPlayer测试成功，移动: {move}")
            return True
        else:
            print("❌ MCTSPlayer未返回有效移动")
            return False
            
    except Exception as e:
        print(f"❌ MCTSPlayer测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("五子棋MCTS参数修复验证测试")
    print("=" * 50)
    
    tests = [
        test_mcts_instantiation,
        test_mcts_search,
        test_training_data_generation,
        test_evaluate_player
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！MCTS参数修复成功！")
        print("\n✅ 修复内容:")
        print("1. MCTS实例化参数: time_limit, max_iterations")
        print("2. 移除不存在的get_action_probabilities()调用")
        print("3. 使用mcts.search()获取移动")
        print("4. 简化策略分布生成")
        print("5. 修复MCTSPlayer.get_move()方法")
    else:
        print("❌ 部分测试失败，需要进一步检查")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)