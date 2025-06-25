#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 测试MCTS基础功能
"""

import sys
import time
import numpy as np

# 简化的测试函数
def test_imports():
    """测试导入"""
    try:
        from board import GomokuBoard
        from mcts import MCTS, MCTSNode
        print("✓ 导入测试通过")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_board_basic():
    """测试棋盘基础功能"""
    try:
        from board import GomokuBoard
        
        board = GomokuBoard(size=9)
        
        # 测试落子
        assert board.make_move(4, 4) == True
        assert board.board[4, 4] == 1
        
        # 测试无效移动
        assert board.make_move(4, 4) == False
        
        # 测试获取有效移动
        valid_moves = board.get_valid_moves()
        assert len(valid_moves) == 80  # 9*9 - 1
        
        print("✓ 棋盘基础功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 棋盘测试失败: {e}")
        return False

def test_mcts_node():
    """测试MCTS节点"""
    try:
        from board import GomokuBoard
        from mcts import MCTSNode
        
        board = GomokuBoard(size=9)
        node = MCTSNode(board)
        
        # 测试节点创建
        assert node.visits == 0
        assert node.wins == 0.0
        assert len(node.untried_moves) > 0
        
        # 测试UCB1计算
        ucb1 = node.ucb1_value()
        assert ucb1 == float('inf')  # 未访问节点应该返回无穷大
        
        # 测试更新
        node.update(1.0)
        assert node.visits == 1
        assert node.wins == 1.0
        
        print("✓ MCTS节点测试通过")
        return True
        
    except Exception as e:
        print(f"✗ MCTS节点测试失败: {e}")
        return False

def test_mcts_search():
    """测试MCTS搜索"""
    try:
        from board import GomokuBoard
        from mcts import MCTS
        
        board = GomokuBoard(size=9)
        mcts = MCTS(time_limit=0.1, max_iterations=50)  # 快速测试
        
        start_time = time.time()
        move = mcts.search(board)
        search_time = time.time() - start_time
        
        # 验证结果
        assert move is not None
        assert len(move) == 2
        assert 0 <= move[0] < 9
        assert 0 <= move[1] < 9
        assert board.is_valid_move(move[0], move[1])
        
        print(f"✓ MCTS搜索测试通过 (耗时: {search_time:.3f}秒, 移动: {move})")
        return True
        
    except Exception as e:
        print(f"✗ MCTS搜索测试失败: {e}")
        return False

def test_performance():
    """测试性能"""
    try:
        from board import GomokuBoard
        from mcts import MCTS
        
        board = GomokuBoard(size=9)
        
        # 测试不同时间限制
        time_limits = [0.5, 1.0, 2.0]
        
        for time_limit in time_limits:
            mcts = MCTS(time_limit=time_limit, max_iterations=1000)
            
            start_time = time.time()
            move = mcts.search(board)
            actual_time = time.time() - start_time
            
            # 验证时间控制（允许一定误差）
            assert actual_time <= time_limit + 0.5  # 允许0.5秒误差
            
            print(f"  时间限制 {time_limit}s: 实际耗时 {actual_time:.3f}s, 移动 {move}")
        
        print("✓ 性能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("五子棋MCTS快速验证")
    print("=" * 40)
    
    tests = [
        ("导入测试", test_imports),
        ("棋盘基础功能", test_board_basic),
        ("MCTS节点", test_mcts_node),
        ("MCTS搜索", test_mcts_search),
        ("性能测试", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n运行 {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"  {test_name} 失败")
        except Exception as e:
            print(f"  {test_name} 异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！MCTS基础功能正常工作")
        print("\n✅ 第1天目标完成:")
        print("  - ✓ 基础MCTS实现 (纯随机模拟)")
        print("  - ✓ 搜索功能正常 (选择、扩展、模拟、反向传播)")
        print("  - ✓ UCB1公式工作正常")
        print("  - ✓ 性能控制在1-3秒内")
    else:
        print("❌ 部分测试失败，需要检查实现")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)