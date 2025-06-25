#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的MCTS验证脚本
用于验证MCTS基础功能是否正常工作
"""

import sys
import traceback
from board import GomokuBoard
from mcts import MCTS, MCTSNode
import math

def test_basic_functionality():
    """测试基础功能"""
    print("=== 基础功能测试 ===")
    
    try:
        # 测试1: 创建棋盘和MCTS实例
        print("1. 创建棋盘和MCTS实例...")
        board = GomokuBoard(size=9)
        mcts = MCTS(time_limit=0.5, max_iterations=100)  # 短时间测试
        print("✓ 创建成功")
        
        # 测试2: 创建MCTS节点
        print("2. 创建MCTS节点...")
        root = MCTSNode(board)
        print(f"✓ 节点创建成功，未尝试移动数: {len(root.untried_moves)}")
        
        # 测试3: UCB1计算
        print("3. 测试UCB1计算...")
        # 创建父节点
        parent = MCTSNode(board)
        parent.visits = 10
        
        # 创建子节点
        child = MCTSNode(board, parent=parent)
        child.visits = 5
        child.wins = 3
        
        ucb1_value = child.ucb1_value()
        print(f"✓ UCB1计算成功: {ucb1_value:.3f}")
        
        # 测试4: 节点扩展
        print("4. 测试节点扩展...")
        if root.untried_moves:
            move = root.untried_moves[0]
            child = root.add_child(move)
            print(f"✓ 节点扩展成功，移动: {move}")
        
        # 测试5: 简单搜索
        print("5. 测试简单搜索...")
        board.make_move(4, 4)  # 下一步棋
        best_move = mcts.search(board)
        print(f"✓ 搜索完成，推荐移动: {best_move}")
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        return False

def test_game_simulation():
    """测试完整游戏模拟"""
    print("\n=== 游戏模拟测试 ===")
    
    try:
        board = GomokuBoard(size=9)
        mcts = MCTS(time_limit=0.3, max_iterations=50)
        
        moves_count = 0
        max_moves = 10  # 限制移动数量避免长时间运行
        
        print("开始游戏模拟...")
        
        while moves_count < max_moves:
            game_over, winner = board.is_game_over()
            if game_over:
                print(f"游戏结束！获胜者: {winner}")
                break
            
            # MCTS选择移动
            move = mcts.search(board)
            if move is None:
                print("没有可用移动")
                break
            
            # 执行移动
            board.make_move(move[0], move[1])
            moves_count += 1
            
            current_player = "黑棋" if board.current_player == -1 else "白棋"  # 注意切换后的玩家
            print(f"移动 {moves_count}: {move}, 下一个玩家: {current_player}")
        
        print("\n最终棋盘状态:")
        board.display()
        
        return True
        
    except Exception as e:
        print(f"✗ 游戏模拟失败: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """测试性能"""
    print("\n=== 性能测试 ===")
    
    try:
        import time
        
        board = GomokuBoard(size=9)
        board.make_move(4, 4)  # 添加一些棋子
        board.make_move(4, 5)
        
        # 测试不同时间限制
        time_limits = [0.1, 0.5, 1.0]
        
        for time_limit in time_limits:
            mcts = MCTS(time_limit=time_limit, max_iterations=1000)
            
            start_time = time.time()
            move = mcts.search(board)
            actual_time = time.time() - start_time
            
            print(f"时间限制: {time_limit}s, 实际用时: {actual_time:.2f}s, 推荐移动: {move}")
        
        return True
        
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("MCTS算法验证测试")
    print("=" * 50)
    
    tests = [
        ("基础功能", test_basic_functionality),
        ("游戏模拟", test_game_simulation),
        ("性能测试", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n运行 {test_name} 测试...")
        if test_func():
            print(f"✓ {test_name} 测试通过")
            passed += 1
        else:
            print(f"✗ {test_name} 测试失败")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！MCTS基础实现正常工作")
        return True
    else:
        print("❌ 部分测试失败，需要检查实现")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)