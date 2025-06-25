#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试game.py修复后的功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game import GomokuAI, GameEngine
from board import GomokuBoard

def test_ai_move():
    """测试AI移动是否正常"""
    print("=== 测试AI移动功能 ===")
    
    try:
        # 创建AI玩家
        ai1 = GomokuAI(name="AI-Test1", mcts_time=0.1)
        ai2 = GomokuAI(name="AI-Test2", mcts_time=0.1)
        
        # 创建游戏引擎
        engine = GameEngine(board_size=9)
        
        # 测试单步移动
        board = GomokuBoard(size=9)
        move = ai1.get_move(board)
        print(f"AI1移动: {move}")
        
        if move and len(move) == 2:
            print("✓ AI移动格式正确")
            board.make_move(move[0], move[1])
            print("✓ 移动执行成功")
        else:
            print("✗ AI移动格式错误")
            return False
            
        # 测试短时间对战
        print("\n=== 测试短时间AI对战 ===")
        winner, game_data = engine.play_game(ai1, ai2, verbose=False)
        
        if game_data['move_count'] > 0:
            print(f"✓ 对战完成，移动数: {game_data['move_count']}")
            print(f"✓ 获胜者: {winner}")
            return True
        else:
            print("✗ 对战异常，无移动记录")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试game.py修复...")
    
    success = test_ai_move()
    
    if success:
        print("\n🎉 所有测试通过！game.py修复成功")
    else:
        print("\n❌ 测试失败，仍有问题需要解决")