#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五子棋游戏引擎
整合MCTS算法和神经网络，提供完整的AI对战功能
"""

import numpy as np
import time
import json
import os
from copy import deepcopy
from board import GomokuBoard
from mcts import MCTS, MCTSNode
from net import GomokuNet, GomokuTrainer
import torch

class MCTSWithNet(MCTS):
    """结合神经网络的MCTS算法"""
    
    def __init__(self, net=None, time_limit=1.0, max_iterations=1000, c_puct=1.0):
        super().__init__(time_limit, max_iterations)
        self.net = net
        self.c_puct = c_puct  # PUCT算法的探索常数
        
    def _simulate(self, node):
        """使用神经网络进行模拟（如果可用），否则使用随机模拟"""
        if self.net is not None:
            # 使用神经网络评估
            _, value = self.net.predict(node.board_state)
            # 转换为当前玩家视角的价值
            return value * node.board_state.current_player
        else:
            # 回退到随机模拟
            return super()._simulate(node)
    
    def _expand(self, node):
        """使用神经网络指导的扩展"""
        if not node.untried_moves:
            return node
            
        if self.net is not None:
            # 使用神经网络获取策略概率
            policy_probs, _ = self.net.predict(node.board_state)
            
            # 根据策略概率选择移动
            valid_moves = node.untried_moves
            move_probs = []
            
            for move in valid_moves:
                move_idx = move[0] * node.board_state.size + move[1]
                move_probs.append(policy_probs[move_idx])
            
            # 归一化概率
            move_probs = np.array(move_probs)
            if np.sum(move_probs) > 0:
                move_probs = move_probs / np.sum(move_probs)
                # 根据概率选择移动
                selected_idx = np.random.choice(len(valid_moves), p=move_probs)
                move = valid_moves[selected_idx]
            else:
                # 如果所有概率都是0，随机选择
                move = valid_moves[np.random.choice(len(valid_moves))]
        else:
            # 没有网络时随机选择
            move = node.untried_moves[np.random.choice(len(node.untried_moves))]
        
        return node.add_child(move)

class GomokuAI:
    """五子棋AI玩家"""
    
    def __init__(self, name="AI", net_path=None, mcts_time=1.0, mcts_iterations=1000):
        self.name = name
        self.net = None
        self.mcts_time = mcts_time
        self.mcts_iterations = mcts_iterations
        
        # 加载神经网络（如果提供）
        if net_path and os.path.exists(net_path):
            self.load_network(net_path)
        
        # 创建MCTS实例
        self.mcts = MCTSWithNet(
            net=self.net,
            time_limit=mcts_time,
            max_iterations=mcts_iterations
        )
    
    def load_network(self, net_path):
        """加载神经网络模型"""
        try:
            checkpoint = torch.load(net_path, map_location='cpu')
            board_size = checkpoint.get('board_size', 9)
            num_channels = checkpoint.get('num_channels', 64)
            
            self.net = GomokuNet(board_size=board_size, num_channels=num_channels)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.net.eval()
            
            print(f"{self.name} 成功加载神经网络: {net_path}")
        except Exception as e:
            print(f"{self.name} 加载神经网络失败: {e}")
            self.net = None
    
    def get_move(self, board_state):
        """获取AI的下一步移动"""
        start_time = time.time()
        
        # 使用MCTS搜索最佳移动
        move = self.mcts.search(board_state)
        
        think_time = time.time() - start_time
        print(f"{self.name} 思考时间: {think_time:.2f}秒, 选择移动: {move}")
        
        return move
    
    def set_strength(self, time_limit=None, max_iterations=None):
        """调整AI强度"""
        if time_limit is not None:
            self.mcts_time = time_limit
            self.mcts.time_limit = time_limit
        
        if max_iterations is not None:
            self.mcts_iterations = max_iterations
            self.mcts.max_iterations = max_iterations
        
        print(f"{self.name} 强度调整: 时间限制={self.mcts_time}s, 最大迭代={self.mcts_iterations}")

class GameEngine:
    """游戏引擎，管理完整的对战流程"""
    
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.reset_game()
        
    def reset_game(self):
        """重置游戏"""
        self.board = GomokuBoard(size=self.board_size)
        self.game_history = []  # 游戏历史记录
        self.start_time = time.time()
        
    def play_game(self, player1, player2, verbose=True, save_history=True):
        """执行一局完整的游戏
        
        Args:
            player1: 黑棋玩家（AI或人类）
            player2: 白棋玩家（AI或人类）
            verbose: 是否显示详细信息
            save_history: 是否保存游戏历史
            
        Returns:
            winner: 获胜者 (1=黑棋, -1=白棋, 0=平局)
            game_data: 游戏数据字典
        """
        self.reset_game()
        
        players = {1: player1, -1: player2}
        move_count = 0
        
        if verbose:
            print(f"\n=== 游戏开始 ===")
            print(f"黑棋: {player1.name if hasattr(player1, 'name') else 'Player1'}")
            print(f"白棋: {player2.name if hasattr(player2, 'name') else 'Player2'}")
            print(f"棋盘大小: {self.board_size}x{self.board_size}")
            self.board.display()
        
        while True:
            current_player = self.board.current_player
            player = players[current_player]
            
            # 检查游戏是否结束
            game_over, winner = self.board.is_game_over()
            if game_over:
                break
            
            # 获取玩家移动
            if verbose:
                player_name = player.name if hasattr(player, 'name') else f"Player{current_player}"
                print(f"\n轮到 {player_name} ({'黑棋' if current_player == 1 else '白棋'})")
            
            try:
                if hasattr(player, 'get_move'):
                    # AI玩家
                    move = player.get_move(self.board)
                else:
                    # 人类玩家或其他类型
                    move = player(self.board)  # 假设是可调用对象
                
                if move is None:
                    print("无效移动，游戏结束")
                    winner = -current_player  # 对手获胜
                    break
                
                # 执行移动
                if self.board.make_move(move[0], move[1]):
                    move_count += 1
                    
                    # 记录历史
                    if save_history:
                        self.game_history.append({
                            'move_number': move_count,
                            'player': current_player,
                            'move': move,
                            'board_state': self.board.board.copy()
                        })
                    
                    if verbose:
                        print(f"移动 {move_count}: {move}")
                        self.board.display()
                else:
                    print(f"非法移动: {move}")
                    winner = -current_player  # 对手获胜
                    break
                    
            except Exception as e:
                print(f"玩家移动出错: {e}")
                winner = -current_player  # 对手获胜
                break
        
        # 游戏结束
        game_time = time.time() - self.start_time
        
        if verbose:
            print(f"\n=== 游戏结束 ===")
            if winner == 1:
                print("黑棋获胜！")
            elif winner == -1:
                print("白棋获胜！")
            else:
                print("平局！")
            print(f"总移动数: {move_count}")
            print(f"游戏时间: {game_time:.2f}秒")
        
        # 构建游戏数据
        game_data = {
            'winner': winner,
            'move_count': move_count,
            'game_time': game_time,
            'board_size': self.board_size,
            'final_board': self.board.board.tolist(),
            'history': self.game_history if save_history else []
        }
        
        return winner, game_data
    
    def save_game(self, game_data, filename):
        """保存游戏数据到文件"""
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)
        print(f"游戏数据已保存到: {filename}")
    
    def load_game(self, filename):
        """从文件加载游戏数据"""
        with open(filename, 'r') as f:
            game_data = json.load(f)
        
        # 恢复棋盘状态
        self.board_size = game_data['board_size']
        self.board = GomokuBoard(size=self.board_size)
        self.board.board = np.array(game_data['final_board'])
        self.game_history = game_data.get('history', [])
        
        print(f"游戏数据已从 {filename} 加载")
        return game_data

class Tournament:
    """锦标赛管理器"""
    
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.results = []
    
    def run_match(self, player1, player2, num_games=10, verbose=False):
        """运行两个玩家之间的对战
        
        Args:
            player1, player2: 参赛玩家
            num_games: 对战局数
            verbose: 是否显示详细信息
            
        Returns:
            match_results: 对战结果统计
        """
        engine = GameEngine(board_size=self.board_size)
        
        wins = {1: 0, -1: 0, 0: 0}  # 黑棋胜、白棋胜、平局
        games_data = []
        
        print(f"\n=== 开始对战 ===")
        print(f"参赛者: {getattr(player1, 'name', 'Player1')} vs {getattr(player2, 'name', 'Player2')}")
        print(f"对战局数: {num_games}")
        
        for game_num in range(num_games):
            # 交替先后手
            if game_num % 2 == 0:
                black_player, white_player = player1, player2
            else:
                black_player, white_player = player2, player1
            
            print(f"\n第 {game_num + 1} 局:")
            winner, game_data = engine.play_game(
                black_player, white_player, 
                verbose=verbose, save_history=True
            )
            
            wins[winner] += 1
            games_data.append(game_data)
            
            # 显示当前战绩
            if not verbose:
                result_str = "黑胜" if winner == 1 else "白胜" if winner == -1 else "平局"
                print(f"结果: {result_str}")
        
        # 统计结果
        match_results = {
            'player1_name': getattr(player1, 'name', 'Player1'),
            'player2_name': getattr(player2, 'name', 'Player2'),
            'total_games': num_games,
            'player1_wins': wins[1] + wins[-1],  # player1作为黑棋和白棋的胜利
            'player2_wins': wins[1] + wins[-1],  # 需要重新计算
            'draws': wins[0],
            'games_data': games_data
        }
        
        # 重新计算胜负（考虑交替先后手）
        p1_wins = p2_wins = 0
        for i, game_data in enumerate(games_data):
            winner = game_data['winner']
            if winner == 0:  # 平局
                continue
            elif i % 2 == 0:  # player1执黑
                if winner == 1:
                    p1_wins += 1
                else:
                    p2_wins += 1
            else:  # player2执黑
                if winner == 1:
                    p2_wins += 1
                else:
                    p1_wins += 1
        
        match_results['player1_wins'] = p1_wins
        match_results['player2_wins'] = p2_wins
        
        # 显示最终结果
        print(f"\n=== 对战结果 ===")
        print(f"{match_results['player1_name']}: {p1_wins} 胜")
        print(f"{match_results['player2_name']}: {p2_wins} 胜")
        print(f"平局: {wins[0]} 局")
        print(f"胜率: {match_results['player1_name']} {p1_wins/num_games*100:.1f}%, {match_results['player2_name']} {p2_wins/num_games*100:.1f}%")
        
        self.results.append(match_results)
        return match_results

# 测试函数
def test_ai_vs_ai():
    """测试AI对AI的对战"""
    print("=== AI vs AI 测试 ===")
    
    try:
        # 创建两个不同强度的AI
        ai1 = GomokuAI(name="AI-Strong", mcts_time=0.5, mcts_iterations=200)
        ai2 = GomokuAI(name="AI-Weak", mcts_time=0.2, mcts_iterations=50)
        
        # 创建游戏引擎
        engine = GameEngine(board_size=9)
        
        # 进行一局对战
        winner, game_data = engine.play_game(ai1, ai2, verbose=True)
        
        print(f"\n✓ AI对战测试完成")
        print(f"获胜者: {winner}")
        print(f"移动数: {game_data['move_count']}")
        
        return True
        
    except Exception as e:
        print(f"✗ AI对战测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tournament():
    """测试锦标赛功能"""
    print("\n=== 锦标赛测试 ===")
    
    try:
        # 创建参赛AI
        ai1 = GomokuAI(name="Fast-AI", mcts_time=0.1, mcts_iterations=50)
        ai2 = GomokuAI(name="Slow-AI", mcts_time=0.2, mcts_iterations=100)
        
        # 创建锦标赛
        tournament = Tournament(board_size=9)
        
        # 运行对战
        results = tournament.run_match(ai1, ai2, num_games=4, verbose=False)
        
        print(f"\n✓ 锦标赛测试完成")
        print(f"总对战: {results['total_games']} 局")
        
        return True
        
    except Exception as e:
        print(f"✗ 锦标赛测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("五子棋游戏引擎测试")
    print("=" * 40)
    
    tests = [
        test_ai_vs_ai,
        test_tournament
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n测试结果: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("🎉 游戏引擎实现正常工作！")
    else:
        print("❌ 部分测试失败，需要检查实现")