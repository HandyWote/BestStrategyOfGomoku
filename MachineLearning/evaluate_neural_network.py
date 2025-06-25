#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五子棋神经网络评估脚本

功能：
- 加载训练好的神经网络模型
- 与MCTS进行对战测试
- 评估网络的策略预测准确性
- 分析网络的价值评估能力
- 生成详细的性能报告

作者: AI Assistant
日期: 2024年
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from net import GomokuNet, GomokuTrainer
from board import GomokuBoard
from mcts import MCTS
from game import Game

class NeuralNetworkPlayer:
    """神经网络玩家"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建网络
        self.net = GomokuNet(
            board_size=checkpoint['board_size'],
            num_channels=checkpoint['num_channels'],
            num_residual_blocks=checkpoint.get('num_residual_blocks', 4)
        )
        
        # 加载权重
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.to(self.device)
        self.net.eval()
        
        print(f"神经网络模型已加载: {model_path}")
        print(f"网络参数: {sum(p.numel() for p in self.net.parameters()):,}")
    
    def get_move(self, board):
        """获取神经网络推荐的移动
        
        Args:
            board: 棋盘对象
            
        Returns:
            tuple: (row, col) 推荐的移动位置
        """
        with torch.no_grad():
            # 转换棋盘状态
            board_tensor = self.net._board_to_tensor(board.board).unsqueeze(0)
            board_tensor = board_tensor.to(self.device)
            
            # 获取预测
            policy, value = self.net(board_tensor)
            
            # 转换为概率分布
            policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
            
            # 只考虑合法移动
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                return None
            
            # 找到最佳移动
            best_prob = -1
            best_move = None
            
            for row, col in valid_moves:
                move_idx = row * board.size + col
                if policy_probs[move_idx] > best_prob:
                    best_prob = policy_probs[move_idx]
                    best_move = (row, col)
            
            return best_move
    
    def evaluate_position(self, board):
        """评估当前局面
        
        Args:
            board: 棋盘对象
            
        Returns:
            tuple: (policy_probs, value) 策略概率和价值评估
        """
        with torch.no_grad():
            board_tensor = self.net._board_to_tensor(board.board).unsqueeze(0)
            board_tensor = board_tensor.to(self.device)
            
            policy, value = self.net(board_tensor)
            
            policy_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
            value_score = torch.tanh(value).cpu().numpy()[0][0]
            
            return policy_probs, value_score

class MCTSPlayer:
    """MCTS玩家"""
    
    def __init__(self, iterations=1000, exploration_weight=1.4):
        self.iterations = iterations
        self.exploration_weight = exploration_weight
    
    def get_move(self, board):
        """获取MCTS推荐的移动"""
        mcts = MCTS(time_limit=2.0, max_iterations=self.iterations)
        move = mcts.search(board)
        return move

def play_game(player1, player2, board_size=9, verbose=False):
    """进行一局游戏
    
    Args:
        player1: 玩家1
        player2: 玩家2
        board_size: 棋盘大小
        verbose: 是否打印详细信息
        
    Returns:
        int: 获胜者 (1, 2, 或 0表示平局)
    """
    board = GomokuBoard(size=board_size)
    players = [player1, player2]
    
    move_count = 0
    max_moves = board_size * board_size
    
    while not board.is_game_over() and move_count < max_moves:
        current_player = players[board.current_player - 1]
        
        try:
            move = current_player.get_move(board)
            if move is None:
                break
            
            row, col = move
            if board.is_valid_move(row, col):
                board.make_move(row, col)
                move_count += 1
                
                if verbose:
                    print(f"玩家 {board.current_player} 移动到 ({row}, {col})")
            else:
                if verbose:
                    print(f"无效移动: ({row}, {col})")
                break
                
        except Exception as e:
            if verbose:
                print(f"玩家 {board.current_player} 移动时出错: {e}")
            break
    
    winner = board.get_winner()
    return winner

def evaluate_vs_mcts(nn_player, num_games=100, mcts_iterations=500):
    """评估神经网络对战MCTS的性能
    
    Args:
        nn_player: 神经网络玩家
        num_games: 测试游戏数量
        mcts_iterations: MCTS迭代次数
        
    Returns:
        dict: 评估结果
    """
    print(f"\n=== 神经网络 vs MCTS 评估 ({num_games} 局) ===")
    print(f"MCTS 迭代次数: {mcts_iterations}")
    
    mcts_player = MCTSPlayer(iterations=mcts_iterations)
    
    results = {
        'nn_wins': 0,
        'mcts_wins': 0,
        'draws': 0,
        'nn_as_first': {'wins': 0, 'losses': 0, 'draws': 0},
        'nn_as_second': {'wins': 0, 'losses': 0, 'draws': 0},
        'game_lengths': []
    }
    
    for game_idx in tqdm(range(num_games), desc="对战测试"):
        # 交替先后手
        if game_idx % 2 == 0:
            # 神经网络先手
            winner = play_game(nn_player, mcts_player)
            if winner == 1:
                results['nn_wins'] += 1
                results['nn_as_first']['wins'] += 1
            elif winner == 2:
                results['mcts_wins'] += 1
                results['nn_as_first']['losses'] += 1
            else:
                results['draws'] += 1
                results['nn_as_first']['draws'] += 1
        else:
            # MCTS先手
            winner = play_game(mcts_player, nn_player)
            if winner == 2:
                results['nn_wins'] += 1
                results['nn_as_second']['wins'] += 1
            elif winner == 1:
                results['mcts_wins'] += 1
                results['nn_as_second']['losses'] += 1
            else:
                results['draws'] += 1
                results['nn_as_second']['draws'] += 1
    
    # 计算胜率
    total_games = results['nn_wins'] + results['mcts_wins'] + results['draws']
    nn_win_rate = results['nn_wins'] / total_games * 100
    mcts_win_rate = results['mcts_wins'] / total_games * 100
    draw_rate = results['draws'] / total_games * 100
    
    print(f"\n结果统计:")
    print(f"  神经网络胜利: {results['nn_wins']} 局 ({nn_win_rate:.1f}%)")
    print(f"  MCTS胜利: {results['mcts_wins']} 局 ({mcts_win_rate:.1f}%)")
    print(f"  平局: {results['draws']} 局 ({draw_rate:.1f}%)")
    
    print(f"\n先后手分析:")
    first_total = results['nn_as_first']['wins'] + results['nn_as_first']['losses'] + results['nn_as_first']['draws']
    second_total = results['nn_as_second']['wins'] + results['nn_as_second']['losses'] + results['nn_as_second']['draws']
    
    if first_total > 0:
        first_win_rate = results['nn_as_first']['wins'] / first_total * 100
        print(f"  神经网络先手胜率: {first_win_rate:.1f}% ({results['nn_as_first']['wins']}/{first_total})")
    
    if second_total > 0:
        second_win_rate = results['nn_as_second']['wins'] / second_total * 100
        print(f"  神经网络后手胜率: {second_win_rate:.1f}% ({results['nn_as_second']['wins']}/{second_total})")
    
    results['win_rate'] = nn_win_rate
    return results

def analyze_position_evaluation(nn_player, num_positions=100):
    """分析神经网络的局面评估能力
    
    Args:
        nn_player: 神经网络玩家
        num_positions: 测试局面数量
        
    Returns:
        dict: 分析结果
    """
    print(f"\n=== 局面评估分析 ({num_positions} 个局面) ===")
    
    evaluations = []
    
    for _ in tqdm(range(num_positions), desc="局面评估"):
        # 创建随机局面
        board = GomokuBoard(size=9)
        
        # 随机下几步棋
        num_moves = np.random.randint(5, 20)
        for _ in range(num_moves):
            valid_moves = board.get_valid_moves()
            if not valid_moves or board.is_game_over():
                break
            
            move = valid_moves[np.random.randint(len(valid_moves))]
            board.make_move(move[0], move[1])
        
        if not board.is_game_over():
            # 获取神经网络评估
            policy_probs, value = nn_player.evaluate_position(board)
            
            # 简单的局面特征
            center_control = 0
            center = board.size // 2
            for i in range(max(0, center-1), min(board.size, center+2)):
                for j in range(max(0, center-1), min(board.size, center+2)):
                    if board.board[i][j] != 0:
                        center_control += 1
            
            evaluations.append({
                'value': value,
                'center_control': center_control,
                'move_count': num_moves,
                'current_player': board.current_player
            })
    
    # 分析结果
    values = [e['value'] for e in evaluations]
    
    print(f"价值评估统计:")
    print(f"  平均值: {np.mean(values):.3f}")
    print(f"  标准差: {np.std(values):.3f}")
    print(f"  最小值: {np.min(values):.3f}")
    print(f"  最大值: {np.max(values):.3f}")
    
    return {
        'evaluations': evaluations,
        'mean_value': np.mean(values),
        'std_value': np.std(values)
    }

def generate_evaluation_report(model_path, output_dir='evaluation_results'):
    """生成完整的评估报告
    
    Args:
        model_path: 模型文件路径
        output_dir: 输出目录
    """
    print("=== 五子棋神经网络评估报告 ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载神经网络
    nn_player = NeuralNetworkPlayer(model_path)
    
    # 评估对战性能
    battle_results = evaluate_vs_mcts(nn_player, num_games=50, mcts_iterations=300)
    
    # 分析局面评估
    position_analysis = analyze_position_evaluation(nn_player, num_positions=50)
    
    # 生成报告
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("五子棋神经网络评估报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"模型路径: {model_path}\n")
        f.write(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("对战性能:\n")
        f.write(f"  总胜率: {battle_results['win_rate']:.1f}%\n")
        f.write(f"  神经网络胜利: {battle_results['nn_wins']} 局\n")
        f.write(f"  MCTS胜利: {battle_results['mcts_wins']} 局\n")
        f.write(f"  平局: {battle_results['draws']} 局\n\n")
        
        f.write("局面评估:\n")
        f.write(f"  平均价值: {position_analysis['mean_value']:.3f}\n")
        f.write(f"  价值标准差: {position_analysis['std_value']:.3f}\n")
    
    print(f"\n评估报告已保存到: {report_path}")
    
    return {
        'battle_results': battle_results,
        'position_analysis': position_analysis
    }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='五子棋神经网络评估')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='模型文件路径')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='输出目录')
    parser.add_argument('--games', type=int, default=50,
                       help='对战测试游戏数量')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在 {args.model}")
        print("请先训练模型或指定正确的模型路径")
        return
    
    try:
        # 生成评估报告
        results = generate_evaluation_report(args.model, args.output)
        
        print("\n=== 评估完成 ===")
        print(f"神经网络总体表现: {results['battle_results']['win_rate']:.1f}% 胜率")
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()