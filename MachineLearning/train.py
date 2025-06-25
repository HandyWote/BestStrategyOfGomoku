#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五子棋AI训练框架
实现自我对弈训练和模型优化
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
import time
from datetime import datetime
from collections import deque
import random
from copy import deepcopy

from board import GomokuBoard
from mcts import MCTS
from net import GomokuNet, GomokuTrainer, GomokuDataset
from game import GomokuAI, GameEngine, MCTSWithNet

class SelfPlayDataset(Dataset):
    """自我对弈数据集"""
    
    def __init__(self, max_size=10000):
        self.data = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_game_data(self, game_states, game_policies, game_values):
        """添加一局游戏的数据"""
        for state, policy, value in zip(game_states, game_policies, game_values):
            self.data.append((state, policy, value))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, policy, value = self.data[idx]
        return {
            'board_state': torch.FloatTensor(state),
            'policy': torch.FloatTensor(policy),
            'value': torch.FloatTensor([value])
        }
    
    def clear(self):
        """清空数据"""
        self.data.clear()
    
    def save_to_file(self, filename):
        """保存数据到文件"""
        data_list = list(self.data)
        torch.save(data_list, filename)
        print(f"训练数据已保存到: {filename}")
    
    def load_from_file(self, filename):
        """从文件加载数据"""
        if os.path.exists(filename):
            data_list = torch.load(filename)
            self.data.extend(data_list)
            print(f"从 {filename} 加载了 {len(data_list)} 条训练数据")
        else:
            print(f"文件 {filename} 不存在")

class MCTSTrainer:
    """MCTS训练器，用于生成训练数据"""
    
    def __init__(self, net, board_size=9, mcts_simulations=100, temperature=1.0):
        self.net = net
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        
    def generate_move_probabilities(self, board_state, mcts_tree=None):
        """使用MCTS生成移动概率分布
        
        Args:
            board_state: 当前棋盘状态
            mcts_tree: 可选的MCTS树（用于复用）
            
        Returns:
            move_probs: 移动概率分布
            mcts_tree: 更新后的MCTS树
        """
        # 创建MCTS实例
        if mcts_tree is None:
            mcts = MCTSWithNet(
                net=self.net,
                time_limit=None,  # 使用迭代次数限制
                max_iterations=self.mcts_simulations
            )
        else:
            mcts = mcts_tree
        
        # 进行MCTS搜索
        root = mcts._create_root_node(board_state)
        
        for _ in range(self.mcts_simulations):
            # 选择
            node = mcts._select(root)
            
            # 扩展
            if not node.is_terminal():
                node = mcts._expand(node)
            
            # 模拟
            value = mcts._simulate(node)
            
            # 反向传播
            mcts._backpropagate(node, value)
        
        # 计算移动概率
        move_probs = np.zeros(board_state.size * board_state.size)
        
        if root.children:
            visits = np.array([child.visits for child in root.children.values()])
            
            if self.temperature == 0:
                # 贪婪选择
                best_moves = []
                max_visits = np.max(visits)
                for move, child in root.children.items():
                    if child.visits == max_visits:
                        best_moves.append(move)
                
                # 在最佳移动中均匀分布
                for move in best_moves:
                    move_idx = move[0] * board_state.size + move[1]
                    move_probs[move_idx] = 1.0 / len(best_moves)
            else:
                # 温度采样
                if self.temperature != 1.0:
                    visits = visits ** (1.0 / self.temperature)
                
                visits_sum = np.sum(visits)
                if visits_sum > 0:
                    for i, (move, child) in enumerate(root.children.items()):
                        move_idx = move[0] * board_state.size + move[1]
                        move_probs[move_idx] = visits[i] / visits_sum
        
        return move_probs, mcts
    
    def play_self_game(self, verbose=False):
        """进行一局自我对弈
        
        Returns:
            game_states: 游戏状态列表
            game_policies: 策略概率列表
            game_values: 价值评估列表
        """
        board = GomokuBoard(size=self.board_size)
        game_states = []
        game_policies = []
        
        mcts_tree = None
        move_count = 0
        
        if verbose:
            print("开始自我对弈...")
            board.display()
        
        while True:
            # 检查游戏是否结束
            game_over, winner = board.is_game_over()
            if game_over:
                break
            
            # 记录当前状态
            current_state = self._board_to_input(board)
            game_states.append(current_state)
            
            # 生成移动概率
            move_probs, mcts_tree = self.generate_move_probabilities(board, mcts_tree)
            game_policies.append(move_probs)
            
            # 根据概率选择移动
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break
            
            # 计算有效移动的概率
            valid_probs = []
            for move in valid_moves:
                move_idx = move[0] * board.size + move[1]
                valid_probs.append(move_probs[move_idx])
            
            valid_probs = np.array(valid_probs)
            if np.sum(valid_probs) > 0:
                valid_probs = valid_probs / np.sum(valid_probs)
                selected_idx = np.random.choice(len(valid_moves), p=valid_probs)
            else:
                selected_idx = np.random.choice(len(valid_moves))
            
            selected_move = valid_moves[selected_idx]
            
            # 执行移动
            board.make_move(selected_move[0], selected_move[1])
            move_count += 1
            
            if verbose and move_count <= 10:  # 只显示前10步
                print(f"移动 {move_count}: {selected_move}")
                board.display()
            
            # 重置MCTS树（简化实现）
            mcts_tree = None
        
        # 计算每个状态的价值（从游戏结果反推）
        game_values = []
        for i, state in enumerate(game_states):
            # 当前玩家视角的价值
            current_player = 1 if i % 2 == 0 else -1
            if winner == current_player:
                value = 1.0
            elif winner == -current_player:
                value = -1.0
            else:
                value = 0.0
            game_values.append(value)
        
        if verbose:
            result_str = "黑胜" if winner == 1 else "白胜" if winner == -1 else "平局"
            print(f"自我对弈结束: {result_str}, 总步数: {move_count}")
        
        return game_states, game_policies, game_values
    
    def _board_to_input(self, board):
        """将棋盘转换为神经网络输入格式"""
        # 创建3通道输入：当前玩家棋子、对手棋子、当前玩家标识
        input_planes = np.zeros((3, board.size, board.size))
        
        # 当前玩家的棋子
        input_planes[0] = (board.board == board.current_player).astype(np.float32)
        
        # 对手的棋子
        input_planes[1] = (board.board == -board.current_player).astype(np.float32)
        
        # 当前玩家标识（全1表示当前玩家回合）
        input_planes[2] = np.ones((board.size, board.size)) * (board.current_player == 1)
        
        return input_planes

class AlphaZeroTrainer:
    """AlphaZero风格的训练器"""
    
    def __init__(self, board_size=9, num_channels=64, learning_rate=0.001):
        self.board_size = board_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建神经网络
        self.net = GomokuNet(board_size=board_size, num_channels=num_channels)
        self.net.to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 创建数据集
        self.dataset = SelfPlayDataset(max_size=50000)
        
        # 创建MCTS训练器
        self.mcts_trainer = MCTSTrainer(
            net=self.net,
            board_size=board_size,
            mcts_simulations=100,
            temperature=1.0
        )
        
        # 训练统计
        self.training_stats = {
            'iterations': 0,
            'self_play_games': 0,
            'training_losses': [],
            'policy_losses': [],
            'value_losses': []
        }
    
    def self_play_iteration(self, num_games=10, verbose=False):
        """进行一轮自我对弈"""
        print(f"开始自我对弈: {num_games} 局游戏")
        
        self.net.eval()  # 设置为评估模式
        
        games_data = []
        for game_num in range(num_games):
            if verbose or (game_num + 1) % max(1, num_games // 5) == 0:
                print(f"自我对弈进度: {game_num + 1}/{num_games}")
            
            # 进行一局自我对弈
            states, policies, values = self.mcts_trainer.play_self_game(verbose=False)
            
            if states:  # 确保游戏有效
                games_data.append((states, policies, values))
                self.dataset.add_game_data(states, policies, values)
        
        self.training_stats['self_play_games'] += len(games_data)
        print(f"自我对弈完成: 生成 {len(games_data)} 局有效游戏")
        print(f"数据集大小: {len(self.dataset)}")
        
        return len(games_data)
    
    def train_network(self, epochs=10, batch_size=32, verbose=True):
        """训练神经网络"""
        if len(self.dataset) < batch_size:
            print(f"数据不足，需要至少 {batch_size} 条数据，当前只有 {len(self.dataset)} 条")
            return
        
        print(f"开始训练网络: {epochs} 轮, 批大小: {batch_size}")
        
        self.net.train()  # 设置为训练模式
        
        # 创建数据加载器
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                board_states = batch['board_state'].to(self.device)
                target_policies = batch['policy'].to(self.device)
                target_values = batch['value'].to(self.device)
                
                # 前向传播
                pred_policies, pred_values = self.net(board_states)
                
                # 计算损失
                policy_loss = nn.CrossEntropyLoss()(pred_policies, target_policies)
                value_loss = nn.MSELoss()(pred_values.squeeze(), target_values.squeeze())
                total_loss_batch = policy_loss + value_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
                # 统计
                total_loss += total_loss_batch.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            avg_policy_loss = total_policy_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            
            epoch_losses.append(avg_loss)
            epoch_policy_losses.append(avg_policy_loss)
            epoch_value_losses.append(avg_value_loss)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Loss={avg_loss:.4f}, "
                      f"Policy={avg_policy_loss:.4f}, "
                      f"Value={avg_value_loss:.4f}")
        
        # 更新统计
        self.training_stats['training_losses'].extend(epoch_losses)
        self.training_stats['policy_losses'].extend(epoch_policy_losses)
        self.training_stats['value_losses'].extend(epoch_value_losses)
        
        print(f"网络训练完成")
        return epoch_losses
    
    def training_iteration(self, self_play_games=10, training_epochs=10, batch_size=32):
        """完整的训练迭代"""
        print(f"\n=== 训练迭代 {self.training_stats['iterations'] + 1} ===")
        
        # 自我对弈
        games_generated = self.self_play_iteration(num_games=self_play_games)
        
        if games_generated > 0:
            # 训练网络
            losses = self.train_network(epochs=training_epochs, batch_size=batch_size)
            
            self.training_stats['iterations'] += 1
            
            print(f"训练迭代完成")
            return True
        else:
            print(f"没有生成有效游戏数据，跳过训练")
            return False
    
    def save_model(self, filepath):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'board_size': self.board_size,
            'num_channels': self.net.num_channels if hasattr(self.net, 'num_channels') else 64
        }
        
        torch.save(checkpoint, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            
            print(f"模型已从 {filepath} 加载")
            print(f"训练迭代: {self.training_stats['iterations']}")
            print(f"自我对弈游戏: {self.training_stats['self_play_games']}")
            return True
        else:
            print(f"模型文件 {filepath} 不存在")
            return False
    
    def evaluate_model(self, opponent_model_path=None, num_games=10):
        """评估模型性能"""
        print(f"\n=== 模型评估 ===")
        
        # 创建当前模型的AI
        current_ai = GomokuAI(name="Current-Model", mcts_time=0.5)
        current_ai.net = self.net
        current_ai.mcts = MCTSWithNet(net=self.net, time_limit=0.5)
        
        # 创建对手AI
        if opponent_model_path and os.path.exists(opponent_model_path):
            opponent_ai = GomokuAI(name="Opponent-Model", net_path=opponent_model_path, mcts_time=0.5)
        else:
            # 使用纯MCTS作为对手
            opponent_ai = GomokuAI(name="Pure-MCTS", mcts_time=0.5)
        
        # 进行对战
        engine = GameEngine(board_size=self.board_size)
        wins = 0
        
        for game_num in range(num_games):
            if game_num % 2 == 0:
                winner, _ = engine.play_game(current_ai, opponent_ai, verbose=False)
                if winner == 1:  # 当前模型执黑获胜
                    wins += 1
            else:
                winner, _ = engine.play_game(opponent_ai, current_ai, verbose=False)
                if winner == -1:  # 当前模型执白获胜
                    wins += 1
        
        win_rate = wins / num_games
        print(f"评估结果: {wins}/{num_games} 胜, 胜率: {win_rate:.2%}")
        
        return win_rate

# 测试函数
def test_self_play():
    """测试自我对弈功能"""
    print("=== 自我对弈测试 ===")
    
    try:
        # 创建简单的网络
        net = GomokuNet(board_size=9, num_channels=32)
        
        # 创建MCTS训练器
        mcts_trainer = MCTSTrainer(net=net, board_size=9, mcts_simulations=50)
        
        # 进行一局自我对弈
        states, policies, values = mcts_trainer.play_self_game(verbose=True)
        
        print(f"\n✓ 自我对弈测试完成")
        print(f"生成状态数: {len(states)}")
        print(f"策略数: {len(policies)}")
        print(f"价值数: {len(values)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 自我对弈测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training():
    """测试训练功能"""
    print("\n=== 训练测试 ===")
    
    try:
        # 创建训练器
        trainer = AlphaZeroTrainer(board_size=9, num_channels=32)
        
        # 进行一轮训练迭代
        success = trainer.training_iteration(
            self_play_games=2,
            training_epochs=2,
            batch_size=16
        )
        
        if success:
            print(f"\n✓ 训练测试完成")
            print(f"训练迭代: {trainer.training_stats['iterations']}")
            print(f"数据集大小: {len(trainer.dataset)}")
            return True
        else:
            print(f"✗ 训练测试失败: 没有生成有效数据")
            return False
        
    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("五子棋AI训练框架测试")
    print("=" * 40)
    
    tests = [
        test_self_play,
        test_training
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n测试结果: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("🎉 训练框架实现正常工作！")
    else:
        print("❌ 部分测试失败，需要检查实现")