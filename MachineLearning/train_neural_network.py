#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五子棋神经网络训练脚本（改进版）

实现特性：
- 改进的神经网络架构（残差块、dropout等）
- 学习率调度和早停机制
- 数据增强和验证集分割
- 详细的训练监控和可视化
- 模型性能评估

作者: AI Assistant
日期: 2024年
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from net import GomokuNet, GomokuTrainer, GomokuDataset
from board import GomokuBoard
from mcts import MCTS

class TrainingConfig:
    """训练配置类"""
    
    def __init__(self):
        # 网络参数
        self.board_size = 9
        self.num_channels = 64
        self.num_residual_blocks = 6
        
        # 训练参数
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 100
        self.patience = 15
        
        # 损失权重
        self.policy_weight = 1.0
        self.value_weight = 1.0
        
        # 数据参数
        self.train_ratio = 0.8
        self.val_ratio = 0.2
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 文件路径
        self.model_dir = 'models'
        self.data_dir = 'data'
        self.log_dir = 'logs'
        
        # 创建目录
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

def set_random_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_training_data(num_games=1000, board_size=9):
    """生成训练数据
    
    Args:
        num_games: 生成的游戏数量
        board_size: 棋盘大小
        
    Returns:
        GomokuDataset: 训练数据集
    """
    print(f"正在生成 {num_games} 局游戏的训练数据...")
    
    dataset = GomokuDataset()
    
    for game_idx in tqdm(range(num_games), desc="生成训练数据"):
        board = GomokuBoard(size=board_size)
        mcts = MCTS(time_limit=2.0, max_iterations=100)
        
        game_data = []
        
        while not board.is_game_over():
            # 记录当前状态
            current_player = board.current_player
            board_state = board.get_board_copy()
            
            # 使用MCTS搜索最佳移动
            move = mcts.search(board)
            if move is None:
                # 如果MCTS没有找到移动，随机选择
                valid_moves = board.get_valid_moves()
                if not valid_moves:
                    break
                move = random.choice(valid_moves)
            
            row, col = move
            
            # 创建简单的策略分布（在有效位置上均匀分布）
            valid_moves = board.get_valid_moves()
            action_probs = np.zeros(board_size * board_size)
            if valid_moves:
                prob_per_move = 1.0 / len(valid_moves)
                for valid_move in valid_moves:
                    action_idx = valid_move[0] * board_size + valid_move[1]
                    action_probs[action_idx] = prob_per_move
            
            action = row * board_size + col
            
            # 记录数据
            game_data.append({
                'board': board_state,
                'player': current_player,
                'policy': action_probs,
                'action': action
            })
            
            # 执行动作
            board.make_move(row, col)
        
        # 确定游戏结果
        winner = board.get_winner()
        
        # 为每个状态分配价值
        for data in game_data:
            if winner == 0:  # 平局
                value = 0.0
            elif winner == data['player']:
                value = 1.0  # 胜利
            else:
                value = -1.0  # 失败
            
            # 添加到数据集
            dataset.add_sample(data['board'], data['policy'], value)
    
    print(f"训练数据生成完成，共 {len(dataset)} 个样本")
    return dataset

def create_data_loaders(dataset, config):
    """创建训练和验证数据加载器
    
    Args:
        dataset: 完整数据集
        config: 训练配置
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # 计算分割大小
    total_size = len(dataset)
    train_size = int(total_size * config.train_ratio)
    val_size = total_size - train_size
    
    # 随机分割数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0  # Windows兼容性
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"数据分割完成: 训练集 {train_size} 样本, 验证集 {val_size} 样本")
    return train_loader, val_loader

def plot_training_history(trainer, save_path=None):
    """绘制训练历史
    
    Args:
        trainer: 训练器对象
        save_path: 保存路径
    """
    history = trainer.train_history
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('训练历史', fontsize=16)
    
    # 总损失
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('总损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # 策略损失
    axes[0, 1].plot(history['policy_loss'])
    axes[0, 1].set_title('策略损失')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Policy Loss')
    axes[0, 1].grid(True)
    
    # 价值损失
    axes[1, 0].plot(history['value_loss'])
    axes[1, 0].set_title('价值损失')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Value Loss')
    axes[1, 0].grid(True)
    
    # 学习率
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_title('学习率')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图表已保存到: {save_path}")
    
    plt.show()

def train_model(config):
    """训练模型主函数
    
    Args:
        config: 训练配置
    """
    print("=== 五子棋神经网络训练开始 ===")
    print(f"设备: {config.device}")
    print(f"网络参数: {config.num_channels} 通道, {config.num_residual_blocks} 残差块")
    print(f"训练参数: 学习率 {config.learning_rate}, 批大小 {config.batch_size}")
    
    # 设置随机种子
    set_random_seed(42)
    
    # 生成或加载训练数据
    dataset_path = os.path.join(config.data_dir, 'training_data.pkl')
    
    if os.path.exists(dataset_path):
        print(f"加载现有训练数据: {dataset_path}")
        dataset = GomokuDataset()
        dataset.load_data(dataset_path)
    else:
        print("生成新的训练数据...")
        dataset = generate_training_data(num_games=500, board_size=config.board_size)
        dataset.save_data(dataset_path)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(dataset, config)
    
    # 创建网络和训练器
    net = GomokuNet(
        board_size=config.board_size,
        num_channels=config.num_channels,
        num_residual_blocks=config.num_residual_blocks
    )
    
    trainer = GomokuTrainer(
        net=net,
        learning_rate=config.learning_rate,
        device=config.device,
        patience=config.patience
    )
    
    print(f"网络参数总数: {sum(p.numel() for p in net.parameters()):,}")
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config.num_epochs):
        start_time = time.time()
        
        # 训练阶段
        train_metrics = trainer.train_step(
            train_loader, 
            policy_weight=config.policy_weight,
            value_weight=config.value_weight
        )
        
        # 验证阶段
        val_metrics = trainer.validate(val_loader)
        
        # 记录损失
        train_losses.append(train_metrics['total_loss'])
        val_losses.append(val_metrics['val_total_loss'])
        
        # 保存最佳模型
        if val_metrics['val_total_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_total_loss']
            best_model_path = os.path.join(config.model_dir, 'best_model.pth')
            trainer.save_model(best_model_path)
        
        # 打印进度
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config.num_epochs} ({epoch_time:.2f}s)")
        print(f"  训练 - 总损失: {train_metrics['total_loss']:.6f}, "
              f"策略: {train_metrics['policy_loss']:.6f}, "
              f"价值: {train_metrics['value_loss']:.6f}")
        print(f"  验证 - 总损失: {val_metrics['val_total_loss']:.6f}, "
              f"策略: {val_metrics['val_policy_loss']:.6f}, "
              f"价值: {val_metrics['val_value_loss']:.6f}")
        print(f"  学习率: {train_metrics['learning_rate']:.8f}, "
              f"耐心: {val_metrics['patience_counter']}/{config.patience}")
        
        # 早停检查
        if trainer.early_stop:
            print(f"\n早停触发！在第 {epoch+1} 轮停止训练")
            break
    
    # 保存最终模型
    final_model_path = os.path.join(config.model_dir, 'final_model.pth')
    trainer.save_model(final_model_path)
    
    # 绘制训练历史
    plot_path = os.path.join(config.log_dir, 'training_history.png')
    plot_training_history(trainer, plot_path)
    
    print("\n=== 训练完成 ===")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最佳模型保存在: {best_model_path}")
    print(f"最终模型保存在: {final_model_path}")

def main():
    """主函数"""
    # 创建配置
    config = TrainingConfig()
    
    # 开始训练
    try:
        train_model(config)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()