#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五子棋AI神经网络模型
实现策略网络和价值网络，用于指导MCTS搜索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import os

class ResidualBlock(nn.Module):
    """残差块：提高网络深度和特征提取能力"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(0.1)  # 防止过拟合
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        out = F.relu(out)
        return out

class GomokuNet(nn.Module):
    """五子棋神经网络模型（改进版）
    
    结合策略网络和价值网络：
    - 策略网络：输出每个位置的落子概率
    - 价值网络：评估当前局面的胜率
    
    改进特性：
    - 残差连接：提高网络深度和训练稳定性
    - Dropout层：防止过拟合
    - 改进的批归一化：更好的训练收敛
    """
    
    def __init__(self, board_size=9, num_channels=64, num_residual_blocks=4):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.num_residual_blocks = num_residual_blocks
        
        # 输入层：棋盘状态 (batch_size, 3, board_size, board_size)
        # 3个通道：当前玩家棋子、对手棋子、空位置
        
        # 初始卷积层：将输入转换为特征表示
        self.initial_conv = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(num_channels)
        
        # 残差块：深度特征提取
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_residual_blocks)
        ])
        
        # 特征提取后的dropout
        self.feature_dropout = nn.Dropout2d(0.1)
        
        # 策略头：输出每个位置的落子概率
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # 价值头：评估局面价值
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        """前向传播（改进版）
        
        Args:
            x: 输入张量 (batch_size, 3, board_size, board_size)
            
        Returns:
            policy: 策略概率 (batch_size, board_size * board_size)
            value: 局面价值 (batch_size, 1)
        """
        # 初始特征提取
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # 通过残差块进行深度特征提取
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # 特征dropout
        x = self.feature_dropout(x)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # 展平
        policy = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # 展平
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # 输出范围 [-1, 1]
        
        return policy, value
    
    def predict(self, board_state):
        """预测单个棋盘状态
        
        Args:
            board_state: GomokuBoard实例
            
        Returns:
            policy_probs: 策略概率数组 (board_size * board_size,)
            value: 局面价值 (标量)
        """
        self.eval()
        with torch.no_grad():
            # 转换棋盘状态为网络输入
            input_tensor = self._board_to_tensor(board_state)
            input_tensor = input_tensor.unsqueeze(0)  # 添加batch维度
            
            # 前向传播
            policy_log_probs, value = self.forward(input_tensor)
            
            # 转换为概率
            policy_probs = torch.exp(policy_log_probs).squeeze(0).numpy()
            value = value.squeeze(0).item()
            
            return policy_probs, value
    
    def _board_to_tensor(self, board_state):
        """将棋盘状态转换为网络输入张量
        
        Args:
            board_state: GomokuBoard实例
            
        Returns:
            tensor: (3, board_size, board_size) 张量
        """
        board = board_state.board
        current_player = board_state.current_player
        
        # 创建3个通道
        channels = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 通道0：当前玩家的棋子
        channels[0] = (board == current_player).astype(np.float32)
        
        # 通道1：对手的棋子
        channels[1] = (board == -current_player).astype(np.float32)
        
        # 通道2：空位置
        channels[2] = (board == 0).astype(np.float32)
        
        return torch.FloatTensor(channels)

class GomokuDataset(Dataset):
    """五子棋训练数据集"""
    
    def __init__(self, data_file=None):
        self.data = []
        if data_file and os.path.exists(data_file):
            self.load_data(data_file)
    
    def add_sample(self, board_state, policy_target, value_target):
        """添加训练样本
        
        Args:
            board_state: GomokuBoard实例
            policy_target: 策略目标 (board_size * board_size,)
            value_target: 价值目标 (标量)
        """
        self.data.append({
            'board': board_state.board.copy(),
            'current_player': board_state.current_player,
            'policy': policy_target.copy() if isinstance(policy_target, np.ndarray) else policy_target,
            'value': float(value_target)
        })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 重建棋盘状态
        from board import GomokuBoard
        board_state = GomokuBoard(size=len(sample['board']))
        board_state.board = np.array(sample['board'])
        board_state.current_player = sample['current_player']
        
        # 转换为张量
        net = GomokuNet(board_size=len(sample['board']))
        input_tensor = net._board_to_tensor(board_state)
        
        policy_tensor = torch.FloatTensor(sample['policy'])
        value_tensor = torch.FloatTensor([sample['value']])
        
        return input_tensor, policy_tensor, value_tensor
    
    def save_data(self, filename):
        """保存数据到文件"""
        with open(filename, 'w') as f:
            json.dump(self.data, f)
    
    def load_data(self, filename):
        """从文件加载数据"""
        with open(filename, 'r') as f:
            self.data = json.load(f)

class GomokuTrainer:
    """五子棋网络训练器（改进版）
    
    改进特性：
    - 学习率调度：自适应调整学习率
    - 早停机制：防止过拟合
    - 梯度裁剪：稳定训练过程
    - 详细的训练指标：监控训练进度
    """
    
    def __init__(self, net, learning_rate=0.001, device=None, patience=10):
        self.net = net
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        # 学习率调度器：当验证损失不再下降时减少学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 早停机制
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.early_stop = False
        
        # 训练历史
        self.train_history = {
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'learning_rate': []
        }
        
        # 损失函数
        self.policy_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.value_loss_fn = nn.MSELoss()
        
        print(f"训练器初始化完成，使用设备: {self.device}")
    
    def train_step(self, dataloader, policy_weight=1.0, value_weight=1.0):
        """执行一个训练步骤（改进版）
        
        Args:
            dataloader: 数据加载器
            policy_weight: 策略损失权重
            value_weight: 价值损失权重
            
        Returns:
            dict: 包含详细训练指标的字典
        """
        self.net.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for batch_inputs, batch_policy_targets, batch_value_targets in dataloader:
            # 移动到设备
            batch_inputs = batch_inputs.to(self.device)
            batch_policy_targets = batch_policy_targets.to(self.device)
            batch_value_targets = batch_value_targets.to(self.device)
            
            # 前向传播
            policy_outputs, value_outputs = self.net(batch_inputs)
            
            # 计算损失
            policy_loss = self.policy_loss_fn(policy_outputs, batch_policy_targets)
            value_loss = self.value_loss_fn(value_outputs, batch_value_targets)
            
            # 总损失
            loss = policy_weight * policy_loss + value_weight * value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        # 计算平均损失
        avg_total_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # 记录训练历史
        self.train_history['total_loss'].append(avg_total_loss)
        self.train_history['policy_loss'].append(avg_policy_loss)
        self.train_history['value_loss'].append(avg_value_loss)
        self.train_history['learning_rate'].append(current_lr)
        
        return {
            'total_loss': avg_total_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'learning_rate': current_lr,
            'num_batches': num_batches
        }
    
    def validate(self, val_dataloader):
        """验证模型性能
        
        Args:
            val_dataloader: 验证数据加载器
            
        Returns:
            dict: 验证指标
        """
        self.net.eval()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_policy_targets, batch_value_targets in val_dataloader:
                # 移动到设备
                batch_inputs = batch_inputs.to(self.device)
                batch_policy_targets = batch_policy_targets.to(self.device)
                batch_value_targets = batch_value_targets.to(self.device)
                
                # 前向传播
                policy_outputs, value_outputs = self.net(batch_inputs)
                
                # 计算损失
                policy_loss = self.policy_loss_fn(policy_outputs, batch_policy_targets)
                value_loss = self.value_loss_fn(value_outputs, batch_value_targets)
                loss = policy_loss + value_loss
                
                # 统计
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        
        # 学习率调度
        self.scheduler.step(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < self.best_loss:
            self.best_loss = avg_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stop = True
        
        return {
            'val_total_loss': avg_val_loss,
            'val_policy_loss': total_policy_loss / num_batches,
            'val_value_loss': total_value_loss / num_batches,
            'early_stop': self.early_stop,
            'patience_counter': self.patience_counter
        }
    
    def save_model(self, filepath):
        """保存模型（改进版）"""
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'board_size': self.net.board_size,
            'num_channels': self.net.num_channels,
            'num_residual_blocks': self.net.num_residual_blocks,
            'train_history': self.train_history,
            'best_loss': self.best_loss,
            'patience_counter': self.patience_counter
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型（改进版）"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态（如果存在）
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载训练历史（如果存在）
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        # 加载早停相关状态（如果存在）
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']
        
        print(f"模型已从 {filepath} 加载")
        print(f"最佳验证损失: {self.best_loss:.6f}")
        print(f"当前耐心计数: {self.patience_counter}/{self.patience}")

# 测试函数
def test_network():
    """测试神经网络基本功能"""
    print("=== 神经网络测试 ===")
    
    try:
        # 创建网络
        net = GomokuNet(board_size=9, num_channels=32)
        print(f"✓ 网络创建成功，参数数量: {sum(p.numel() for p in net.parameters())}")
        
        # 测试前向传播
        batch_size = 4
        input_tensor = torch.randn(batch_size, 3, 9, 9)
        policy, value = net(input_tensor)
        
        print(f"✓ 前向传播成功")
        print(f"  策略输出形状: {policy.shape}")
        print(f"  价值输出形状: {value.shape}")
        
        # 测试单个预测
        from board import GomokuBoard
        board = GomokuBoard(size=9)
        board.make_move(4, 4)
        
        policy_probs, value_pred = net.predict(board)
        print(f"✓ 单个预测成功")
        print(f"  策略概率和: {np.sum(policy_probs):.3f}")
        print(f"  价值预测: {value_pred:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training():
    """测试训练功能"""
    print("\n=== 训练功能测试 ===")
    
    try:
        # 创建网络和训练器
        net = GomokuNet(board_size=9, num_channels=16)  # 小网络用于测试
        trainer = GomokuTrainer(net, learning_rate=0.01)
        
        # 创建测试数据
        dataset = GomokuDataset()
        
        from board import GomokuBoard
        for i in range(10):  # 创建10个样本
            board = GomokuBoard(size=9)
            # 随机下几步棋
            for _ in range(np.random.randint(1, 5)):
                valid_moves = board.get_valid_moves()
                if valid_moves:
                    move = np.random.choice(len(valid_moves))
                    board.make_move(valid_moves[move][0], valid_moves[move][1])
            
            # 创建随机目标
            policy_target = np.random.random(81)
            policy_target = policy_target / np.sum(policy_target)  # 归一化
            value_target = np.random.uniform(-1, 1)
            
            dataset.add_sample(board, policy_target, value_target)
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # 执行一个训练步骤
        total_loss, policy_loss, value_loss = trainer.train_step(dataloader)
        
        print(f"✓ 训练步骤成功")
        print(f"  总损失: {total_loss:.4f}")
        print(f"  策略损失: {policy_loss:.4f}")
        print(f"  价值损失: {value_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("五子棋神经网络测试")
    print("=" * 40)
    
    tests = [
        test_network,
        test_training
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n测试结果: {passed}/{len(tests)} 通过")
    
    if passed == len(tests):
        print("🎉 神经网络实现正常工作！")
    else:
        print("❌ 部分测试失败，需要检查实现")