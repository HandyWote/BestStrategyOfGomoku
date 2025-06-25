#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经网络测试脚本
测试五子棋神经网络的基本功能
"""

import sys
import os
import traceback

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ NumPy版本: {np.__version__}")
        
        from net import GomokuNet, GomokuDataset, GomokuTrainer
        print("✓ 神经网络模块导入成功")
        
        from board import GomokuBoard
        print("✓ 棋盘模块导入成功")
        
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        traceback.print_exc()
        return False

def test_network_creation():
    """测试网络创建"""
    print("\n=== 测试网络创建 ===")
    try:
        from net import GomokuNet
        
        # 创建网络
        net = GomokuNet(board_size=9, num_channels=32)
        print(f"✓ 网络创建成功")
        
        # 计算参数数量
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"✓ 总参数数量: {total_params:,}")
        print(f"✓ 可训练参数: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ 网络创建失败: {e}")
        traceback.print_exc()
        return False

def test_forward_pass():
    """测试前向传播"""
    print("\n=== 测试前向传播 ===")
    try:
        import torch
        from net import GomokuNet
        
        net = GomokuNet(board_size=9, num_channels=16)  # 小网络用于测试
        
        # 创建测试输入
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 9, 9)
        print(f"✓ 输入张量形状: {input_tensor.shape}")
        
        # 前向传播
        policy, value = net(input_tensor)
        print(f"✓ 策略输出形状: {policy.shape}")
        print(f"✓ 价值输出形状: {value.shape}")
        
        # 检查输出范围
        print(f"✓ 策略输出范围: [{policy.min().item():.3f}, {policy.max().item():.3f}]")
        print(f"✓ 价值输出范围: [{value.min().item():.3f}, {value.max().item():.3f}]")
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        traceback.print_exc()
        return False

def test_board_conversion():
    """测试棋盘状态转换"""
    print("\n=== 测试棋盘状态转换 ===")
    try:
        from net import GomokuNet
        from board import GomokuBoard
        
        net = GomokuNet(board_size=9)
        board = GomokuBoard(size=9)
        
        # 下几步棋
        board.make_move(4, 4)  # 中心位置
        board.make_move(4, 5)  # 相邻位置
        print(f"✓ 棋盘状态设置完成")
        
        # 转换为张量
        tensor = net._board_to_tensor(board)
        print(f"✓ 张量形状: {tensor.shape}")
        print(f"✓ 张量数据类型: {tensor.dtype}")
        
        # 检查通道内容
        import torch
        print(f"✓ 通道0(当前玩家)非零元素: {torch.sum(tensor[0]).item()}")
        print(f"✓ 通道1(对手)非零元素: {torch.sum(tensor[1]).item()}")
        print(f"✓ 通道2(空位)非零元素: {torch.sum(tensor[2]).item()}")
        
        return True
    except Exception as e:
        print(f"✗ 棋盘转换失败: {e}")
        traceback.print_exc()
        return False

def test_prediction():
    """测试单个预测"""
    print("\n=== 测试单个预测 ===")
    try:
        from net import GomokuNet
        from board import GomokuBoard
        
        net = GomokuNet(board_size=9, num_channels=16)
        board = GomokuBoard(size=9)
        
        # 下几步棋
        board.make_move(4, 4)
        board.make_move(3, 3)
        
        # 预测
        policy_probs, value_pred = net.predict(board)
        print(f"✓ 策略概率形状: {policy_probs.shape}")
        print(f"✓ 策略概率和: {policy_probs.sum():.6f}")
        print(f"✓ 价值预测: {value_pred:.6f}")
        
        # 找到最高概率的位置
        best_move_idx = policy_probs.argmax()
        best_row = best_move_idx // 9
        best_col = best_move_idx % 9
        print(f"✓ 推荐移动: ({best_row}, {best_col})")
        
        return True
    except Exception as e:
        print(f"✗ 预测失败: {e}")
        traceback.print_exc()
        return False

def test_dataset():
    """测试数据集"""
    print("\n=== 测试数据集 ===")
    try:
        import numpy as np
        from net import GomokuDataset
        from board import GomokuBoard
        
        dataset = GomokuDataset()
        
        # 创建几个样本
        for i in range(3):
            board = GomokuBoard(size=9)
            # 随机下几步棋
            for _ in range(np.random.randint(1, 4)):
                valid_moves = board.get_valid_moves()
                if valid_moves:
                    move = valid_moves[np.random.randint(len(valid_moves))]
                    board.make_move(move[0], move[1])
            
            # 创建随机目标
            policy_target = np.random.random(81)
            policy_target = policy_target / np.sum(policy_target)
            value_target = np.random.uniform(-1, 1)
            
            dataset.add_sample(board, policy_target, value_target)
        
        print(f"✓ 数据集大小: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"✓ 样本输入形状: {sample[0].shape}")
        print(f"✓ 样本策略形状: {sample[1].shape}")
        print(f"✓ 样本价值形状: {sample[2].shape}")
        
        return True
    except Exception as e:
        print(f"✗ 数据集测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("五子棋神经网络测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_network_creation,
        test_forward_pass,
        test_board_conversion,
        test_prediction,
        test_dataset
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ 测试 {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！神经网络实现正常工作")
    else:
        print(f"❌ {total - passed} 个测试失败，需要检查实现")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)