#!/usr/bin/env python3
"""
五子棋神经网络快速测试脚本
用于验证修复后的功能是否正常
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """快速测试核心功能"""
    print("🚀 五子棋神经网络快速测试")
    print("=" * 50)
    
    try:
        # 测试1: 导入模块
        print("📦 测试模块导入...")
        import torch
        import numpy as np
        from net import GomokuNet, GomokuTrainer
        from board import Board
        print("✅ 所有模块导入成功")
        
        # 测试2: 创建网络
        print("\n🧠 测试网络创建...")
        net = GomokuNet(num_residual_blocks=2)  # 使用较小的网络进行快速测试
        print(f"✅ 网络创建成功，参数数量: {sum(p.numel() for p in net.parameters()):,}")
        
        # 测试3: 前向传播
        print("\n⚡ 测试前向传播...")
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 9, 9)
        with torch.no_grad():
            policy, value = net(input_tensor)
        print(f"✅ 策略输出形状: {policy.shape}")
        print(f"✅ 价值输出形状: {value.shape}")
        
        # 测试4: 棋盘转换（修复后的功能）
        print("\n🎯 测试棋盘转换...")
        board = Board()
        board.make_move(4, 4, 1)  # 黑子
        board.make_move(4, 5, 2)  # 白子
        
        tensor = net._board_to_tensor(board)
        print(f"✅ 棋盘转换成功，张量形状: {tensor.shape}")
        
        # 检查通道内容（这里是之前出错的地方）
        print(f"✅ 通道0非零元素: {torch.sum(tensor[0]).item()}")
        print(f"✅ 通道1非零元素: {torch.sum(tensor[1]).item()}")
        print(f"✅ 通道2非零元素: {torch.sum(tensor[2]).item()}")
        
        # 测试5: 预测功能
        print("\n🎲 测试预测功能...")
        policy_probs, value_pred = net.predict(board)
        print(f"✅ 策略概率和: {np.sum(policy_probs):.6f}")
        print(f"✅ 价值预测: {value_pred:.6f}")
        
        # 测试6: 训练器创建
        print("\n🏋️ 测试训练器...")
        trainer = GomokuTrainer(net)
        print("✅ 训练器创建成功")
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！神经网络功能正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script():
    """测试训练脚本是否可以正常启动"""
    print("\n🔧 测试训练脚本...")
    try:
        # 检查训练脚本文件
        if os.path.exists('train_neural_network.py'):
            print("✅ 训练脚本文件存在")
        else:
            print("❌ 训练脚本文件不存在")
            return False
            
        # 检查评估脚本文件
        if os.path.exists('evaluate_neural_network.py'):
            print("✅ 评估脚本文件存在")
        else:
            print("❌ 评估脚本文件不存在")
            return False
            
        print("✅ 所有脚本文件检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 脚本检查失败: {e}")
        return False

def show_usage_guide():
    """显示使用指南"""
    print("\n" + "=" * 60)
    print("📖 使用指南")
    print("=" * 60)
    
    print("\n🎯 快速开始:")
    print("1. 运行完整测试: python test_neural_network.py")
    print("2. 开始训练: python train_neural_network.py")
    print("3. 评估模型: python evaluate_neural_network.py")
    print("4. 人机对战: python play.py")
    
    print("\n⚙️ 训练配置建议:")
    print("- 快速测试: num_residual_blocks=2, train_size=200, epochs=10")
    print("- 正常训练: num_residual_blocks=4, train_size=1000, epochs=50")
    print("- 深度训练: num_residual_blocks=6, train_size=2000, epochs=100")
    
    print("\n📊 性能监控:")
    print("- 训练过程会显示损失曲线")
    print("- 自动保存最佳模型")
    print("- 早停机制防止过拟合")
    
    print("\n🔍 故障排除:")
    print("- 内存不足: 减少batch_size和train_size")
    print("- 训练太慢: 减少num_residual_blocks")
    print("- 性能不佳: 增加训练数据和轮数")
    
    print("\n📁 重要文件:")
    print("- net.py: 神经网络核心")
    print("- train_neural_network.py: 训练脚本")
    print("- evaluate_neural_network.py: 评估脚本")
    print("- 使用指南.md: 详细文档")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # 运行快速测试
    success = quick_test()
    
    # 测试脚本文件
    script_ok = test_training_script()
    
    # 显示使用指南
    show_usage_guide()
    
    # 总结
    print("\n🏁 测试总结:")
    if success and script_ok:
        print("✅ 所有功能正常，可以开始使用！")
        print("\n💡 建议下一步:")
        print("1. 查看详细文档: 使用指南.md")
        print("2. 开始训练: python train_neural_network.py")
    else:
        print("❌ 存在问题，请检查错误信息")
        print("\n🔧 建议操作:")
        print("1. 检查依赖安装: pip install torch numpy matplotlib")
        print("2. 查看错误日志")
        print("3. 参考使用指南.md解决问题")