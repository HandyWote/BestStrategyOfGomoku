#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
五子棋AI对战接口
提供人机对战、AI对战、训练等功能的统一入口
"""

import numpy as np
import os
import sys
import time
import argparse
from datetime import datetime

from board import GomokuBoard
from mcts import MCTS
from net import GomokuNet
from game import GomokuAI, GameEngine, Tournament
from train import AlphaZeroTrainer

class HumanPlayer:
    """人类玩家"""
    
    def __init__(self, name="Human"):
        self.name = name
    
    def get_move(self, board_state):
        """获取人类玩家的移动"""
        print(f"\n轮到 {self.name} ({'黑棋' if board_state.current_player == 1 else '白棋'})")
        board_state.display()
        
        while True:
            try:
                # 获取用户输入
                move_input = input("请输入移动 (格式: 行,列 或 行 列): ").strip()
                
                if move_input.lower() in ['quit', 'exit', 'q']:
                    print("游戏退出")
                    return None
                
                # 解析输入
                if ',' in move_input:
                    row, col = map(int, move_input.split(','))
                else:
                    parts = move_input.split()
                    if len(parts) == 2:
                        row, col = map(int, parts)
                    else:
                        raise ValueError("输入格式错误")
                
                # 检查移动是否有效
                if board_state.is_valid_move(row, col):
                    return (row, col)
                else:
                    print(f"无效移动: ({row}, {col})，请重新输入")
                    
            except (ValueError, IndexError):
                print("输入格式错误，请使用 '行,列' 或 '行 列' 格式 (例如: 4,4 或 4 4)")
                print("输入 'quit' 或 'q' 退出游戏")
            except KeyboardInterrupt:
                print("\n游戏被中断")
                return None

class GameInterface:
    """游戏界面管理器"""
    
    def __init__(self):
        self.board_size = 9
        self.models_dir = "models"
        self.ensure_models_dir()
    
    def ensure_models_dir(self):
        """确保模型目录存在"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"创建模型目录: {self.models_dir}")
    
    def list_available_models(self):
        """列出可用的模型"""
        models = []
        if os.path.exists(self.models_dir):
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.pth'):
                    models.append(filename)
        return sorted(models)
    
    def select_model(self, prompt="选择模型"):
        """让用户选择模型"""
        models = self.list_available_models()
        
        if not models:
            print("没有找到可用的模型")
            return None
        
        print(f"\n{prompt}:")
        print("0. 不使用模型 (纯MCTS)")
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        while True:
            try:
                choice = input("请选择 (输入数字): ").strip()
                choice = int(choice)
                
                if choice == 0:
                    return None
                elif 1 <= choice <= len(models):
                    return os.path.join(self.models_dir, models[choice - 1])
                else:
                    print(f"请输入 0-{len(models)} 之间的数字")
                    
            except (ValueError, KeyboardInterrupt):
                print("输入无效，请重新输入")
    
    def create_ai_player(self, name, model_path=None, strength="normal"):
        """创建AI玩家"""
        # 根据强度设置参数
        strength_configs = {
            "easy": {"mcts_time": 0.2, "mcts_iterations": 50},
            "normal": {"mcts_time": 1.0, "mcts_iterations": 200},
            "hard": {"mcts_time": 2.0, "mcts_iterations": 500},
            "expert": {"mcts_time": 3.0, "mcts_iterations": 1000}
        }
        
        config = strength_configs.get(strength, strength_configs["normal"])
        
        return GomokuAI(
            name=name,
            net_path=model_path,
            mcts_time=config["mcts_time"],
            mcts_iterations=config["mcts_iterations"]
        )
    
    def human_vs_ai(self):
        """人机对战"""
        print("\n=== 人机对战 ===")
        
        # 选择AI强度
        print("选择AI强度:")
        print("1. 简单 (0.2秒思考)")
        print("2. 普通 (1.0秒思考)")
        print("3. 困难 (2.0秒思考)")
        print("4. 专家 (3.0秒思考)")
        
        strength_map = {"1": "easy", "2": "normal", "3": "hard", "4": "expert"}
        
        while True:
            choice = input("请选择强度 (1-4): ").strip()
            if choice in strength_map:
                strength = strength_map[choice]
                break
            print("请输入 1-4")
        
        # 选择模型
        model_path = self.select_model("选择AI模型")
        
        # 选择先后手
        while True:
            color_choice = input("选择你的颜色 (1=黑棋先手, 2=白棋后手): ").strip()
            if color_choice in ["1", "2"]:
                human_first = (color_choice == "1")
                break
            print("请输入 1 或 2")
        
        # 创建玩家
        human = HumanPlayer("玩家")
        ai = self.create_ai_player(f"AI-{strength.title()}", model_path, strength)
        
        # 开始游戏
        engine = GameEngine(board_size=self.board_size)
        
        if human_first:
            winner, game_data = engine.play_game(human, ai, verbose=True)
        else:
            winner, game_data = engine.play_game(ai, human, verbose=True)
        
        # 显示结果
        if winner == 1:
            result = "黑棋获胜" if human_first else "AI获胜"
        elif winner == -1:
            result = "AI获胜" if human_first else "白棋获胜"
        else:
            result = "平局"
        
        print(f"\n游戏结束: {result}")
        
        # 询问是否保存游戏
        save_choice = input("是否保存游戏记录? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{timestamp}.json"
            engine.save_game(game_data, filename)
    
    def ai_vs_ai(self):
        """AI对战"""
        print("\n=== AI对战 ===")
        
        # 选择第一个AI
        print("配置第一个AI (黑棋):")
        model1_path = self.select_model("选择第一个AI的模型")
        
        print("选择第一个AI强度:")
        print("1. 简单  2. 普通  3. 困难  4. 专家")
        strength1 = ["easy", "normal", "hard", "expert"][int(input("请选择 (1-4): ")) - 1]
        
        # 选择第二个AI
        print("\n配置第二个AI (白棋):")
        model2_path = self.select_model("选择第二个AI的模型")
        
        print("选择第二个AI强度:")
        print("1. 简单  2. 普通  3. 困难  4. 专家")
        strength2 = ["easy", "normal", "hard", "expert"][int(input("请选择 (1-4): ")) - 1]
        
        # 创建AI
        ai1 = self.create_ai_player(f"AI1-{strength1.title()}", model1_path, strength1)
        ai2 = self.create_ai_player(f"AI2-{strength2.title()}", model2_path, strength2)
        
        # 选择对战局数
        num_games = int(input("对战局数 (默认1): ") or "1")
        
        if num_games == 1:
            # 单局对战
            engine = GameEngine(board_size=self.board_size)
            winner, game_data = engine.play_game(ai1, ai2, verbose=True)
        else:
            # 多局对战
            tournament = Tournament(board_size=self.board_size)
            results = tournament.run_match(ai1, ai2, num_games=num_games, verbose=False)
    
    def train_model(self):
        """训练模型"""
        print("\n=== 模型训练 ===")
        
        # 训练参数配置
        print("配置训练参数:")
        
        board_size = int(input(f"棋盘大小 (默认{self.board_size}): ") or str(self.board_size))
        num_channels = int(input("网络通道数 (默认64): ") or "64")
        
        iterations = int(input("训练迭代次数 (默认10): ") or "10")
        self_play_games = int(input("每轮自我对弈局数 (默认20): ") or "20")
        training_epochs = int(input("每轮训练轮数 (默认10): ") or "10")
        
        # 创建训练器
        trainer = AlphaZeroTrainer(
            board_size=board_size,
            num_channels=num_channels
        )
        
        # 检查是否有已有模型
        models = self.list_available_models()
        if models:
            load_choice = input("是否加载已有模型继续训练? (y/n): ").strip().lower()
            if load_choice in ['y', 'yes']:
                model_path = self.select_model("选择要加载的模型")
                if model_path:
                    trainer.load_model(model_path)
        
        # 开始训练
        print(f"\n开始训练: {iterations} 轮迭代")
        start_time = time.time()
        
        try:
            for iteration in range(iterations):
                print(f"\n{'='*50}")
                print(f"训练迭代 {iteration + 1}/{iterations}")
                print(f"{'='*50}")
                
                success = trainer.training_iteration(
                    self_play_games=self_play_games,
                    training_epochs=training_epochs,
                    batch_size=32
                )
                
                if not success:
                    print("训练迭代失败，停止训练")
                    break
                
                # 每5轮保存一次模型
                if (iteration + 1) % 5 == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = f"gomoku_model_iter{iteration + 1}_{timestamp}.pth"
                    model_path = os.path.join(self.models_dir, model_name)
                    trainer.save_model(model_path)
                
                # 每10轮评估一次
                if (iteration + 1) % 10 == 0:
                    trainer.evaluate_model(num_games=10)
        
        except KeyboardInterrupt:
            print("\n训练被中断")
        
        # 保存最终模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_name = f"gomoku_model_final_{timestamp}.pth"
        final_model_path = os.path.join(self.models_dir, final_model_name)
        trainer.save_model(final_model_path)
        
        training_time = time.time() - start_time
        print(f"\n训练完成！")
        print(f"训练时间: {training_time/60:.1f} 分钟")
        print(f"最终模型: {final_model_path}")
    
    def benchmark_test(self):
        """基准测试"""
        print("\n=== 基准测试 ===")
        
        # 测试MCTS性能
        print("测试MCTS性能...")
        
        board = GomokuBoard(size=self.board_size)
        mcts = MCTS(time_limit=1.0, max_iterations=1000)
        
        start_time = time.time()
        move = mcts.search(board)
        search_time = time.time() - start_time
        
        print(f"MCTS搜索时间: {search_time:.3f}秒")
        print(f"选择移动: {move}")
        
        # 测试不同强度AI的性能
        print("\n测试AI性能...")
        
        strengths = ["easy", "normal", "hard"]
        for strength in strengths:
            ai = self.create_ai_player(f"Test-{strength}", None, strength)
            
            start_time = time.time()
            move = ai.get_move(board)
            think_time = time.time() - start_time
            
            print(f"{strength.title()} AI: {think_time:.3f}秒, 移动: {move}")
    
    def show_main_menu(self):
        """显示主菜单"""
        print("\n" + "="*50)
        print("           五子棋AI对战系统")
        print("="*50)
        print("1. 人机对战")
        print("2. AI对战")
        print("3. 训练模型")
        print("4. 基准测试")
        print("5. 查看模型")
        print("0. 退出")
        print("="*50)
    
    def show_models(self):
        """显示可用模型"""
        print("\n=== 可用模型 ===")
        models = self.list_available_models()
        
        if not models:
            print("没有找到任何模型")
            print("请先训练模型或将模型文件放入 models/ 目录")
        else:
            for i, model in enumerate(models, 1):
                model_path = os.path.join(self.models_dir, model)
                size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
                print(f"{i}. {model}")
                print(f"   大小: {size:.1f} MB")
                print(f"   修改时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run(self):
        """运行主程序"""
        print("欢迎使用五子棋AI对战系统！")
        
        while True:
            try:
                self.show_main_menu()
                choice = input("请选择功能 (0-5): ").strip()
                
                if choice == "0":
                    print("感谢使用，再见！")
                    break
                elif choice == "1":
                    self.human_vs_ai()
                elif choice == "2":
                    self.ai_vs_ai()
                elif choice == "3":
                    self.train_model()
                elif choice == "4":
                    self.benchmark_test()
                elif choice == "5":
                    self.show_models()
                else:
                    print("无效选择，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n程序被中断")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                import traceback
                traceback.print_exc()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="五子棋AI对战系统")
    parser.add_argument("--mode", choices=["play", "train", "test"], 
                       default="play", help="运行模式")
    parser.add_argument("--board-size", type=int, default=9, 
                       help="棋盘大小")
    parser.add_argument("--model", type=str, 
                       help="模型文件路径")
    parser.add_argument("--iterations", type=int, default=10, 
                       help="训练迭代次数")
    parser.add_argument("--games", type=int, default=20, 
                       help="每轮自我对弈局数")
    
    args = parser.parse_args()
    
    if args.mode == "play":
        # 交互式游戏界面
        interface = GameInterface()
        interface.board_size = args.board_size
        interface.run()
        
    elif args.mode == "train":
        # 命令行训练模式
        print("开始训练模式...")
        trainer = AlphaZeroTrainer(board_size=args.board_size)
        
        if args.model and os.path.exists(args.model):
            trainer.load_model(args.model)
        
        for i in range(args.iterations):
            print(f"训练迭代 {i+1}/{args.iterations}")
            trainer.training_iteration(
                self_play_games=args.games,
                training_epochs=10
            )
        
        # 保存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"gomoku_model_{timestamp}.pth"
        trainer.save_model(model_path)
        
    elif args.mode == "test":
        # 测试模式
        interface = GameInterface()
        interface.board_size = args.board_size
        interface.benchmark_test()

if __name__ == "__main__":
    main()