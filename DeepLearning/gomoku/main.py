import argparse
import torch
import time
from board import GomokuBoard
from model import GomokuNet
from mcts import MCTS
from train import Trainer

def train_model():
    """训练五子棋AI模型"""
    model = GomokuNet()
    trainer = Trainer(model)
    
    print("\n开始训练五子棋AI模型...")
    for iteration in range(1, 10001):  # 10000次迭代
        metrics = trainer.train_iteration()
        
        print(f"\n迭代 {iteration}:")
        print(f"  总损失: {metrics['loss']:.4f}")
        print(f"  策略损失: {metrics['policy_loss']:.4f}") 
        print(f"  价值损失: {metrics['value_loss']:.4f}")
        print(f"  胜率: {metrics['win_rate']:.2%}")
        
        # 定期保存模型
        if iteration % 50 == 0:
            torch.save(model.state_dict(), f"gomoku_model_iter{iteration}.pth")
            print(f"已保存第{iteration}次迭代的模型")
            
    print("\n训练完成!")
    torch.save(model.state_dict(), "gomoku_model_final.pth")
    print("最终模型已保存为 gomoku_model_final.pth")

def play_human_vs_ai(model_path: str = "gomoku_model_final.pth"):
    """人机对战"""
    model = GomokuNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    mcts = MCTS(model)
    board = GomokuBoard()
    
    print("\n五子棋人机对战")
    print("您执白棋(○), AI执黑棋(●)")
    print("输入格式: 行号,列号 (例如: 4,4)")
    print("----------------------------------")
    
    while not board.winner:
        print(f"\n当前回合: {'玩家(白棋)' if board.current_player == -1 else 'AI(黑棋)'}")
        board.display(show_coords=True)
        
        if board.current_player == -1:  # 玩家回合
            while True:
                try:
                    move = input("请输入您的落子位置: ")
                    row, col = map(int, move.split(','))
                    if not (0 <= row < 9 and 0 <= col < 9):
                        print("错误: 行号和列号必须在0-8范围内")
                        continue
                        
                    if board.make_move(row, col):
                        break
                    print("错误: 该位置已有棋子，请重新输入")
                except:
                    print("错误: 请输入正确的格式，如: 4,4")
        else:  # AI回合
            print("\nAI正在思考...")
            action = mcts.get_move(board, temperature=0.1)
            row, col = action // 9, action % 9
            board.make_move(row, col)
            print(f"AI落子位置: {row},{col}")
            time.sleep(1)  # 暂停以便观察
            
    print("\n最终棋盘:")
    board.display(show_coords=True)
    if board.winner == -1:
        print("\n恭喜您获胜了!")
    else:
        print("\nAI获胜!")
    print("游戏结束")

def main():
    parser = argparse.ArgumentParser(description="五子棋AI系统")
    parser.add_argument('--train', action='store_true', help="训练模型")
    parser.add_argument('--play', action='store_true', help="与AI对战")
    parser.add_argument('--model', type=str, default="gomoku_model_final.pth", 
                       help="模型权重文件路径")
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.play:
        play_human_vs_ai(args.model)
    else:
        print("请指定 --train 或 --play 参数")

if __name__ == "__main__":
    main()
