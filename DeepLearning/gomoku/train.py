import numpy as np
import torch
import torch.optim as optim
from collections import deque
import random
from board import GomokuBoard
from model import GomokuNet
from mcts import MCTS
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict
import os
import shutil

class Trainer:
    """Training pipeline for Gomoku AI"""
    
    def __init__(self, model: GomokuNet, lr: float = 0.001, batch_size: int = 256, opponent_model_path: str = None, model_dir: str = "models"):
        self.model = model
        # 自动检测GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[INFO] 检测到GPU，使用GPU进行训练。")
        else:
            self.device = torch.device("cpu")
            print("[INFO] 未检测到GPU，使用CPU进行训练。")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.mcts = MCTS(model)
        self.replay_buffer = deque(maxlen=50000)  # Store 50,000 games
        self.batch_size = batch_size
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        # 新增：对手模型
        self.opponent_model = None
        if opponent_model_path and os.path.exists(opponent_model_path):
            self.opponent_model = GomokuNet(device=self.device)
            self.opponent_model.load_state_dict(torch.load(opponent_model_path, map_location=self.device))
            self.opponent_model.to(self.device)
            print(f"[INFO] 加载上一代模型作为对手: {opponent_model_path}")
        else:
            print("[INFO] 未找到上一代模型，对手暂用当前模型。")
            self.opponent_model = self.model
        # 新增：记录最优胜率
        self.best_win_rate = 0.0
        self.best_model_path = os.path.join(self.model_dir, "best_model.pth")

    def update_opponent(self, model_path: str):
        self.opponent_model = GomokuNet(device=self.device)
        self.opponent_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.opponent_model.to(self.device)
        print(f"[INFO] 更新对手模型: {model_path}")

    def get_best_model_path(self):
        return self.best_model_path

    def generate_self_play_games(self, model_state_dict, opponent_state_dict, num_games):
        import torch
        from model import GomokuNet
        from mcts import MCTS
        from board import GomokuBoard
        import numpy as np
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = GomokuNet(device=device)
        model.load_state_dict(model_state_dict)
        mcts = MCTS(model)
        opponent_model = GomokuNet(device=device)
        opponent_model.load_state_dict(opponent_state_dict)
        opponent_mcts = MCTS(opponent_model)
        games = []
        for game_idx in range(num_games):
            board = GomokuBoard()
            game_history = []
            step = 0
            # 轮流执黑白
            if game_idx % 2 == 0:
                black_mcts = mcts
                white_mcts = opponent_mcts
            else:
                black_mcts = opponent_mcts
                white_mcts = mcts
            while not board.winner:
                if board.current_player == 1:
                    action = black_mcts.get_move(board, temperature=1.0)
                else:
                    action = white_mcts.get_move(board, temperature=1.0)
                row, col = action // 9, action % 9
                action_probs = (black_mcts if board.current_player == 1 else white_mcts).search(board)
                game_history.append({
                    'state': board.get_state(),
                    'policy': action_probs,
                    'player': board.current_player
                })
                board.make_move(row, col)
                step += 1
            for sample in game_history:
                sample['value'] = 1 if board.winner == sample['player'] else -1
            games.append(game_history)
        return games

    def self_play(self, num_games: int = 100) -> None:
        print(f"\n开始生成{num_games}局自对弈数据...")
        num_workers = 8  # 可根据显卡和CPU调整
        games_per_worker = num_games // num_workers
        remainder = num_games % num_workers
        tasks = [games_per_worker] * num_workers
        for i in range(remainder):
            tasks[i] += 1
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        opponent_state_dict = {k: v.cpu() for k, v in self.opponent_model.state_dict().items()}
        all_games = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.generate_self_play_games, model_state_dict, opponent_state_dict, n) for n in tasks]
            with tqdm(total=num_games, desc="自对弈进度") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    all_games.extend(result)
                    pbar.update(len(result))
        for game_idx, game_history in enumerate(all_games):
            print(f"\n对局 {game_idx + 1}, 步数 {len(game_history)}:")
            board = GomokuBoard()
            for move in game_history:
                board.board = np.array(move['state']['board'])
                board.current_player = move['state']['current_player']
            board.display()
            for sample in game_history:
                self.replay_buffer.append(sample)

    def train_step(self) -> Dict[str, float]:
        """Perform one training step on a batch of samples"""
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0, 'policy_loss': 0, 'value_loss': 0}
            
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare inputs
        states = []
        policy_targets = []
        value_targets = []
        
        for sample in batch:
            state = sample['state']
            # Normalize board for current player
            board = np.array(state['board'])
            if state['current_player'] == -1:
                board *= -1
            states.append(board)
            
            # Create policy target vector
            policy = np.zeros(81)
            for action, prob in sample['policy'].items():
                policy[action] = prob
            policy_targets.append(policy)
            
            value_targets.append(sample['value'])
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).unsqueeze(1).to(self.device)  # Add channel dim
        policy_targets = torch.FloatTensor(np.array(policy_targets)).to(self.device)
        value_targets = torch.FloatTensor(np.array(value_targets)).unsqueeze(1).to(self.device)
        
        # Forward pass
        policy_logits, values = self.model(states)
        
        # Calculate losses
        policy_loss = -torch.mean(torch.sum(policy_targets * policy_logits, dim=1))
        value_loss = torch.mean((values - value_targets) ** 2)
        total_loss = policy_loss + value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
        
    def evaluate(self, num_games: int, mcts_simulations: int) -> float:
        """通用评估函数，可调对局数和MCTS模拟次数"""
        eval_mcts = MCTS(self.model, num_simulations=mcts_simulations)
        wins = 0
        for _ in tqdm(range(num_games), desc=f"评估进度({num_games}局, MCTS={mcts_simulations})"):
            board = GomokuBoard()
            while not board.winner:
                if board.current_player == 1:  # AI's turn
                    action = eval_mcts.get_move(board, temperature=0.1)
                    row, col = action // 9, action % 9
                else:  # Random opponent
                    valid_moves = board.get_valid_moves()
                    if not valid_moves:
                        break
                    row, col = random.choice(valid_moves)
                board.make_move(row, col)
            if board.winner == 1:
                wins += 1
        return wins / num_games

    def evaluate_fast(self) -> float:
        """快评估：10局，每步MCTS 20次，适合训练中快速监控"""
        return self.evaluate(num_games=10, mcts_simulations=20)

    def evaluate_worker(self, model_state_dict, num_games, mcts_simulations, device_str):
        import torch
        from model import GomokuNet
        from mcts import MCTS
        from board import GomokuBoard
        import numpy as np
        device = torch.device(device_str)
        model = GomokuNet(device=device)
        model.load_state_dict(model_state_dict)
        eval_mcts = MCTS(model, num_simulations=mcts_simulations)
        wins = 0
        for _ in range(num_games):
            board = GomokuBoard()
            while not board.winner:
                if board.current_player == 1:  # AI's turn
                    action = eval_mcts.get_move(board, temperature=0.1)
                    row, col = action // 9, action % 9
                else:  # Random opponent
                    valid_moves = board.get_valid_moves()
                    if not valid_moves:
                        break
                    row, col = random.choice(valid_moves)
                board.make_move(row, col)
            if board.winner == 1:
                wins += 1
        return wins

    def evaluate_parallel(self, num_games=100, mcts_simulations=200, num_workers=4):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] 评估进程将使用设备: {device_str}")
        games_per_worker = num_games // num_workers
        remainder = num_games % num_workers
        tasks = [games_per_worker] * num_workers
        for i in range(remainder):
            tasks[i] += 1
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        total_wins = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.evaluate_worker, model_state_dict, n, mcts_simulations, device_str) for n in tasks]
            for future in as_completed(futures):
                total_wins += future.result()
        return total_wins / num_games

    def evaluate_full(self) -> float:
        """全量评估：100局，每步MCTS 200次，适合最终模型实力评测（多进程加速）"""
        print("[INFO] 使用多进程并发评估...")
        return self.evaluate_parallel(num_games=100, mcts_simulations=200, num_workers=4)

    def train_iteration(self, num_self_play: int = 30, num_train_steps: int = 300) -> Dict[str, float]:
        """完成一次完整的训练迭代(自对弈+训练)"""
        # 生成自对弈数据
        self.self_play(num_self_play)
        print("\n开始训练模型...")
        metrics = {'loss': [], 'policy_loss': [], 'value_loss': []}
        with tqdm(range(num_train_steps), desc="训练进度") as pbar:
            for step in pbar:
                step_metrics = self.train_step()
                for k, v in step_metrics.items():
                    metrics[k].append(v)
                if step % 100 == 0:
                    pbar.set_postfix({
                        'loss': f"{np.mean(metrics['loss'][-100:]):.4f}",
                        'policy_loss': f"{np.mean(metrics['policy_loss'][-100:]):.4f}",
                        'value_loss': f"{np.mean(metrics['value_loss'][-100:]):.4f}"
                    })
                    sample = random.choice(self.replay_buffer)
                    board = GomokuBoard()
                    board.board = np.array(sample['state']['board'])
                    board.current_player = sample['state']['current_player']
                    print("\n示例棋盘状态:")
                    board.display()
        # 全量评估
        print("\n全量评估模型表现...")
        win_rate = self.evaluate_full()
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        avg_metrics['win_rate'] = win_rate
        print(f"\n训练完成! 平均损失: {avg_metrics['loss']:.4f}")
        print(f"策略损失: {avg_metrics['policy_loss']:.4f}")
        print(f"价值损失: {avg_metrics['value_loss']:.4f}")
        print(f"全量评估胜率: {win_rate:.2%}")
        # 保存模型，命名带战斗力（保留两位小数）
        power_str = f"{win_rate:.2f}"
        model_path = os.path.join(self.model_dir, f"model_power_{power_str}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"[INFO] 已保存当前模型权重到 {model_path}")
        # 若胜率更高，更新best_model.pth
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            shutil.copy(model_path, self.best_model_path)
            print(f"[INFO] 当前模型为最优，已更新 {self.best_model_path}")
        # 更新对手模型为本轮最新
        self.update_opponent(self.best_model_path)
        return avg_metrics
