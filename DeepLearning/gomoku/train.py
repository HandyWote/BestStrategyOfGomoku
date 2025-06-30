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

class Trainer:
    """Training pipeline for Gomoku AI"""
    
    def __init__(self, model: GomokuNet, lr: float = 0.001, batch_size: int = 256):
        self.model = model
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.mcts = MCTS(model)
        self.replay_buffer = deque(maxlen=50000)  # Store 50,000 games
        self.batch_size = batch_size
        
    def generate_self_play_games(self, model_state_dict, num_games):
        import torch
        from model import GomokuNet
        from mcts import MCTS
        from board import GomokuBoard
        import numpy as np
        model = GomokuNet(device=torch.device("cpu"))
        model.load_state_dict(model_state_dict)
        mcts = MCTS(model)
        games = []
        for _ in range(num_games):
            board = GomokuBoard()
            game_history = []
            while not board.winner:
                action_probs = mcts.search(board)
                game_history.append({
                    'state': board.get_state(),
                    'policy': action_probs,
                    'player': board.current_player
                })
                action = mcts.get_move(board, temperature=1.0)
                row, col = action // 9, action % 9
                board.make_move(row, col)
            for sample in game_history:
                sample['value'] = 1 if board.winner == sample['player'] else -1
            games.append(game_history)
        return games

    def self_play(self, num_games: int = 100) -> None:
        print(f"\n开始生成{num_games}局自对弈数据...")

        num_workers = 4
        games_per_worker = num_games // num_workers
        remainder = num_games % num_workers
        tasks = [games_per_worker] * num_workers
        for i in range(remainder):
            tasks[i] += 1

        # 保证参数在CPU
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        all_games = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.generate_self_play_games, model_state_dict, n) for n in tasks]
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
        
    def evaluate(self, num_games: int = 100) -> float:
        """Evaluate current model against previous version"""
        # For simplicity, we'll just evaluate against random moves
        wins = 0
        for _ in tqdm(range(num_games), desc="评估进度"):
            board = GomokuBoard()
            while not board.winner:
                if board.current_player == 1:  # AI's turn
                    action = self.mcts.get_move(board, temperature=0.1)
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
        
    def train_iteration(self, num_self_play: int = 30, num_train_steps: int = 300) -> Dict[str, float]:
        """完成一次完整的训练迭代(自对弈+训练)"""
        # 生成自对弈数据
        self.self_play(num_self_play)
        
        print("\n开始训练模型...")
        metrics = {'loss': [], 'policy_loss': [], 'value_loss': []}
        
        # 使用进度条显示训练过程
        with tqdm(range(num_train_steps), desc="训练进度") as pbar:
            for step in pbar:
                step_metrics = self.train_step()
                for k, v in step_metrics.items():
                    metrics[k].append(v)
                
                # 每100步显示一次训练指标
                if step % 100 == 0:
                    pbar.set_postfix({
                        'loss': f"{np.mean(metrics['loss'][-100:]):.4f}",
                        'policy_loss': f"{np.mean(metrics['policy_loss'][-100:]):.4f}",
                        'value_loss': f"{np.mean(metrics['value_loss'][-100:]):.4f}"
                    })
                    
                    # 显示一个示例棋盘
                    sample = random.choice(self.replay_buffer)
                    board = GomokuBoard()
                    board.board = np.array(sample['state']['board'])
                    board.current_player = sample['state']['current_player']
                    print("\n示例棋盘状态:")
                    board.display()
                    time.sleep(1)
        
        # 评估模型
        print("\n评估模型表现...")
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        win_rate = self.evaluate()
        avg_metrics['win_rate'] = win_rate
        
        print(f"\n训练完成! 平均损失: {avg_metrics['loss']:.4f}")
        print(f"策略损失: {avg_metrics['policy_loss']:.4f}")
        print(f"价值损失: {avg_metrics['value_loss']:.4f}")
        print(f"对随机玩家的胜率: {win_rate:.2%}")
        
        return avg_metrics
