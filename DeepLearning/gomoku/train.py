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
import traceback

class RandomPlayer:
    def get_move(self, board):
        valid_moves = list(zip(*np.where(board == 0)))
        return random.choice(valid_moves)

class Trainer:
    """Training pipeline for Gomoku AI"""
    
    def __init__(self, model: GomokuNet, lr: float = 0.001, batch_size: int = 256, opponent_model_path: str = None, model_dir: str = "models", device=None):
        self.model = model
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def generate_self_play_games_minimax(self, model_state_dict, num_games, mcts_first=True):
        import os
        import traceback
        print(f"[自对弈进程] 进入generate_self_play_games_minimax, pid={os.getpid()}")
        try:
            from board import GomokuBoard
            import numpy as np
            print(f"[自对弈进程] 初始化MinimaxAI, pid={os.getpid()}")
            minimax_black = MinimaxAI(depth=2)
            minimax_white = MinimaxAI(depth=2)
            games = []
            print(f"[自对弈进程] 开始生成{num_games}局, pid={os.getpid()}")
            for game_idx in range(num_games):
                try:
                    board = GomokuBoard()
                    game_history = []
                    step = 0
                    max_steps = 81
                    while not board.winner and step < max_steps:
                        if board.current_player == 1:
                            row, col = minimax_black.get_move(board.board)
                        else:
                            row, col = minimax_white.get_move(board.board)
                        action = row * 9 + col
                        # 生成one-hot policy
                        policy = {action: 1.0}
                        game_history.append({
                            'state': board.get_state(),
                            'policy': policy,
                            'player': board.current_player
                        })
                        board.make_move(row, col)
                        step += 1
                    for sample in game_history:
                        sample['value'] = 1 if board.winner == sample['player'] else -1
                    games.append(game_history)
                    print(f"[自对弈进程] 完成第{game_idx+1}局, 步数{step}, pid={os.getpid()}")
                except Exception as e:
                    print(f"[自对弈进程][单局异常] pid={os.getpid()}，game_idx={game_idx}，异常信息: {e}")
                    traceback.print_exc()
            print(f"[自对弈进程] 退出generate_self_play_games_minimax, pid={os.getpid()}, 返回{len(games)}局")
            return games
        except Exception as e:
            print(f"[自对弈进程][函数异常] pid={os.getpid()}，异常信息: {e}")
            traceback.print_exc()
            return []

    def self_play_minimax(self, num_games: int = 100) -> None:
        print(f"\n开始生成{num_games}局与MinimaxAI自对弈数据...")
        num_workers = 4
        games_per_worker = num_games // num_workers
        remainder = num_games % num_workers
        tasks = [games_per_worker] * num_workers
        for i in range(remainder):
            tasks[i] += 1
        model_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        all_games = []
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.generate_self_play_games_minimax, model_state_dict, n, mcts_first=(i%2==0)) for i, n in enumerate(tasks)]
            with tqdm(total=num_games, desc="Minimax自对弈进度") as pbar:
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
        
    def train_iteration(self, num_self_play: int = 30, num_train_steps: int = 300) -> Dict[str, float]:
        """完成一次完整的训练迭代(全部与MinimaxAI对弈+训练)"""
        self.self_play_minimax(num_self_play)
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
        # 智能分级评估与模型保存
        print("\n智能分级评估与模型保存...")
        self.save_by_smart_evaluate()
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        print(f"\n训练完成! 平均损失: {avg_metrics['loss']:.4f}")
        print(f"策略损失: {avg_metrics['policy_loss']:.4f}")
        print(f"价值损失: {avg_metrics['value_loss']:.4f}")
        return avg_metrics

    def load_model_player(self, model_path):
        # 加载历史模型作为对手
        if not os.path.exists(model_path):
            print(f"[WARN] 历史模型文件不存在: {model_path}，跳过该对手。")
            return None
        model = GomokuNet(device=self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        class ModelPlayer:
            def get_move(self, board):
                mcts = MCTS(model, num_simulations=50)
                action = mcts.get_move(board, temperature=0.1)
                row, col = action // 9, action % 9
                return (row, col)
        return ModelPlayer()

    def play_one_game(self, model, opponent):
        # AI执黑，对手执白
        board = GomokuBoard()
        mcts = MCTS(model, num_simulations=50)
        while not board.winner:
            if board.current_player == 1:
                action = mcts.get_move(board, temperature=0.1)
                row, col = action // 9, action % 9
            else:
                row, col = opponent.get_move(board.board)
            board.make_move(row, col)
        return board.winner

    def smart_evaluate(self, games_per_level=30, win_threshold=0.6):
        """
        智能分级评估：AI依次挑战不同难度对手，胜率达标才晋级
        返回：(最高通过等级index, 等级名, 胜率, [(等级名, 胜率)])
        """
        levels = [
            ("随机玩家", RandomPlayer()),
            ("历史弱模型", self.load_model_player(os.path.join(self.model_dir, "model_power_0.00.pth"))),
            ("历史中等模型", self.load_model_player(os.path.join(self.model_dir, "model_power_0.50.pth"))),
            ("MinimaxAI-2", MinimaxAI(depth=2)),
            ("MinimaxAI-4", MinimaxAI(depth=4)),
            ("当前最优模型", self.load_model_player(self.best_model_path)),
        ]
        results = []
        max_level = -1
        max_level_name = ""
        max_win_rate = 0.0
        for idx, (level_name, opponent) in enumerate(levels):
            if opponent is None:
                print(f"[WARN] 跳过对手：{level_name}（模型文件不存在）")
                continue
            print(f"\n[评估] 挑战对手：{level_name}")
            wins = 0
            for _ in range(games_per_level):
                winner = self.play_one_game(self.model, opponent)
                if winner == 1:
                    wins += 1
            win_rate = wins / games_per_level
            print(f"胜率：{win_rate:.2%}")
            results.append((level_name, win_rate))
            if win_rate >= win_threshold:
                max_level = idx
                max_level_name = level_name
                max_win_rate = win_rate
            else:
                print(f"未通过{level_name}，评估终止。")
                break
        print("\n智能分级评估结果：")
        for level_name, win_rate in results:
            print(f"{level_name}：胜率 {win_rate:.2%}")
        return max_level, max_level_name, max_win_rate, results

    def save_by_smart_evaluate(self, games_per_level=30, win_threshold=0.6):
        """
        智能评估后，按等级和胜率命名模型，最高等级最高胜率命名为best_model.pth
        """
        max_level, max_level_name, max_win_rate, results = self.smart_evaluate(games_per_level, win_threshold)
        # 命名规则：model_level{等级index}_{胜率}.pth
        if max_level >= 0:
            model_name = f"model_level{max_level+1}_{max_win_rate:.2f}.pth"
            model_path = os.path.join(self.model_dir, model_name)
            torch.save(self.model.state_dict(), model_path)
            print(f"[INFO] 已保存当前模型权重到 {model_path}")
            # 若为最高等级且胜率最高，更新best_model.pth
            best_model_path = os.path.join(self.model_dir, "best_model.pth")
            shutil.copy(model_path, best_model_path)
            print(f"[INFO] 当前模型为最优，已更新 {best_model_path}")
        else:
            print("[WARN] 没有通过任何等级，未保存为best_model")
        # 也可保留每一级的评估结果
        for idx, (level_name, win_rate) in enumerate(results):
            level_model_name = f"model_level{idx+1}_{win_rate:.2f}.pth"
            level_model_path = os.path.join(self.model_dir, level_model_name)
            torch.save(self.model.state_dict(), level_model_path)
            print(f"[INFO] 备份：{level_model_path}")
        return max_level, max_level_name, max_win_rate

# MinimaxAI集成
class MinimaxAI:
    def __init__(self, depth=2):
        self.depth = depth
        self.BOARD_SIZE = 9
        self.EMPTY = 0
        self.BLACK = 1
        self.WHITE = -1
        self.pattern_scores = {
            "FIVE": 100000,
            "FOUR": 10000,
            "BLOCKED_FOUR": 1000,
            "THREE": 1000,
            "BLOCKED_THREE": 100,
            "TWO": 100,
            "BLOCKED_TWO": 10,
            "ONE": 10,
            "BLOCKED_ONE": 1
        }
    def get_move(self, board):
        _, move = self.minimax(board.copy(), self.depth, -float('inf'), float('inf'), True)
        return move
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_game_over(board):
            return self.evaluate(board), None
        if maximizing_player:
            max_eval = -float('inf')
            best_move = None
            for move in self.get_possible_moves(board):
                row, col = move
                board[row, col] = self.WHITE
                eval, _ = self.minimax(board, depth-1, alpha, beta, False)
                board[row, col] = self.EMPTY
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in self.get_possible_moves(board):
                row, col = move
                board[row, col] = self.BLACK
                eval, _ = self.minimax(board, depth-1, alpha, beta, True)
                board[row, col] = self.EMPTY
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move
    def get_possible_moves(self, board):
        empty_positions = list(zip(*np.where(board == self.EMPTY)))
        return empty_positions
    def is_game_over(self, board):
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if board[row, col] != self.EMPTY and self.check_win(board, row, col):
                    return True
        return np.all(board != self.EMPTY)
    def evaluate(self, board):
        score = 0
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if board[row, col] == self.WHITE:
                    score += self.evaluate_position(board, row, col, self.WHITE)
                elif board[row, col] == self.BLACK:
                    score -= self.evaluate_position(board, row, col, self.BLACK)
        return score
    def evaluate_position(self, board, row, col, player):
        directions = [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ]
        total_score = 0
        for axis in directions:
            line = [player]
            r, c = row + axis[0][0], col + axis[0][1]
            while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE:
                line.append(board[r, c])
                r += axis[0][0]
                c += axis[0][1]
            r, c = row + axis[1][0], col + axis[1][1]
            while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE:
                line.insert(0, board[r, c])
                r += axis[1][0]
                c += axis[1][1]
            total_score += self.analyze_line(line, player)
        return total_score
    def analyze_line(self, line, player):
        score = 0
        count = 0
        block = 0
        empty = 0
        for i in range(len(line)):
            if line[i] == player:
                count += 1
            elif line[i] == self.EMPTY:
                if count > 0:
                    empty += 1
                    block = 0
                else:
                    block = 0
                    empty = 1
            else:
                if count > 0:
                    block += 1
                    score += self.get_pattern_score(count, block, empty)
                    count = 0
                    block = 1
                    empty = 0
                else:
                    block = 1
                    empty = 0
        if count > 0:
            score += self.get_pattern_score(count, block, empty)
        return score
    def get_pattern_score(self, count, block, empty):
        if empty == 0:
            if count >= 5:
                return self.pattern_scores["FIVE"]
            if block == 0:
                if count == 4:
                    return self.pattern_scores["FOUR"]
                elif count == 3:
                    return self.pattern_scores["THREE"]
                elif count == 2:
                    return self.pattern_scores["TWO"]
                elif count == 1:
                    return self.pattern_scores["ONE"]
            elif block == 1:
                if count == 4:
                    return self.pattern_scores["BLOCKED_FOUR"]
                elif count == 3:
                    return self.pattern_scores["BLOCKED_THREE"]
                elif count == 2:
                    return self.pattern_scores["BLOCKED_TWO"]
                elif count == 1:
                    return self.pattern_scores["BLOCKED_ONE"]
        elif empty == 1 or empty == count - 1:
            if count >= 5:
                return self.pattern_scores["FIVE"]
            if block == 0:
                if count == 4:
                    return self.pattern_scores["FOUR"]
                elif count == 3:
                    return self.pattern_scores["THREE"]
                elif count == 2:
                    return self.pattern_scores["TWO"]
            elif block == 1:
                if count == 4:
                    return self.pattern_scores["BLOCKED_FOUR"]
                elif count == 3:
                    return self.pattern_scores["BLOCKED_THREE"]
                elif count == 2:
                    return self.pattern_scores["BLOCKED_TWO"]
        elif empty == 2 or empty == count - 2:
            if count >= 5:
                return self.pattern_scores["FIVE"]
            if block == 0:
                if count == 4:
                    return self.pattern_scores["FOUR"]
                elif count == 3:
                    return self.pattern_scores["THREE"]
            elif block == 1:
                if count == 4:
                    return self.pattern_scores["BLOCKED_FOUR"]
                elif count == 3:
                    return self.pattern_scores["BLOCKED_THREE"]
        return 0
    def check_win(self, board, row, col):
        directions = [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ]
        player = board[row, col]
        for axis in directions:
            count = 1
            for (dr, dc) in axis:
                r, c = row + dr, col + dc
                while 0 <= r < self.BOARD_SIZE and 0 <= c < self.BOARD_SIZE and board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
            if count >= 5:
                return True
        return False
