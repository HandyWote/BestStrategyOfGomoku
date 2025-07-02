import argparse
import torch
import time
import os
import multiprocessing as mp
import threading
mp.set_start_method('spawn', force=True)
from board import GomokuBoard
from model import GomokuNet
from mcts import MCTS
from train import Trainer

# 自动安装依赖
try:
    import psutil
except ImportError:
    os.system('pip install psutil')
    import psutil
try:
    import pynvml
except ImportError:
    os.system('pip install pynvml')
    import pynvml

def detect_resources():
    # CPU
    logical_cores = os.cpu_count()
    physical_cores = psutil.cpu_count(logical=False)
    # GPU
    gpu_info = []
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            gpu_info.append({
                'id': i,
                'name': name.decode(),
                'total_mem': mem.total // (1024*1024),
                'free_mem': mem.free // (1024*1024)
            })
    except Exception as e:
        print("[WARN] 未检测到GPU或未安装pynvml:", e)
    return logical_cores, physical_cores, gpu_info

def self_play_worker(replay_buffer, trainer_args, num_games_per_worker):
    # 持续生成自对弈棋局
    from train import Trainer
    import torch
    model = GomokuNet(device=torch.device("cpu"))
    trainer = Trainer(model, **trainer_args)
    while True:
        games = trainer.generate_self_play_games_minimax({k: v.cpu() for k, v in model.state_dict().items()}, num_games_per_worker)
        for game in games:
            replay_buffer.put(game)

def dynamic_adjust(self_play_workers, loss, win_rate, buffer_util, min_workers, max_workers):
    # 经验池利用率优先
    if buffer_util < 0.5 and self_play_workers < max_workers:
        self_play_workers += 1
        print("[动态调整] 经验池利用率低，增加自对弈进程数:", self_play_workers)
    elif buffer_util > 0.9 and self_play_workers > min_workers:
        self_play_workers -= 1
        print("[动态调整] 经验池利用率高，减少自对弈进程数:", self_play_workers)
    # 训练损失和胜率辅助微调
    elif loss > 1.0 and self_play_workers < max_workers:
        self_play_workers += 1
        print("[动态调整] 损失高，增加自对弈进程数:", self_play_workers)
    elif win_rate > 0.8 and self_play_workers > min_workers:
        self_play_workers -= 1
        print("[动态调整] 胜率高，减少自对弈进程数:", self_play_workers)
    return self_play_workers

def auto_batch_size(gpu_info, min_bs=64, max_bs=512):
    # 简单策略：按单卡显存分配batch size
    if not gpu_info:
        return min_bs
    mem_gb = gpu_info[0]['total_mem'] // 1024
    if mem_gb >= 16:
        return min(max_bs, 512)
    elif mem_gb >= 8:
        return min(max_bs, 256)
    else:
        return min_bs

def monitor_and_adjust_worker(shared_metrics, replay_buffer, self_play_procs, train_proc, cpu_workers, gpu_info, min_workers, max_workers):
    import time
    batch_size = shared_metrics['batch_size']
    while True:
        time.sleep(300)  # 每5分钟调整一次
        # 采集指标
        loss = shared_metrics.get('loss', 1.0)
        win_rate = shared_metrics.get('win_rate', 0.5)
        buffer_util = replay_buffer.qsize() / shared_metrics['buffer_maxsize']
        # 动态调整自对弈进程数
        new_workers = dynamic_adjust(len(self_play_procs), loss, win_rate, buffer_util, min_workers, max_workers)
        # 动态调整batch size
        new_batch_size = auto_batch_size(gpu_info)
        if new_batch_size != shared_metrics['batch_size']:
            shared_metrics['batch_size'] = new_batch_size
            print(f"[动态调整] 训练batch size调整为: {new_batch_size}")
        # 动态增减自对弈进程
        diff = new_workers - len(self_play_procs)
        if diff > 0:
            for _ in range(diff):
                p = mp.Process(target=self_play_worker, args=(replay_buffer, shared_metrics['trainer_args'], shared_metrics['num_games_per_worker']))
                p.start()
                self_play_procs.append(p)
        elif diff < 0:
            for _ in range(-diff):
                p = self_play_procs.pop()
                p.terminate()
                print("[动态调整] 终止一个自对弈进程")
        print(f"[监控] 当前自对弈进程数: {len(self_play_procs)}, batch size: {shared_metrics['batch_size']}, 经验池利用率: {buffer_util:.2f}, 损失: {loss:.4f}, 胜率: {win_rate:.2%}")

def train_worker(replay_buffer, trainer_args, shared_metrics):
    from train import Trainer
    import torch
    model = GomokuNet(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = Trainer(model, **trainer_args)
    while True:
        batch_size = shared_metrics['batch_size']
        if replay_buffer.qsize() >= batch_size:
            batch = [replay_buffer.get() for _ in range(batch_size)]
            metrics = trainer.train_step()  # 需适配batch输入
            shared_metrics['loss'] = metrics.get('loss', 1.0)
            shared_metrics['policy_loss'] = metrics.get('policy_loss', 0.0)
            shared_metrics['value_loss'] = metrics.get('value_loss', 0.0)
        else:
            time.sleep(1)

def evaluate_worker(trainer_args, shared_metrics, interval=600):
    from train import Trainer
    import torch
    model = GomokuNet(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = Trainer(model, **trainer_args)
    while True:
        time.sleep(interval)
        max_level, max_level_name, max_win_rate = trainer.save_by_smart_evaluate()
        shared_metrics['win_rate'] = max_win_rate

def async_train_main(model_path=None):
    logical_cores, physical_cores, gpu_info = detect_resources()
    print(f"[资源检测] CPU逻辑核心数: {logical_cores}, 物理核心数: {physical_cores}")
    if gpu_info:
        for g in gpu_info:
            print(f"[资源检测] GPU {g['id']}: {g['name']}, 总显存: {g['total_mem']}MB, 空闲: {g['free_mem']}MB")
    else:
        print("[资源检测] 未检测到GPU")
    cpu_workers = max(1, int(logical_cores * 0.7))
    min_workers = 1
    max_workers = max(2, int(logical_cores * 0.8))
    batch_size = auto_batch_size(gpu_info)
    print(f"[分配] 初始自对弈进程数: {cpu_workers}")
    print(f"[分配] 初始训练batch size: {batch_size}")
    buffer_maxsize = 50000
    mp_manager = mp.Manager()
    replay_buffer = mp_manager.Queue(maxsize=buffer_maxsize)
    shared_metrics = mp_manager.dict()
    trainer_args = {'opponent_model_path': "models/best_model.pth", 'model_dir': "models"}
    num_games_per_worker = 10
    shared_metrics['batch_size'] = batch_size
    shared_metrics['trainer_args'] = trainer_args
    shared_metrics['num_games_per_worker'] = num_games_per_worker
    shared_metrics['loss'] = 1.0
    shared_metrics['win_rate'] = 0.5
    shared_metrics['buffer_maxsize'] = buffer_maxsize
    # 启动自对弈进程
    self_play_procs = [mp.Process(target=self_play_worker, args=(replay_buffer, trainer_args, num_games_per_worker)) for _ in range(cpu_workers)]
    for p in self_play_procs:
        p.start()
    # 启动训练进程
    train_proc = mp.Process(target=train_worker, args=(replay_buffer, trainer_args, shared_metrics))
    train_proc.start()
    # 启动评估进程
    eval_proc = mp.Process(target=evaluate_worker, args=(trainer_args, shared_metrics))
    eval_proc.start()
    # 启动监控与动态调整线程
    monitor_thread = threading.Thread(target=monitor_and_adjust_worker, args=(shared_metrics, replay_buffer, self_play_procs, train_proc, cpu_workers, gpu_info, min_workers, max_workers), daemon=True)
    monitor_thread.start()
    for p in self_play_procs:
        p.join()
    train_proc.join()
    eval_proc.join()

def train_model(model_path=None):
    """训练五子棋AI模型"""
    if model_path is None:
        default_best = "models/best_model.pth"
        if os.path.exists(default_best):
            print(f"未指定模型，自动加载最优模型: {default_best}")
            model_path = default_best
        else:
            print("未指定模型，也未找到最优模型，将从头训练。")
            model_path = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GomokuNet(device=device)
    if model_path and os.path.exists(model_path):
        print(f"加载已有模型权重: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    trainer = Trainer(model, opponent_model_path="models/best_model.pth")
    os.makedirs("models", exist_ok=True)
    print("\n开始训练五子棋AI模型...")
    for iteration in range(1, 1001):  # 1000次迭代
        metrics = trainer.train_iteration()
        print(f"\n迭代 {iteration}:")
        print(f"  总损失: {metrics['loss']:.4f}")
        print(f"  策略损失: {metrics['policy_loss']:.4f}") 
        print(f"  价值损失: {metrics['value_loss']:.4f}")
        print(f"  胜率: {metrics['win_rate']:.2%}")
    print("\n训练完成!")

def play_human_vs_ai(model_path: str = "gomoku_model_final.pth"):
    """人机对战"""
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请检查路径！")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GomokuNet(device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    parser.add_argument('--model', type=str, default=None, help="模型权重文件路径")
    args = parser.parse_args()

    if args.train:
        async_train_main(args.model)
    elif args.play:
        play_human_vs_ai(args.model)
    else:
        print("请指定 --train 或 --play 参数")

if __name__ == "__main__":
    main()
