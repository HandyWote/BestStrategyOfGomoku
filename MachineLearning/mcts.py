import numpy as np
import math
import random
import time
from copy import deepcopy
from board import GomokuBoard

class MCTSNode:
    """蒙特卡洛树搜索节点"""
    
    def __init__(self, board_state, parent=None, move=None):
        self.board_state = deepcopy(board_state)  # 当前棋盘状态
        self.parent = parent  # 父节点
        self.move = move  # 到达此节点的移动 (x, y)
        self.children = []  # 子节点列表
        self.visits = 0  # 访问次数
        self.wins = 0  # 胜利次数
        self.untried_moves = board_state.get_valid_moves()  # 未尝试的移动
        
    def is_fully_expanded(self):
        """检查是否所有可能的移动都已经扩展"""
        return len(self.untried_moves) == 0
    
    def is_terminal(self):
        """检查是否为终端节点（游戏结束）"""
        game_over, _ = self.board_state.is_game_over()
        return game_over
    
    def ucb1_value(self, c=math.sqrt(2)):
        """计算UCB1值用于节点选择"""
        if self.visits == 0:
            return float('inf')  # 未访问的节点优先级最高
        
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
    
    def select_best_child(self, c=math.sqrt(2)):
        """使用UCB1选择最佳子节点"""
        return max(self.children, key=lambda child: child.ucb1_value(c))
    
    def add_child(self, move):
        """添加子节点"""
        # 创建新的棋盘状态
        new_board = deepcopy(self.board_state)
        new_board.make_move(move[0], move[1])
        
        # 创建子节点
        child = MCTSNode(new_board, parent=self, move=move)
        self.children.append(child)
        self.untried_moves.remove(move)
        return child
    
    def update(self, result):
        """反向传播更新节点统计"""
        self.visits += 1
        # result: 1表示当前玩家胜利，-1表示失败，0表示平局
        # 需要根据节点的玩家身份调整胜利计数
        if result == self.board_state.current_player:
            self.wins += 1
        elif result == 0:  # 平局
            self.wins += 0.5

class MCTS:
    """蒙特卡洛树搜索算法"""
    
    def __init__(self, time_limit=1.0, max_iterations=1000):
        self.time_limit = time_limit  # 搜索时间限制（秒）
        self.max_iterations = max_iterations  # 最大迭代次数
    
    def search(self, board_state):
        """执行MCTS搜索，返回最佳移动"""
        root = MCTSNode(board_state)
        
        start_time = time.time()
        iterations = 0
        
        # 主搜索循环
        while (time.time() - start_time < self.time_limit and 
               iterations < self.max_iterations):
            
            # 1. 选择阶段：从根节点开始选择到叶子节点
            node = self._select(root)
            
            # 2. 扩展阶段：如果不是终端节点，扩展一个子节点
            if not node.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            
            # 3. 模拟阶段：从当前节点进行随机模拟
            result = self._simulate(node)
            
            # 4. 反向传播阶段：更新路径上所有节点的统计
            self._backpropagate(node, result)
            
            iterations += 1
        
        # 选择访问次数最多的子节点作为最佳移动
        if not root.children:
            # 如果没有子节点，随机选择一个有效移动
            valid_moves = board_state.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else None
        
        best_child = max(root.children, key=lambda child: child.visits)
        print(f"MCTS搜索完成: {iterations}次迭代, {time.time() - start_time:.2f}秒")
        print(f"最佳移动: {best_child.move}, 访问次数: {best_child.visits}, 胜率: {best_child.wins/best_child.visits:.3f}")
        
        return best_child.move
    
    def _select(self, node):
        """选择阶段：使用UCB1选择路径直到叶子节点"""
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_best_child()
        return node
    
    def _expand(self, node):
        """扩展阶段：为节点添加一个新的子节点"""
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            return node.add_child(move)
        return node
    
    def _simulate(self, node):
        """模拟阶段：从当前节点进行随机对弈直到游戏结束"""
        # 复制当前棋盘状态进行模拟
        simulation_board = deepcopy(node.board_state)
        
        # 随机对弈直到游戏结束
        while True:
            game_over, winner = simulation_board.is_game_over()
            if game_over:
                return winner
            
            # 随机选择一个有效移动
            valid_moves = simulation_board.get_valid_moves()
            if not valid_moves:
                return 0  # 平局
            
            move = random.choice(valid_moves)
            simulation_board.make_move(move[0], move[1])
    
    def _backpropagate(self, node, result):
        """反向传播阶段：更新路径上所有节点的统计"""
        while node is not None:
            node.update(result)
            node = node.parent

# 测试函数
def test_mcts():
    """测试MCTS算法的基本功能"""
    print("=== MCTS算法测试 ===")
    
    # 创建测试棋盘
    board = GomokuBoard(size=9)
    
    # 下几步棋创建一个测试局面
    board.make_move(4, 4)  # 黑棋
    board.make_move(4, 5)  # 白棋
    board.make_move(3, 3)  # 黑棋
    
    print("当前棋盘状态:")
    board.display()
    print(f"当前玩家: {'黑棋' if board.current_player == 1 else '白棋'}")
    
    # 创建MCTS实例并搜索最佳移动
    mcts = MCTS(time_limit=2.0, max_iterations=1000)
    
    print("\n开始MCTS搜索...")
    best_move = mcts.search(board)
    
    if best_move:
        print(f"\n推荐移动: ({best_move[0]}, {best_move[1]})")
        
        # 执行推荐移动
        board.make_move(best_move[0], best_move[1])
        print("\n执行移动后的棋盘:")
        board.display()
    else:
        print("\n没有找到有效移动")

def test_ucb1_formula():
    """测试UCB1公式的计算"""
    print("\n=== UCB1公式测试 ===")
    
    # 创建测试节点
    board = GomokuBoard()
    parent = MCTSNode(board)
    parent.visits = 100
    
    # 创建几个子节点进行测试
    child1 = MCTSNode(board, parent=parent)
    child1.visits = 10
    child1.wins = 7  # 胜率70%
    
    child2 = MCTSNode(board, parent=parent)
    child2.visits = 5
    child2.wins = 2  # 胜率40%
    
    child3 = MCTSNode(board, parent=parent)
    child3.visits = 20
    child3.wins = 12  # 胜率60%
    
    print(f"子节点1: 访问{child1.visits}次, 胜率{child1.wins/child1.visits:.2f}, UCB1={child1.ucb1_value():.3f}")
    print(f"子节点2: 访问{child2.visits}次, 胜率{child2.wins/child2.visits:.2f}, UCB1={child2.ucb1_value():.3f}")
    print(f"子节点3: 访问{child3.visits}次, 胜率{child3.wins/child3.visits:.2f}, UCB1={child3.ucb1_value():.3f}")
    
    # 测试选择最佳子节点
    parent.children = [child1, child2, child3]
    best = parent.select_best_child()
    print(f"\n选择的最佳子节点: 访问{best.visits}次, 胜率{best.wins/best.visits:.2f}")

if __name__ == "__main__":
    # 运行测试
    test_ucb1_formula()
    test_mcts()