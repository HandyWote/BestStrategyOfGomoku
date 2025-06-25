import numpy as np
import json

class GomokuBoard:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0空, 1黑棋, -1白棋
        self.current_player = 1  # 黑棋先行
        
    def make_move(self, x, y):
        if self.board[x][y] != 0:
            return False  # 非法移动
        self.board[x][y] = self.current_player
        self.current_player = -self.current_player  # 切换玩家 (1 <-> -1)
        return True
    
    def check_win(self):
        # 检查四个方向的五连珠
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in directions:
            for x in range(self.size):
                for y in range(self.size):
                    if all(0 <= x+i*dx < self.size and 0 <= y+i*dy < self.size and 
                           self.board[x][y] != 0 and
                           self.board[x+i*dx][y+i*dy] == self.board[x][y]
                           for i in range(5)):
                        return self.board[x][y]  # 返回获胜玩家
        return 0  # 未分胜负
    
    def display(self):
        print("  " + " ".join(str(i) for i in range(self.size)))
        for i in range(self.size):
            print(f"{i} " + " ".join("●" if x == 1 else "○" if x == -1 else "·" for x in self.board[i]))

    def to_json(self):
        """将棋盘转换为项目标准的JSON格式"""
        return {"board": self.board.tolist()}
    
    def from_json(self, json_data):
        """从JSON格式加载棋盘状态"""
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        board_data = data.get("board", [])
        if len(board_data) == self.size and all(len(row) == self.size for row in board_data):
            self.board = np.array(board_data, dtype=int)
            # 根据棋盘状态计算当前玩家
            total_pieces = np.sum(np.abs(self.board))
            black_pieces = np.sum(self.board == 1)
            white_pieces = np.sum(self.board == -1)
            # 如果黑棋数量等于白棋数量，轮到黑棋；否则轮到白棋
            self.current_player = 1 if black_pieces == white_pieces else -1
        else:
            raise ValueError("Invalid board data format")
    
    def save_to_file(self, filename="board.json"):
        """保存棋盘状态到JSON文件"""
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f)
    
    def load_from_file(self, filename="board.json"):
        """从JSON文件加载棋盘状态"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.from_json(data)
        except FileNotFoundError:
            print(f"文件 {filename} 不存在，使用默认空棋盘")
    
    def is_valid_move(self, x, y):
        """检查指定位置是否可以落子"""
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return self.board[x][y] == 0
    
    def get_valid_moves(self):
        """获取所有有效的落子位置"""
        valid_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    valid_moves.append((i, j))
        return valid_moves
    
    def is_game_over(self):
        """检查游戏是否结束（有人获胜或棋盘满）"""
        winner = self.check_win()
        if winner != 0:
            return True, winner
        
        # 检查棋盘是否已满
        if len(self.get_valid_moves()) == 0:
            return True, 0  # 平局
        
        return False, 0

if __name__ == "__main__":
    # 测试基本功能
    board = GomokuBoard()
    board.make_move(4, 4)  # 黑棋下在中心
    board.make_move(3, 3)  # 白棋
    board.display()
    print("获胜玩家:", board.check_win())
    
    # 测试JSON功能
    print("\n测试JSON格式:")
    json_data = board.to_json()
    print("JSON格式:", json.dumps(json_data))
    
    # 测试从JSON加载
    new_board = GomokuBoard()
    new_board.from_json(json_data)
    print("\n从JSON加载的棋盘:")
    new_board.display()
    
    # 测试文件保存和加载
    board.save_to_file("test_board.json")
    test_board = GomokuBoard()
    test_board.load_from_file("test_board.json")
    print("\n从文件加载的棋盘:")
    test_board.display()