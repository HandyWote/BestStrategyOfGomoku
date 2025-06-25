import numpy as np

class GomokuBoard:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0空, 1黑棋, 2白棋
        self.current_player = 1  # 黑棋先行
        
    def make_move(self, x, y):
        if self.board[x][y] != 0:
            return False  # 非法移动
        self.board[x][y] = self.current_player
        self.current_player = 3 - self.current_player  # 切换玩家
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
            print(f"{i} " + " ".join("●" if x == 1 else "○" if x == 2 else "·" for x in self.board[i]))

if __name__ == "__main__":
    board = GomokuBoard()
    board.make_move(4, 4)  # 黑棋下在中心
    board.make_move(3, 3)  # 白棋
    board.display()
    print("获胜玩家:", board.check_win())