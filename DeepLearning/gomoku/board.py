import numpy as np
from typing import List, Tuple, Optional

class GomokuBoard:
    """9x9 Gomoku board implementation"""
    
    def __init__(self):
        """Initialize empty 9x9 board"""
        self.board = np.zeros((9, 9), dtype=int)
        self.current_player = 1  # 1 for black, -1 for white
        self.last_move = None
        self.winner = None
        
    def reset(self) -> None:
        """Reset the board to initial state"""
        self.board.fill(0)
        self.current_player = 1
        self.last_move = None
        self.winner = None
        
    def make_move(self, row: int, col: int) -> bool:
        """
        Make a move on the board
        Returns True if move was valid and made, False otherwise
        """
        if self.winner is not None:
            return False  # Game already over
            
        if not (0 <= row < 9 and 0 <= col < 9):
            return False  # Out of bounds
            
        if self.board[row, col] != 0:
            return False  # Position already taken
            
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        
        # Check for win
        if self._check_win(row, col):
            self.winner = self.current_player
        else:
            self.current_player *= -1  # Switch player
            
        return True
        
    def _check_win(self, row: int, col: int) -> bool:
        """Check if last move caused a win (5 in a row)"""
        directions = [
            (0, 1),   # horizontal
            (1, 0),    # vertical
            (1, 1),    # diagonal down-right
            (1, -1)    # diagonal down-left
        ]
        
        player = self.board[row, col]
        
        for dr, dc in directions:
            count = 1  # current stone
            
            # Check in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < 9 and 0 <= c < 9 and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
                
            # Check in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < 9 and 0 <= c < 9 and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
                
            if count >= 5:
                return True
                
        return False
        
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get list of all valid moves as (row, col) tuples"""
        if self.winner is not None:
            return []
            
        return [(r, c) for r in range(9) for c in range(9) if self.board[r, c] == 0]
        
    def get_state(self) -> dict:
        """Serialize board state to dictionary"""
        return {
            'board': self.board.tolist(),
            'current_player': self.current_player,
            'last_move': self.last_move,
            'winner': self.winner
        }
        
    def display(self, show_coords: bool = True) -> None:
        """打印棋盘到控制台"""
        symbols = {0: '·', 1: '●', -1: '○'}
        
        if show_coords:
            print('  ' + ' '.join(str(i) for i in range(9)))
        
        for r in range(9):
            row_str = []
            for c in range(9):
                if self.last_move == (r, c):
                    row_str.append(f'[{symbols[self.board[r, c]]}]')
                else:
                    row_str.append(f' {symbols[self.board[r, c]]} ')
            
            if show_coords:
                print(f'{r} ' + ''.join(row_str))
            else:
                print(' ' + ''.join(row_str))
                
        if self.winner:
            print(f"\n游戏结束! 获胜方: {'黑棋' if self.winner == 1 else '白棋'}")
