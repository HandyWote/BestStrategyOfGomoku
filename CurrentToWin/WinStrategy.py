import json
import numpy as np
import tkinter as tk
from tkinter import Label
import WinningPart

class WinStrategy:
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=int)
        self.player = 1  # 1:黑棋, -1:白棋
        self.WINNING_PATTERNS = WinningPart.WINNING_PATTERNS
        self.THREAT_PATTERNS = {
            1: {"冲四": ["011110"], "活四": ["011112", "211110"], "活三": ["011100", "001110"]},
            -1: {"反四": ["211112"], "反三": ["211100", "001112"], "禁手": ["11112", "21111"]}
        }

    def load_board(self, board_json):
        data = json.loads(board_json)
        self.board = np.array(data["board"])
        self._update_player()

    def _update_player(self):
        """更新当前玩家"""
        current_sum = np.sum(self.board)
        if current_sum == 0:
            self.player = 1
        elif current_sum == 1:
            self.player = -1

    def generate_move(self):
        # 优先处理对方必胜局面
        opponent_threats = self._scan_threats(-self.player)
        if opponent_threats:
            move = max(opponent_threats, key=lambda x: self._threat_priority(x[1]))[0]
            valid_move = self._get_valid_move(move)
            if valid_move:
                return self._convert_coords(valid_move), f"防御对方[{opponent_threats[0][1]}]"
            else:
                return self._get_any_valid_move(), f"防御对方[{opponent_threats[0][1]}]"

        # 匹配阵法
        pattern_move, reason = self.match_pattern()
        if pattern_move:
            valid_move = self._get_valid_move(pattern_move)
            if valid_move:
                return pattern_move, reason
            else:
                return self._get_any_valid_move(), reason

        # 黑棋首步天元
        if np.sum(np.abs(self.board)) == 0 and self.player == 1:
            return "5,5", "黑棋首步天元定式"

        # 主动进攻
        player_threats = self._scan_threats(self.player)
        if player_threats:
            move = max(player_threats, key=lambda x: self._threat_priority(x[1]))[0]
            valid_move = self._get_valid_move(move)
            if valid_move:
                return self._convert_coords(valid_move), f"发动[{player_threats[0][1]}]进攻"
            else:
                return self._get_any_valid_move(), f"发动[{player_threats[0][1]}]进攻"

        # 围绕有效棋点扩展
        occupied = np.argwhere(self.board != 0)
        if occupied.size > 0:
            r_center, c_center = np.mean(occupied, axis=0).astype(int)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = r_center + dr, c_center + dc
                    if 0 <= r < 9 and 0 <= c < 9 and self.board[r][c] == 0:
                        return self._convert_coords((r, c)), "围绕有效棋点扩展"

        # 中心扩展
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = 4 + dr, 4 + dc
                if 0 <= r < 9 and 0 <= c < 9 and self.board[r][c] == 0:
                    return self._convert_coords((r, c)), "扩展中心势力"

        # 随机空位
        empty = np.argwhere(self.board == 0)
        if empty.size > 0:
            r, c = empty[0]
            return self._convert_coords((r, c)), "随机空位保底"

        return None, "棋盘已满"

    def _get_valid_move(self, move):
        """验证落子位置是否有效"""
        if isinstance(move, str):
            x, y = move.split(',')
            r, c = 9 - int(y), int(x) - 1
        elif isinstance(move, tuple):
            r, c = move
        else:
            return None

        return (r, c) if 0 <= r < 9 and 0 <= c < 9 and self.board[r][c] == 0 else None

    def _get_any_valid_move(self):
        """获取任意有效落子位置"""
        empty = np.argwhere(self.board == 0)
        if empty.size > 0:
            r, c = empty[0]
            return r, c
        return None

    def match_pattern(self):
        current_sum = np.sum(self.board)
        move_count = np.count_nonzero(self.board)

        for pattern_name, pattern_data in self.WINNING_PATTERNS.items():
            if pattern_data["player"] != self.player:
                continue

            for stage in pattern_data["stages"]:
                if self._is_board_matching(stage["board"]):
                    move = self._convert_stage_move(stage["next_move"])
                    valid_move = self._get_valid_move(move)
                    if valid_move:
                        return move, f"匹配阵法：{pattern_name} - {stage['reason']}"
        return None, None

    def _is_board_matching(self, pattern_board):
        # 检查原始棋盘
        if np.array_equal(self.board, pattern_board):
            return True

        # 检查旋转和镜像
        transformations = [self.board, np.rot90(self.board), np.rot90(self.board, 2),
                           np.rot90(self.board, 3), np.fliplr(self.board), np.flipud(self.board)]
        for transformed in transformations:
            if np.array_equal(transformed, pattern_board):
                return True
        return False

    @staticmethod
    def _convert_stage_move(stage_move):
        """转换阵法中的坐标到数学坐标"""
        r, c = stage_move
        return f"{c+1},{9-r}"

    @staticmethod
    def _convert_coords(coords):
        """将数组坐标转换为数学坐标"""
        r, c = coords
        return f"{c+1},{9-r}"

    def _scan_threats(self, player):
        """威胁检测"""
        threats = []
        rotations = {"0": self.board, "90": np.rot90(self.board), "180": np.rot90(self.board, 2),
                     "270": np.rot90(self.board, 3)}
        for rot_name, board in rotations.items():
            for r in range(9):
                line = "".join(["1" if x == player else "2" if x == -player else "0" for x in board[r]])
                for t_type, patterns in self.THREAT_PATTERNS[1 if player == 1 else -1].items():
                    for p in patterns:
                        if p in line:
                            for c in range(9):
                                if board[r][c] == 0:
                                    if rot_name == "0":
                                        orig_r, orig_c = r, c
                                    elif rot_name == "90":
                                        orig_r, orig_c = c, 8 - r
                                    elif rot_name == "180":
                                        orig_r, orig_c = 8 - r, 8 - c
                                    else:
                                        orig_r, orig_c = 8 - c, r
                                    threats.append(((orig_r, orig_c), t_type))
                                    break
        return threats

    @staticmethod
    def _threat_priority(t_type):
        """威胁优先级"""
        priority_map = {"活四": 5, "冲四": 4, "反四": 5, "活三": 3, "反三": 3, "禁手": 2}
        return priority_map.get(t_type, 0)