import numpy as np
import json
import tkinter as tk
from tkinter import Label

from CurrentToWin import WinningPart


class WinStrategy:
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=int)
        self.player = 1  # 1:黑棋, -1:白棋
        self.WINNING_PATTERNS = WinningPart # 这里假设为空，实际使用时需要填充
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

    def _enhanced_threat_priority(self, threat_type, pattern, line_str):
        """增强版威胁优先级计算，考虑更多因素"""
        base_priority = {
            "活四": 100,
            "冲四": 80,
            "活三": 60,
            "眠三": 40,
            "活二": 20,
            "禁手": 70
        }.get(threat_type, 0)

        # 计算开放端数量（决定棋型潜力）
        open_ends = 0
        if '0' + pattern + '0' in line_str:
            open_ends = 2  # 两端开放
        elif ('0' + pattern in line_str) or (pattern + '0' in line_str):
            open_ends = 1  # 一端开放

        # 开放端越多，优先级越高
        priority = base_priority + open_ends * 10

        # 对特殊棋型增加额外权重
        if threat_type == "活三" and open_ends >= 1:
            priority += 15  # 活三是非常危险的棋型

        return priority

    def generate_move(self):
        """改进版决策逻辑，增加前瞻性分析"""
        # 特殊情况：棋盘为空，黑棋首步天元
        if np.sum(np.abs(self.board)) == 0 and self.player == 1:
            return "5,5", "黑棋首步天元定式"

        # 提前检查数学坐标(6,7)的战略价值
        math_x, math_y = 6, 7
        arr_r, arr_c = 9 - math_y, math_x - 1

        # 检查(6,7)是否为空且能形成威胁
        if 0 <= arr_r < 9 and 0 <= arr_c < 9 and self.board[arr_r][arr_c] == 0:
            # 临时落子评估
            self.board[arr_r][arr_c] = self.player
            player_threats = self._scan_threats(self.player)
            self.board[arr_r][arr_c] = 0  # 恢复棋盘

            # 检查对手在此处落子的威胁
            self.board[arr_r][arr_c] = -self.player
            opponent_threats = self._scan_threats(-self.player)
            self.board[arr_r][arr_c] = 0  # 恢复棋盘

            # 如果在此落子能形成必胜局面，立即返回
            if player_threats and player_threats[0][2] >= 100:
                return f"{math_x},{math_y}", f"主动[{player_threats[0][1]}]必胜进攻"

            # 如果对手在此处有必胜局面，必须防守
            if opponent_threats and opponent_threats[0][2] >= 100:
                return f"{math_x},{math_y}", f"防御对方[{opponent_threats[0][1]}]必胜局面"

            # 如果(6,7)是活三且优先级高，优先选择
            if any(threat[1] == "活三" and threat[2] >= 75 for threat in player_threats):
                return f"{math_x},{math_y}", f"主动[{player_threats[0][1]}]关键进攻"

            # 如果对手在(6,7)形成活三，必须防守
            if any(threat[1] == "活三" and threat[2] >= 75 for threat in opponent_threats):
                return f"{math_x},{math_y}", f"防御对方[{opponent_threats[0][1]}]关键威胁"

        # 正常逻辑：先处理对手威胁
        opponent_threats = self._scan_threats(-self.player)
        if opponent_threats:
            move, threat_type, priority = opponent_threats[0]
            if priority >= 60:  # 只处理高优先级威胁
                return self._convert_coords(move), f"防御对方[{threat_type}]"

        # 再尝试主动进攻
        player_threats = self._scan_threats(self.player)
        if player_threats:
            move, threat_type, priority = player_threats[0]
            if priority >= 60:  # 只考虑高优先级进攻
                return self._convert_coords(move), f"发动[{threat_type}]进攻"

        # 没有明显威胁，采用策略性落子
        # 1. 优先在已有点周围落子
        occupied = np.argwhere(self.board != 0)
        if occupied.size > 0:
            center_r, center_c = np.mean(occupied, axis=0).astype(int)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = center_r + dr, center_c + dc
                    if 0 <= r < 9 and 0 <= c < 9 and self.board[r][c] == 0:
                        # 检查在此处落子是否能形成低级威胁
                        self.board[r][c] = self.player
                        local_threats = self._scan_threats(self.player)
                        self.board[r][c] = 0

                        if local_threats and local_threats[0][2] >= 40:
                            return self._convert_coords((r, c)), f"构建[{local_threats[0][1]}]趋势"
                        else:
                            return self._convert_coords((r, c)), "围绕中心扩展"

        # 2. 占据中心区域
        for r in [3, 4, 5]:
            for c in [3, 4, 5]:
                if self.board[r][c] == 0:
                    return self._convert_coords((r, c)), "占据中心区域"

        # 3. 随机找个空位
        empty = np.argwhere(self.board == 0)
        if empty.size > 0:
            r, c = empty[0]
            return self._convert_coords((r, c)), "随机落子"

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
        """增强版威胁检测，覆盖更多棋型和关键位置"""
        threats = []
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、对角线、反对角线

        # 扩展威胁模式库，增加更多关键棋型
        EXTENDED_THREAT_PATTERNS = {
            1: {  # 黑棋威胁
                "活四": ["011110"],  # 活四
                "冲四": ["011112", "211110", "11110", "01111"],  # 冲四
                "活三": ["011100", "001110", "010110", "011010"],  # 活三
                "眠三": ["211100", "001112", "210110", "010112"],  # 眠三
                "活二": ["011000", "001100", "000110"],  # 活二
            },
            -1: {  # 白棋威胁（考虑禁手）
                "活四": ["011110"],
                "冲四": ["011112", "211110", "11110", "01111"],
                "活三": ["011100", "001110", "010110", "011010"],
                "禁手": ["11112", "21111", "11011", "10111", "11101"]  # 白棋需要防守的禁手点
            }
        }

        for r in range(9):
            for c in range(9):
                if self.board[r][c] != 0:
                    continue

                for dr, dc in directions:
                    line = []
                    for i in range(-4, 5):  # 检查以(r,c)为中心的9个格子
                        nr, nc = r + i * dr, c + i * dc
                        if 0 <= nr < 9 and 0 <= nc < 9:
                            line.append(self.board[nr][nc])
                        else:
                            line.append(None)  # 边界外标记为None

                    # 转换为字符串模式
                    line_str = ""
                    for cell in line:
                        if cell is None:
                            line_str += "x"  # 边界外
                        elif cell == player:
                            line_str += "1"  # 当前玩家棋子
                        elif cell == -player:
                            line_str += "2"  # 对手棋子
                        else:
                            line_str += "0"  # 空位

                    # 检查所有威胁模式
                    for threat_type, patterns in EXTENDED_THREAT_PATTERNS[1 if player == 1 else -1].items():
                        for pattern in patterns:
                            if pattern in line_str:
                                # 计算威胁等级
                                priority = self._enhanced_threat_priority(threat_type, pattern, line_str)

                                # 特殊处理关键位置(6,7) 对应数组坐标(2,5)
                                if (r, c) == (2, 5):
                                    priority += 5  # 大幅提升优先级

                                # 特殊处理天元位置(5,5)
                                if (r, c) == (4, 4):
                                    priority += 2

                                threats.append(((r, c), threat_type, priority))
                                break  # 每个位置每个方向只记录最高威胁

        # 按优先级排序，优先级相同则按坐标顺序
        threats.sort(key=lambda x: (-x[2], x[0][0], x[0][1]))
        return threats

    @staticmethod
    def _threat_priority(t_type):
        """威胁优先级"""
        priority_map = {"活四": 5, "冲四": 4, "反四": 5, "活三": 3, "反三": 3, "禁手": 2}
        return priority_map.get(t_type, 0)