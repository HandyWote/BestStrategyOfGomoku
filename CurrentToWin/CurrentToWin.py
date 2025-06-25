import json
import os

import numpy as np

WINNING_PATTERNS = {
    # 花月阵法
    "flower_moon": [
        # 第一阶段：建立基础阵型
        {"board": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], "next_move": (5, 3)},

        # 第二阶段：扩展攻势
        {"board": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], "next_move": (3, 5)},

        # 第三阶段：形成杀局
        {"board": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ], "next_move": (4, 2)}
    ],
    # 浦月阵法
    "river_moon": [
        {"board": [
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
        ], "next_move": (5, 5)},

        {"board": [
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0],
            [0,0,0,0,-1,0,0,0,0],
            [0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
        ], "next_move": (3, 1)}
    ]
}

#方向向量：水平、垂直、对角线
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

def check_winning_pattern(board, current_player, WinningPartterns=None):
    """检查并应用必胜阵法"""
    if current_player != 1:  # 仅黑棋使用必胜阵法
        return None

    # 尝试所有必胜阵法
    for pattern_name, patterns in WinningPartterns.WINNING_PATTERNS.items():
        for pattern in patterns:
            match = True
            # 检查当前棋盘是否符合阵法状态
            for i in range(9):
                for j in range(9):
                    if pattern["board"][i][j] != 0:  # 只检查阵法指定位置
                        if board[i][j] != pattern["board"][i][j]:
                            match = False
                            break
                if not match:
                    break

            # 如果匹配且落子点为空
            if match:
                move_x, move_y = pattern["next_move"]
                if board[move_x][move_y] == 0:
                    return (move_x, move_y)

    return None
def is_win(board, x, y, player):
    """检查(x,y)落子后是否五连获胜"""
    for dx, dy in DIRECTIONS:
        count = 1  # 当前落子位置

        # 正向检查
        for step in range(1, 5):
            nx, ny = x + dx * step, y + dy * step
            if 0 <= nx < 9 and 0 <= ny < 9 and board[nx][ny] == player:
                count += 1
            else:
                break

        # 反向检查
        for step in range(1, 5):
            nx, ny = x - dx * step, y - dy * step
            if 0 <= nx < 9 and 0 <= ny < 9 and board[nx][ny] == player:
                count += 1
            else:
                break

        if count >= 5:
            return True
    return False

def find_open_threes(board, player):
    """检测player的活三并返回防守位置"""
    defense_points = []
    size = 9

    for i in range(size):
        for j in range(size):
            if board[i][j] != player:
                continue

            for dx, dy in DIRECTIONS:
                # 检查连续三子模式
                positions = []
                for step in range(3):
                    ni, nj = i + dx * step, j + dy * step
                    if 0 <= ni < size and 0 <= nj < size:
                        positions.append((ni, nj))

                if len(positions) < 3:
                    continue

                # 验证连续三子
                if all(board[x][y] == player for x, y in positions):
                    # 检查两端是否可防守
                    start_x, start_y = positions[0][0] - dx, positions[0][1] - dy
                    end_x, end_y = positions[2][0] + dx, positions[2][1] + dy

                    if (0 <= start_x < size and 0 <= start_y < size
                            and board[start_x][start_y] == 0):
                        defense_points.append((start_x, start_y))

                    if (0 <= end_x < size and 0 <= end_y < size
                            and board[end_x][end_y] == 0):
                        defense_points.append((end_x, end_y))

    return list(set(defense_points))  # 去重

def evaluate_position(board, x, y, player):
    """评估(x,y)位置对player的价值"""
    if board[x][y] != 0:
        return 0  # 非空位置无效

    score = 0
    for dx, dy in DIRECTIONS:
        # 计算该方向潜力
        line_score = 0
        for step in range(1, 5):
            nx, ny = x + dx * step, y + dy * step
            if 0 <= nx < 9 and 0 <= ny < 9:
                if board[nx][ny] == player:
                    line_score += 1
                elif board[nx][ny] == -player:
                    line_score -= 0.5
                else:
                    break
        score += line_score

    # 中央位置加成
    center_bonus = 1 - (abs(x - 4) + abs(y - 4)) / 16
    return score * (1 + center_bonus)

def get_next_move(board_data):
    """核心决策函数"""
    board = np.array(board_data['board'])
    size = 9

    # 确定当前玩家（黑棋先手）
    black_count = np.sum(board == 1)
    white_count = np.sum(board == -1)
    current_player = 1 if black_count == white_count else -1

    # 1. 检查必胜阵法
    pattern_move = check_winning_pattern(board, current_player)
    if pattern_move:
        return {"move": list(pattern_move)}

    # 2. 检查自己是否能立即获胜
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0 and is_win(board, i, j, current_player):
                return {"move": [i, j]}

    # 3. 检查对手是否能立即获胜
    opponent = -current_player
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0 and is_win(board, i, j, opponent):
                return {"move": [i, j]}

    # 4. 防守对手的活三
    defense_points = find_open_threes(board, opponent)
    if defense_points:
        # 选择评分最高的防守点
        best_score = -float('inf')
        best_move = None
        for x, y in defense_points:
            score = evaluate_position(board, x, y, current_player)
            if score > best_score:
                best_score = score
                best_move = [x, y]
        return {"move": best_move}

    # 5. 评估函数选择最佳位置
    best_score = -float('inf')
    best_move = [4, 4]  # 默认中心位置
    for i in range(size):
        for j in range(size):
            if board[i][j] == 0:
                player_score = evaluate_position(board, i, j, current_player)
                opponent_score = evaluate_position(board, i, j, opponent) * 0.7
                total_score = player_score + opponent_score

                if total_score > best_score:
                    best_score = total_score
                    best_move = [i, j]

    return {"move": best_move}

def load_input_data(file_path):
    """安全加载JSON文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not data or 'board' not in data:
                raise ValueError("JSON必须包含'board'字段")
            return data
        except json.JSONDecodeError:
            raise ValueError("文件内容不是有效的JSON")

if __name__ == "__main__":
    try:
        # 文件路径改为你的实际路径
        input_data = load_input_data('input.json')
        print("成功加载输入数据")

        # 打印调试信息
        print("棋盘数据预览:")
        for row in input_data['board'][:3]:  # 只打印前3行
            print(row)

        # 调用核心逻辑
        result = get_next_move(input_data)
        print("AI推荐落子:", result)

    except Exception as e:
        print(f"程序出错: {str(e)}")
        print("请检查：")
        print("1. input.json文件是否存在")
        print("2. 文件内容是否符合要求")
        print("3. 文件编码是否为UTF-8")