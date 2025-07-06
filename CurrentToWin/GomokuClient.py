import requests
import json
from WinStrategy import WinStrategy
from typing import Dict, List, Tuple


class GomokuClient:
    """五子棋客户端，负责与前端API交互并调用算法落子"""

    def __init__(self, base_url: str = "https://gomoku.handywote.site/api"):
        """初始化客户端，设置API基础地址和算法实例"""
        self.base_url = base_url
        self.strategy = WinStrategy()
        self.session = requests.Session()  # 显式创建会话，便于释放

    def __del__(self):
        """析构函数：释放资源"""
        try:
            self.session.close()
        except:
            pass

    def create_chessboard(self, game_id: int = 0, board_size: int = 9) -> Dict:
        """创建新棋盘"""
        url = f"{self.base_url}/gomoku"
        payload = {"id": game_id, "x": board_size, "y": board_size}
        return self._send_request("POST", url, payload)

    def get_chessboard(self, game_id: int = 0, show_status: bool = True) -> Dict:
        """获取棋盘状态"""
        url = f"{self.base_url}/gomoku/{game_id}"
        if show_status:
            url += "?showStatus=true"
        return self._send_request("GET", url)

    def update_chessboard(self, game_id: int, x_api: int, y_api: int, player: int) -> Dict:
        """落子操作"""
        url = f"{self.base_url}/gomoku/{game_id}"
        payload = {"x": x_api, "y": y_api, "player": player}
        return self._send_request("PUT", url, payload)

    def auto_move(self, game_id: int = 0) -> Dict:
        """自动落子完整流程，返回详细结果"""
        # 1. 获取当前棋盘
        board_data = self.get_chessboard(game_id, show_status=True)
        if board_data.get("code") != 0:
            return {"code": -1, "msg": "获取棋盘失败", "details": board_data}

        # 2. 加载棋盘到算法
        board_json = json.dumps({
            "board": board_data["board"],
            "nextPlayer": board_data["nextPlayer"]
        })
        self.strategy.load_board(board_json)
        self.strategy.player = board_data["nextPlayer"]

        # 3. 算法生成落子及理由
        move_math, reason = self.strategy.generate_move()
        if not move_math:
            return {"code": -1, "msg": "算法未生成有效落子"}

        # 4. 坐标转换（数学坐标→数组坐标）
        try:
            x_math, y_math = map(int, move_math.split(','))
            x_api = 9 - y_math  # 数学x → 数组x（1-based → 0-based）
            y_api = x_math - 1  # 数学y → 数组y（反转，因为数学坐标y=1是棋盘最上方）
        except:
            return {"code": -1, "msg": "算法坐标格式错误", "move": move_math}

        # 5. 校验落子位置
        board = board_data["board"]
        if 0 <= x_api < len(board) and 0 <= y_api < len(board[0]):
            if board[x_api][y_api] != 0:
                return {"code": -1, "msg": "算法生成位置已被占用", "move": move_math}
        else:
            return {"code": -1, "msg": "算法生成位置越界", "move": move_math}

        # 6. 提交落子并返回完整结果
        result = self.update_chessboard(game_id, x_api, y_api, self.strategy.player)
        if result.get("code") == 0:
            result["move_math"] = move_math  # 数学坐标
            result["move_api"] = (x_api, y_api)  # 数组坐标
            result["reason"] = reason  # 决策理由（直接来自算法）
            result["player"] = "黑棋" if self.strategy.player == 1 else "白棋"

            # 补充落子后的棋盘状态和游戏状态
            updated_board = self.get_chessboard(game_id)
            result["updated_board"] = updated_board.get("board", [])
            result["game_status"] = {
                "nextPlayer": updated_board.get("nextPlayer"),
                "isGameOver": updated_board.get("isGameOver", False),
                "winner": updated_board.get("winner")
            }
        return result

    def submit_move_math(self, game_id, move_math, player):
        """接收数学坐标字符串，转换为API坐标并提交落子"""
        x_math, y_math = map(int, move_math.split(','))
        row = 9 - y_math
        col = x_math - 1
        print(f"[DEBUG] 数学坐标: ({x_math},{y_math}) -> 数组坐标: ({row},{col}) -> player: {player}")
        print(f"[DEBUG] 提交API: update_chessboard(game_id={game_id}, x_api={row}, y_api={col}, player={player})")
        return self.update_chessboard(game_id, row, col, player)

    def _send_request(self, method: str, url: str, payload: Dict = None) -> Dict:
        """发送HTTP请求并处理响应"""
        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=payload, timeout=10)
            elif method == "PUT":
                response = requests.put(url, json=payload, timeout=10)
            else:
                return {"code": -1, "msg": "不支持的请求方法"}

            if response.status_code != 200:
                return {"code": -1, "msg": f"HTTP错误: {response.status_code}", "content": response.text}

            return response.json()
        except Exception as e:
            return {"code": -1, "msg": f"请求异常: {str(e)}"}