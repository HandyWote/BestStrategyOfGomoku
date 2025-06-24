import abc, json


class AI(abc.ABC):
    @abc.abstractmethod
    def get_response(self, prompt)->str:
        pass
    @abc.abstractmethod
    def __init__(self)->None:
        pass

class Deepseek(AI):
    from openai import OpenAI
    def __init__(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
            self.api_key = config.get("deepseek_api_key")
            self.model_name = config.get("deepseek_model_name")
            self.url = config.get("deepseek_url")
        
    def get_response(self, prompt) -> tuple[int, int]:
        system_prompt = """
【严格指令】你必须完全按照以下规则执行，禁止自由发挥！

一个专业的五子棋大师角色(gomoku-master)，包含以下核心内容：

1. 五子棋基本规则：
- 9x9棋盘，先连成五子者胜
- 无禁手规则
- 黑棋先行(1)，白棋后行(-1)
2. 棋盘JSON格式： { "board": [ [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], ... [0,0,0,0,0,0,0,0,0] ] }
- 0: 空位
- 1: 黑棋
- -1: 白棋
- 9x9网格，坐标从(0,0)到(8,8)
- 横坐标(x)表示行号(0-8)
- 纵坐标(y)表示列号(0-8)
3. 胜利条件：
- 横着连续五个相同棋子
- 竖着连续五个相同棋子
- 斜着连续五个相同棋子 (任意方向连五即获胜)

4. 专业策略优先级：
1) 自己连五胜
2) 阻止对方连五
3) 创造活四/冲四
4) 创造活三
5) 防守对方活三/冲四
6) 占据中心位置
7) 创造双活二
8) 随机选择有效位置

【战术指导】
- 活三：两端未被阻挡的三连子，必须防守
- 冲四：一端被阻挡的四连子，必须防守
- 活四：两端未被阻挡的四连子，直接获胜
- 双活三：同时形成两个活三，必胜局面
- 优先占据中心位置(4,4)
- 当多个同等优先级选择时，随机选择

【绝对要求】
1. 必须严格按上述优先级决策
2. 必须选择值为0的空位
3. 必须返回"x y"格式坐标
4. 禁止返回任何其他内容
5. 禁止修改棋盘数据

5. 输出要求：
- 直接返回"x y"格式坐标
- 如"4 5"表示第4行第5列
- 必须确保是有效空位

"""
        client = self.OpenAI(api_key=self.api_key, base_url=self.url)
        re = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            top_p=0.3,  # 适度控制随机性
            temperature=0.5  # 平衡确定性和多样性
        )
        try:
            content = re.choices[0].message.content.strip()
            if not content.replace(" ","").isdigit() or len(content.split()) != 2:
                raise ValueError("返回格式错误")
            
            x, y = map(int, content.split())
            board = json.loads(prompt)["board"]
            
            # 严格验证落子位置
            if not (0 <= x < 9 and 0 <= y < 9):
                raise ValueError("坐标超出范围")
            if board[x][y] != 0:
                raise ValueError("目标位置已有棋子")
                
            return x, y
        except Exception as e:
            # 如果AI返回无效位置，选择第一个有效空位
            board = json.loads(prompt)["board"]
            for i in range(9):
                for j in range(9):
                    if board[i][j] == 0:
                        return i, j
            return 4, 4  # 默认中心位置

def get_board():
    try:
        with open("board.json", "r") as f:
            return json.load(f).get("board")
    except FileNotFoundError:
        return [
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
        ]

def write_board(x: int, y: int):
    board = get_board()
    board_sum = sum(sum(row) for row in board)
    board[x][y] = 1 if board_sum == 0 else -1
    with open("board.json", "w") as f:
        json.dump({"board": board}, f)

if __name__ == "__main__":
    ai = Deepseek()
    current_board = get_board()
    response = ai.get_response(json.dumps({"board": current_board}))
    print(response)
    write_board(response[0], response[1])
