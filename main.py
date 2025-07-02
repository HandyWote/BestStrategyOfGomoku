# 文件1：main.py
import json
import tkinter as tk
from tkinter import Label
from WinStrategy import WinStrategy

def main():
    # 初始化策略
    strategy = WinStrategy()

    # 选择黑棋或白棋
    color_choice = input("请选择您的颜色（1:黑棋, -1:白棋）：")
    if color_choice == "1":
        strategy.player = 1
    else:
        strategy.player = -1

    # 输入处理
    choice = input("输入方式（1:文件，2:直接粘贴JSON）：")
    if choice == "1":
        path = input("JSON文件路径：")
        try:
            with open(path, "r", encoding="utf-8") as f:
                board_json = f.read()
        except FileNotFoundError:
            print("文件不存在！")
            return
    else:
        board_json = input("JSON数据：")

    # 校验JSON
    try:
        json.loads(board_json)
    except json.JSONDecodeError:
        print("JSON格式错误！")
        return

    # 生成决策
    try:
        strategy.load_board(board_json)
        move, reason = strategy.generate_move()
    except Exception as e:
        print(f"错误：{e}")
        return

    # 输出结果
    result = {
        "next_move": move,
        "reason": reason,
        "player": "黑棋" if strategy.player == 1 else "白棋"
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 弹窗展示（可选）
    try:
        root = tk.Tk()
        root.title("五子棋现在->必胜静态算法")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        root.geometry(f"350x140+{int(sw * 0.75)}+{int(sh / 2)}")
        Label(root, text=f"下一步：{move}\n理由：{reason}", font=("SimHei", 10)).pack(padx=20, pady=20)
        root.mainloop()
    except:
        pass

if __name__ == "__main__":
    main()