import json
from GomokuClient import GomokuClient
import tkinter as tk
from tkinter import messagebox, scrolledtext
from WinStrategy import WinStrategy
import time

def create_new_root():
    """创建新的Tkinter根窗口，解决重复运行问题"""
    try:
        # 尝试销毁可能存在的旧窗口
        root = tk.Tk()
        root.withdraw()
        root.destroy()
    except:
        pass
    return tk.Tk()

def show_result_dialog(result):
    """弹窗显示落子结果，包含详细的决策理由"""
    root = tk.Tk()
    root.title("五子棋算法落子结果")
    root.geometry("600x400")
    root.resizable(True, True)

    # 顶部信息框架
    top_frame = tk.Frame(root)
    top_frame.pack(fill="x", padx=20, pady=10)

    # 玩家信息
    player_label = tk.Label(top_frame, text=f"玩家: {result.get('player', '未知')}", font=("SimHei", 12, "bold"))
    player_label.pack(anchor="w", pady=5)

    # 落子位置信息
    pos_label = tk.Label(top_frame, text=f"落子位置: {result.get('move_math', '未生成')} (数学坐标)",
                         font=("SimHei", 12))
    pos_label.pack(anchor="w", pady=5)

    # 决策理由标题
    reason_title = tk.Label(root, text="决策理由:", font=("SimHei", 12, "bold"))
    reason_title.pack(anchor="w", padx=20, pady=5)

    # 决策理由文本框
    reason_text = scrolledtext.ScrolledText(root, height=8, width=70, wrap=tk.WORD, font=("SimHei", 10))
    reason_text.pack(fill="both", expand=True, padx=20, pady=5)
    reason_text.insert(tk.END, result.get("reason", "无具体理由"))
    reason_text.config(state=tk.DISABLED)

    # 游戏状态信息
    status_frame = tk.Frame(root)
    status_frame.pack(fill="x", padx=20, pady=10)

    status_label = tk.Label(status_frame, text="游戏状态:", font=("SimHei", 12, "bold"))
    status_label.pack(anchor="w", pady=5)

    is_over = result.get("game_status", {}).get("isGameOver", False)
    status_text = f"进行中 - 下一步: {'黑棋' if result.get('game_status', {}).get('nextPlayer') == 1 else '白棋'}"
    if is_over:
        winner = result.get("game_status", {}).get("winner")
        status_text = f"已结束 - {'黑棋胜' if winner == 1 else '白棋胜' if winner == -1 else '平局'}"

    status_value = tk.Label(status_frame, text=status_text, font=("SimHei", 10))
    status_value.pack(anchor="w", pady=5)

    root.mainloop()


def print_board(board):
    """打印棋盘状态"""
    if not board:
        return
    print("\n当前棋盘状态:")
    for row in board:
        print(" ".join(f"{cell:2d}" for cell in row))


def auto_play(client, game_id, my_player):
    while True:
        board_data = client.get_chessboard(game_id, show_status=True)
        if board_data.get("isGameOver"):
            print("游戏结束！")
            break

        if board_data.get("nextPlayer") == my_player:
            board_json = json.dumps({"board": board_data["board"]})
            strategy = WinStrategy()
            strategy.load_board(board_json)
            strategy.player = my_player
            move_math, reason = strategy.generate_move()
            print(f"[DEBUG] Web_main.py generate_move返回: move_math={move_math}")
            result = client.submit_move_math(game_id, move_math, my_player)
            print(f"[DEBUG] Web_main.py调用submit_move_math: move_math={move_math}")
            print(f"自动落子: {move_math} 理由: {reason}")
            time.sleep(1)
        else:
            print("等待对手落子...")
            time.sleep(2)


def main():
    """主程序：交互界面与落子流程控制（增强循环稳定性）"""
    print("===== 五子棋算法自动落子系统 =====")

    # 初始化客户端（放入try块，确保资源可释放）
    base_url = input(
        "请输入前端API地址（默认: https://gomoku.handywote.site/api）: ") or "https://gomoku.handywote.site/api"
    client = None
    try:
        client = GomokuClient(base_url)
    except Exception as e:
        print(f"客户端初始化失败: {str(e)}")
        return

    # 获取或设置游戏ID
    game_id = 0
    try:
        game_id_input = input("请输入游戏ID（默认: 0）: ").strip()
        game_id = int(game_id_input) if game_id_input else 0
    except ValueError:
        print("游戏ID输入无效，使用默认值0")
        game_id = 0

    # 主循环（添加异常捕获，防止意外跳出）
    while True:
        try:
            print("\n操作菜单:")
            print("1. 创建新游戏")
            print("2. 查看当前棋盘")
            print("3. 算法自动落子（含详细理由）")
            print("4. 显示帮助")
            print("5. 自动对弈（无需手动，直到结束）")
            print("0. 退出程序")

            choice = input("请选择操作 (0-5): ").strip()

            if choice == "0":
                print("程序已退出，再见！")
                break

            elif choice == "1":
                # 创建新游戏（添加输入验证）
                board_size_input = input("请输入棋盘大小（默认: 9）: ").strip()
                board_size = int(board_size_input) if board_size_input else 9
                result = client.create_chessboard(game_id, board_size)
                print(json.dumps(result, indent=2, ensure_ascii=False))

            elif choice == "2":
                # 查看当前棋盘
                result = client.get_chessboard(game_id)
                print("当前棋盘状态:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print_board(result.get("board", []))

            elif choice == "3":
                # 算法自动落子（添加详细异常捕获）
                print("算法计算中...")
                try:
                    # 1. 获取棋盘数据
                    board_data = client.get_chessboard(game_id, show_status=True)
                    if board_data.get("code") != 0:
                        print("获取棋盘失败")
                        continue
                    board_json = json.dumps({
                        "board": board_data["board"]
                    })
                    # 2. 算法生成落子
                    strategy = WinStrategy()
                    strategy.load_board(board_json)
                    strategy.player = board_data.get("nextPlayer", 1)
                    move_math, reason = strategy.generate_move()
                    print(f"[DEBUG] Web_main.py generate_move返回: move_math={move_math}")
                    # 3. 提交落子
                    print(f"[DEBUG] Web_main.py调用submit_move_math: move_math={move_math}")
                    result = client.submit_move_math(game_id, move_math, strategy.player)
                    # 4. 展示结果
                    print("\n===== 落子结果 =====")
                    print(f"玩家: {'黑棋' if strategy.player == 1 else '白棋'}")
                    print(f"落子位置: {move_math} (数学坐标)")
                    print(f"决策理由: {reason}")
                    if result.get("code") == 0:
                        print("落子结果: 成功")
                    else:
                        print(f"落子失败: {result.get('msg', '未知错误')}")
                except Exception as e:
                    print(f"落子过程异常: {str(e)}")
                    continue  # 异常后继续显示菜单

            elif choice == "4":
                # 显示帮助信息
                print("\n帮助信息:")
                print("1. 创建新游戏: 初始化一个新的五子棋棋盘")
                print("2. 查看棋盘: 获取当前棋盘数据和游戏状态")
                print("3. 算法自动落子: 使用五子棋算法计算最佳落子位置，并展示详细理由")
                print("4. 自动对弈: 自动检测是否轮到自己，自动落子直到游戏结束")
                print("0. 退出程序: 结束当前运行")

            elif choice == "5":
                # 自动对弈模式
                my_player = input("请输入你的执棋颜色（1:黑棋, -1:白棋）：").strip()
                try:
                    my_player = int(my_player)
                    if my_player not in [1, -1]:
                        print("输入有误，仅支持1或-1")
                        continue
                except:
                    print("输入有误，仅支持1或-1")
                    continue
                print("自动对弈开始，按Ctrl+C可随时中止...")
                auto_play(client, game_id, my_player)
                print("自动对弈结束，返回菜单。")

            else:
                print("无效选项，请重新输入")

        except Exception as e:
            # 全局异常捕获，确保循环不中断
            print(f"程序发生异常，继续运行: {str(e)}")
            continue

    # 程序结束前释放资源
    try:
        if client:
            del client
    except:
        pass
if __name__ == "__main__":
    main()