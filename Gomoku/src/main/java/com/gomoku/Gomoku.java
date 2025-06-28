package com.gomoku;

public class Gomoku {
    private int[][] chessboard;     // 二维数组存储棋盘状态
    private int currentPlayer;      // 当前玩家，1表示黑子，-1表示白子
    private boolean gameOver;       // 游戏是否结束
    private int winner;             // 胜利者，1表示黑子赢，-1表示白子赢，2表示平局
    private String wrongMessage; // 错误信息

    // 创建9x9的默认棋盘
    public Gomoku() {
        this(9, 9);
    }

    // 自定义棋盘大小
    public Gomoku(int x, int y) {
        if (x <= 0 || y <= 0) {
            throw new IllegalArgumentException("棋盘大小必须为正整数");
        }
        this.chessboard = new int[x][y];
        this.currentPlayer = 1;     // 黑子先行
        this.gameOver = false;
        this.winner = 0;
    }

    // 从已有棋盘数组创建游戏
    public Gomoku(int[][] board) {
        if (board == null || board.length == 0) {
            throw new IllegalArgumentException("棋盘数组不能为空");
        }
        this.chessboard = new int[board.length][];
        for (int i = 0; i < board.length; i++) {
            if (board[i] == null || board[i].length != board[0].length) {
                throw new IllegalArgumentException("棋盘必须是矩形");
            }
            this.chessboard[i] = board[i].clone();
        }
        // 根据已有棋子计算当前玩家
        int blackCount = 0;
        int whiteCount = 0;
        for (int[] row : board) {
            for (int cell : row) {
                if (cell == 1) blackCount++;
                else if (cell == -1) whiteCount++;
            }
        }
        this.currentPlayer = blackCount == whiteCount ? 1 : -1;
        this.gameOver = false;
        this.winner = 0;
        // 检查已有棋盘是否已经有胜者
        this.winner = checkWinner();
        this.gameOver = this.winner != 0;
    }

    // 检查胜利者
    public int checkWinner() {
        if (gameOver) return winner;

        int rows = chessboard.length;
        int cols = chessboard[0].length;

        // 检查横向连续五个相同棋子
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j <= cols - 5; j++) {
                int current = chessboard[i][j];
                if (current != 0 &&
                        current == chessboard[i][j+1] &&
                        current == chessboard[i][j+2] &&
                        current == chessboard[i][j+3] &&
                        current == chessboard[i][j+4]) {
                    gameOver = true;
                    winner = current;
                    return winner;
                }
            }
        }

        // 检查纵向连续五个相同棋子
        for (int i = 0; i <= rows - 5; i++) {
            for (int j = 0; j < cols; j++) {
                int current = chessboard[i][j];
                if (current != 0 &&
                        current == chessboard[i+1][j] &&
                        current == chessboard[i+2][j] &&
                        current == chessboard[i+3][j] &&
                        current == chessboard[i+4][j]) {
                    gameOver = true;
                    winner = current;
                    return winner;
                }
            }
        }

        // 检查左上到右下对角线连续五个相同棋子
        for (int i = 0; i <= rows - 5; i++) {
            for (int j = 0; j <= cols - 5; j++) {
                int current = chessboard[i][j];
                if (current != 0 &&
                        current == chessboard[i+1][j+1] &&
                        current == chessboard[i+2][j+2] &&
                        current == chessboard[i+3][j+3] &&
                        current == chessboard[i+4][j+4]) {
                    gameOver = true;
                    winner = current;
                    return winner;
                }
            }
        }

        // 检查右上到左下对角线连续五个相同棋子
        for (int i = 0; i <= rows - 5; i++) {
            for (int j = 4; j < cols; j++) {
                int current = chessboard[i][j];
                if (current != 0 &&
                        current == chessboard[i+1][j-1] &&
                        current == chessboard[i+2][j-2] &&
                        current == chessboard[i+3][j-3] &&
                        current == chessboard[i+4][j-4]) {
                    gameOver = true;
                    winner = current;
                    return winner;
                }
            }
        }

        // 检查是否平局（棋盘已满）
        boolean isFull = true;
        for (int[] row : chessboard) {
            for (int cell : row) {
                if (cell == 0) {
                    isFull = false;
                    break;
                }
            }
            if (!isFull) break;
        }

        if (isFull) {
            gameOver = true;
            winner = 2; // 平局
            return winner;
        }

        return 0; // 胜负未分
    }

    // 获取当前轮到的玩家
    public int checkPlayerNow() {
        return gameOver ? 0 : currentPlayer;
    }

    // 获取当前棋盘状态
    public int[][] getChessboard() {
        int[][] copy = new int[chessboard.length][];
        for (int i = 0; i < chessboard.length; i++) {
            copy[i] = chessboard[i].clone();
        }
        return copy;
    }

    // 在指定位置落子
    public boolean updateChess(int x, int y, int player) {
        // 检查游戏是否已结束
        if (gameOver) {
            wrongMessage = "游戏已结束";
            return false;
        }

        // 检查是否是当前玩家的回合
        if (player != currentPlayer) {
            wrongMessage = "不是当前玩家的回合";
            return false;
        }

        // 检查坐标是否在棋盘范围内
        if (x < 0 || x >= chessboard.length || y < 0 || y >= chessboard[0].length) {
            wrongMessage = "坐标超出棋盘范围";
            return false;
        }

        // 检查该位置是否为空
        if (chessboard[x][y] != 0) {
            wrongMessage = "该位置已有棋子";
            return false;
        }

        // 落子
        chessboard[x][y] = player;

        // 检查游戏状态
        checkWinner();

        // 如果游戏未结束，更换当前玩家
        if (!gameOver) {
            currentPlayer = -currentPlayer;
        }

        return true;
    }

    // 通过数组落子
    public boolean updateChess(int[][] newBoard) {
        int[][] currentBoard = this.chessboard;
        int x = -1, y = -1, diffCount = 0;
        // 检查新棋盘是否与当前棋盘大小一致
        if (newBoard.length != chessboard.length || newBoard[0].length != chessboard[0].length) {
            wrongMessage = "新棋盘大小与当前棋盘不一致";
            return false;
        }

        // 找出修改的位置
        for (int i = 0; i < chessboard.length; i++) {
            for (int j = 0; j < chessboard[0].length; j++) {
                if(newBoard[i] == null || newBoard[i].length != chessboard[0].length) {
                    wrongMessage = "新棋盘必须是矩形";
                    return false;
                }
                if(newBoard[i][j] != currentBoard[i][j]) {
                    x = i;
                    y = j;
                    diffCount++;
                }
            }
        }

        // 只允许修改一个位置
        if (diffCount != 1) {
            wrongMessage = "更新的棋盘必须只在一个位置有变化";
            return false;
        }

        return updateChess(x, y, newBoard[x][y]);
    }

    // 判断游戏是否结束
    public boolean isGameOver() {
        return gameOver;
    }

    // 获取胜利者
    public int getWinner() {
        return winner;
    }

    public String getWrongMessage() {
        return wrongMessage;
    }
}