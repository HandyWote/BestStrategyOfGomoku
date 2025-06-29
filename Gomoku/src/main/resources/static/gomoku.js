class GomokuGame {
    constructor() {
        this.gameId = 1;
        this.boardSize = 9;
        this.currentPlayer = 1; // 1为黑棋，2为白棋
        this.gameBoard = [];
        this.isGameOver = false;
        this.winner = null;

        this.init();
    }

    init() {
        this.bindEvents();
        this.createBoard();
        this.showMessage('欢迎来到五子棋游戏！选择棋盘大小后点击"创建游戏"开始新游戏。', 'info');
    }

    bindEvents() {
        document.getElementById('createGame').addEventListener('click', () => this.createGame());
        document.getElementById('loadGame').addEventListener('click', () => this.loadGame());
        document.getElementById('resetGame').addEventListener('click', () => this.resetGame());
        document.getElementById('deleteGame').addEventListener('click', () => this.deleteGame());
        document.getElementById('gameId').addEventListener('change', (e) => {
            this.gameId = parseInt(e.target.value) || 1;
        });

        // 添加棋盘大小输入事件
        document.getElementById('boardSize').addEventListener('change', (e) => {
            this.updateBoardSize(parseInt(e.target.value) || 9);
        });
    }

    updateBoardSize(size) {
        // 限制棋盘大小在合理范围内
        if (size < 5) size = 5;
        if (size > 25) size = 25;

        this.boardSize = size;
        document.getElementById('boardSize').value = size; // 更新输入框显示
        this.createBoard();
        this.showMessage(`棋盘大小已设置为 ${size}x${size}`, 'info');
    }

    createBoard() {
        const boardContainer = document.getElementById('gameBoard');
        boardContainer.innerHTML = '';

        const grid = document.createElement('div');
        grid.className = 'board-grid';
        grid.style.gridTemplateColumns = `repeat(${this.boardSize}, 1fr)`;

        // 根据棋盘大小添加相应的CSS类
        boardContainer.className = 'game-board ' + this.getBoardSizeClass();

        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                const cell = document.createElement('button');
                cell.className = 'cell';
                cell.dataset.row = i;
                cell.dataset.col = j;
                cell.addEventListener('click', () => this.makeMove(i, j));
                grid.appendChild(cell);
            }
        }

        boardContainer.appendChild(grid);
    }

    getBoardSizeClass() {
        if (this.boardSize <= 9) {
            return 'board-small';
        } else if (this.boardSize <= 15) {
            return 'board-medium';
        } else if (this.boardSize <= 19) {
            return 'board-large';
        } else {
            return 'board-xlarge';
        }
    }

    async createGame() {
        try {
            this.gameId = parseInt(document.getElementById('gameId').value) || 1;

            const response = await fetch('/api/gomoku', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    id: this.gameId,
                    x: this.boardSize,
                    y: this.boardSize
                })
            });

            const result = await response.json();

            if (result.code === 0) {
                this.showMessage(`游戏 ${this.gameId} (${this.boardSize}x${this.boardSize}) 创建成功！`, 'success');
                await this.loadGame();
            } else {
                this.showMessage(`创建游戏失败: ${result.msg}`, 'error');
            }
        } catch (error) {
            this.showMessage(`创建游戏时发生错误: ${error.message}`, 'error');
        }
    }

    async loadGame() {
        try {
            this.gameId = parseInt(document.getElementById('gameId').value) || 1;

            const response = await fetch(`/api/gomoku/${this.gameId}?showStatus=true`);
            const result = await response.json();

            if (result.code === 0) {
                this.gameBoard = result.board;
                this.currentPlayer = result.nextPlayer || 1;
                this.isGameOver = result.isGameOver || false;
                this.winner = result.winner;

                // 根据加载的棋盘数据自动调整棋盘大小
                if (this.gameBoard && this.gameBoard.length > 0) {
                    const loadedBoardSize = this.gameBoard.length;
                    if (loadedBoardSize !== this.boardSize) {
                        this.boardSize = loadedBoardSize;
                        document.getElementById('boardSize').value = loadedBoardSize;
                        this.createBoard();
                    }
                }

                this.updateBoard();
                this.updateGameInfo();
                this.showMessage(`游戏 ${this.gameId} (${this.boardSize}x${this.boardSize}) 加载成功！`, 'success');
            } else {
                this.showMessage(`加载游戏失败: ${result.msg}`, 'error');
            }
        } catch (error) {
            this.showMessage(`加载游戏时发生错误: ${error.message}`, 'error');
        }
    }

    async makeMove(row, col) {
        if (this.isGameOver) {
            this.showMessage('游戏已结束，请重新开始或创建新游戏！', 'warning');
            return;
        }

        if (this.gameBoard[row] && this.gameBoard[row][col] !== 0) {
            this.showMessage('该位置已有棋子，请选择其他位置！', 'warning');
            return;
        }

        try {
            const response = await fetch(`/api/gomoku/${this.gameId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    x: row,
                    y: col,
                    player: this.currentPlayer
                })
            });

            const result = await response.json();

            if (result.code === 0) {
                // 更新成功后重新加载游戏状态
                await this.loadGame();
                this.showMessage(`${this.getPlayerName(this.currentPlayer)} 落子成功！`, 'success');
            } else {
                this.showMessage(`落子失败: ${result.msg}`, 'error');
            }
        } catch (error) {
            this.showMessage(`落子时发生错误: ${error.message}`, 'error');
        }
    }

    updateBoard() {
        const cells = document.querySelectorAll('.cell');

        cells.forEach(cell => {
            const row = parseInt(cell.dataset.row);
            const col = parseInt(cell.dataset.col);

            // 清除现有棋子
            const existingPiece = cell.querySelector('.piece');
            if (existingPiece) {
                existingPiece.remove();
            }

            // 根据棋盘状态添加棋子
            if (this.gameBoard[row] && this.gameBoard[row][col] !== 0) {
                const piece = document.createElement('div');
                piece.className = `piece ${this.gameBoard[row][col] === 1 ? 'black' : 'white'}`;
                cell.appendChild(piece);
                cell.classList.add('disabled');
            } else {
                cell.classList.remove('disabled');
            }
        });
    }

    updateGameInfo() {
        const currentPlayerElement = document.getElementById('currentPlayer');
        const gameStatusElement = document.getElementById('gameStatus');

        if (this.isGameOver) {
            if (this.winner === 0) {
                currentPlayerElement.textContent = '平局';
                gameStatusElement.textContent = '游戏平局';
            } else {
                currentPlayerElement.textContent = this.getPlayerName(this.winner);
                gameStatusElement.textContent = `${this.getPlayerName(this.winner)} 获胜！`;
            }
        } else {
            currentPlayerElement.textContent = this.getPlayerName(this.currentPlayer);
            gameStatusElement.textContent = '进行中';
        }
    }

    getPlayerName(player) {
        return player === 1 ? '黑棋' : '白棋';
    }

    async resetGame() {
        try {
            // 先删除当前游戏
            await this.deleteGame(false);

            // 然后创建新游戏
            await this.createGame();

            this.showMessage('游戏重置成功！', 'success');
        } catch (error) {
            this.showMessage(`重置游戏时发生错误: ${error.message}`, 'error');
        }
    }

    async deleteGame(showMessage = true) {
        try {
            const response = await fetch(`/api/gomoku/${this.gameId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (result.code === 0) {
                // 重置游戏状态
                this.gameBoard = [];
                this.currentPlayer = 1;
                this.isGameOver = false;
                this.winner = null;

                // 清空棋盘显示
                this.createBoard();
                this.updateGameInfo();

                if (showMessage) {
                    this.showMessage(`游戏 ${this.gameId} 删除成功！`, 'success');
                }
            } else {
                if (showMessage) {
                    this.showMessage(`删除游戏失败: ${result.msg}`, 'error');
                }
            }
        } catch (error) {
            if (showMessage) {
                this.showMessage(`删除游戏时发生错误: ${error.message}`, 'error');
            }
        }
    }

    showMessage(message, type = 'info') {
        const messageBox = document.getElementById('messageBox');
        messageBox.textContent = message;
        messageBox.className = `message-box ${type}`;

        // 3秒后清除消息
        setTimeout(() => {
            if (messageBox.textContent === message) {
                messageBox.textContent = '';
                messageBox.className = 'message-box';
            }
        }, 3000);
    }
}

// 页面加载完成后初始化游戏
document.addEventListener('DOMContentLoaded', () => {
    new GomokuGame();
});
