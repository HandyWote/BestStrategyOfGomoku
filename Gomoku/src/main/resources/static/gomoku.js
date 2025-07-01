// GomokuGame 类：负责游戏逻辑与界面交互
class GomokuGame {
    constructor() {
        // 游戏ID（用于标识不同棋局）
        this.gameId = 1;
        // 棋盘尺寸（x 和 y 相同）
        this.boardSize = 9;
        // 当前玩家: 1 为黑棋，-1 为白棋
        this.currentPlayer = 1;
        // 二维数组表示棋盘状态，0 = 空，1 = 黑棋，-1 = 白棋
        this.gameBoard = [];
        // 游戏是否结束标志
        this.isGameOver = false;
        // 获胜方: 1 黑棋，-1 白棋，0 平局
        this.winner = null;

        this.init();
    }

    /**
     * 初始化方法：绑定事件、创建棋盘、显示欢迎信息
     */
    init() {
        this.bindEvents();
        this.createBoard();
        this.showMessage('欢迎来到五子棋游戏！选择棋盘大小后点击"创建游戏"开始新游戏。', 'info');

        // 添加窗口大小变化监听器
        window.addEventListener('resize', () => {
            this.createBoard();
        });
    }

    /**
     * 绑定页面上各按钮和输入框的事件处理函数
     */
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

        // 胜利弹窗事件绑定
        document.getElementById('newGameBtn').addEventListener('click', () => this.startNewGame());
        document.getElementById('closeModalBtn').addEventListener('click', () => this.hideVictoryModal());

        // 点击弹窗背景关闭
        document.getElementById('victoryModal').addEventListener('click', (e) => {
            if (e.target.id === 'victoryModal') {
                this.hideVictoryModal();
            }
        });
    }

    /**
     * 更新棋盘大小：校验范围、重建棋盘、提示消息
     * @param {number} size 新的棋盘边长
     */
    updateBoardSize(size) {
        // 限制棋盘大小在合理范围内
        if (size < 5) size = 5;
        if (size > 25) size = 25;

        this.boardSize = size;
        document.getElementById('boardSize').value = size; // 更新输入框显示
        this.createBoard();
        this.showMessage(`棋盘大小已设置为 ${size}x${size}`, 'info');
    }

    /**
     * 获取可用的棋盘容器尺寸
     */
    getAvailableBoardSize() {
        const boardArea = document.querySelector('.board-area');
        const boardContainer = document.getElementById('gameBoard');

        // 获取容器的实际可用空间
        const containerRect = boardArea.getBoundingClientRect();
        const containerPadding = 80; // board-area的padding (40px * 2)
        const boardPadding = 48; // game-board的padding (24px * 2)

        // 计算可用空间
        const availableWidth = containerRect.width - containerPadding - boardPadding;
        const availableHeight = Math.min(containerRect.height - containerPadding - boardPadding, availableWidth);

        // 在移动设备上，还需要考虑视口限制
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let maxBoardSize;

        if (viewportWidth <= 768) {
            // 移动设备：使用较小的尺寸，留出更多边距
            maxBoardSize = Math.min(availableWidth, viewportWidth * 0.85, viewportHeight * 0.6);
        } else {
            // 桌面设备：使用容器可用空间
            maxBoardSize = Math.min(availableWidth, availableHeight, 600); // 最大600px
        }

        return Math.max(200, maxBoardSize); // 最小200px
    }

    /**
     * 根据可用空间和棋盘大小计算最适合的单元格尺寸
     */
    getResponsiveCellSize() {
        const availableSize = this.getAvailableBoardSize();

        // 计算单元格大小：可用空间除以棋盘格数减1（因为是交点数）
        const calculatedCellSize = availableSize / (this.boardSize - 1);

        // 设置合理的单元格大小范围
        const minCellSize = 15;
        const maxCellSize = 50;

        let cellSize = Math.max(minCellSize, Math.min(maxCellSize, calculatedCellSize));

        // 在移动设备上进一步限制尺寸
        if (window.innerWidth <= 768) {
            cellSize = Math.min(cellSize, 30);
        }

        return Math.floor(cellSize);
    }

    /**
     * 根据单元格尺寸计算按钮尺寸
     */
    getResponsiveCellButtonSize(cellSize) {
        // 按钮尺寸通常是单元格尺寸的80%
        const buttonSize = Math.floor(cellSize * 0.8);
        return Math.max(12, Math.min(40, buttonSize));
    }

    /**
     * 根据单元格尺寸计算棋子尺寸
     */
    getResponsivePieceSize(cellSize) {
        // 棋子尺寸通常是单元格尺寸的70%
        const pieceSize = Math.floor(cellSize * 0.7);
        return Math.max(10, Math.min(35, pieceSize));
    }

    /**
     * 创建棋盘DOM：生成网格线和交点按钮（响应式版本）
     */
    createBoard() {
        const boardContainer = document.getElementById('gameBoard');
        boardContainer.innerHTML = '';

        const grid = document.createElement('div');
        grid.className = 'board-grid responsive';

        // 使用响应式计算的尺寸
        const cellSize = this.getResponsiveCellSize();
        const buttonSize = this.getResponsiveCellButtonSize(cellSize);
        const pieceSize = this.getResponsivePieceSize(cellSize);

        // 计算棋盘总尺寸
        const boardWidth = (this.boardSize - 1) * cellSize;
        const boardHeight = (this.boardSize - 1) * cellSize;

        // 设置CSS变量，用于网格线和其他样式
        grid.style.setProperty('--cell-size', cellSize + 'px');
        grid.style.setProperty('--button-size', buttonSize + 'px');
        grid.style.setProperty('--piece-size', pieceSize + 'px');

        grid.style.width = boardWidth + 'px';
        grid.style.height = boardHeight + 'px';

        // 创建交点按钮
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                const cell = document.createElement('button');
                cell.className = 'cell responsive';
                cell.dataset.row = i.toString();
                cell.dataset.col = j.toString();

                // 计算交点位置
                const x = j * cellSize - buttonSize / 2;
                const y = i * cellSize - buttonSize / 2;

                cell.style.left = x + 'px';
                cell.style.top = y + 'px';
                cell.style.width = buttonSize + 'px';
                cell.style.height = buttonSize + 'px';

                cell.addEventListener('click', () => this.makeMove(i, j));
                grid.appendChild(cell);
            }
        }

        boardContainer.appendChild(grid);

        // 如果有游戏数据，重新渲染棋子
        if (this.gameBoard && this.gameBoard.length > 0) {
            this.updateBoard();
        }
    }

    /**
     * 发起创建游戏的 API 请求
     */
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
                this.showMessage(`创建游戏失败: ${result.msg} `, 'error');
            }
        } catch (error) {
            this.showMessage(`创建游戏时发生错误: ${error.message}`, 'error');
        }
    }

    /**
     * 加载指定游戏数据并更新界面
     */
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

    /**
     * 执行落子操作：调用API并刷新游戏状态
     * @param {number} row 行索引
     * @param {number} col 列索引
     */
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

    /**
     * 更新棋盘显示：根据 this.gameBoard 渲染棋子（响应式版本）
     */
    updateBoard() {
        const cells = document.querySelectorAll('.cell');
        const pieceSize = this.getResponsivePieceSize(this.getResponsiveCellSize());

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
                piece.className = `piece ${this.gameBoard[row][col] === 1 ? 'black' : 'white'} responsive`;
                piece.style.width = pieceSize + 'px';
                piece.style.height = pieceSize + 'px';
                cell.appendChild(piece);
                cell.classList.add('disabled');
            } else {
                cell.classList.remove('disabled');
            }
        });
    }

    /**
     * 更新游戏信息面板：当前玩家和游戏结果
     */
    updateGameInfo() {
        const currentPlayerElement = document.getElementById('currentPlayer');
        const gameStatusElement = document.getElementById('gameStatus');

        if (this.isGameOver) {
            if (this.winner === 0) {
                currentPlayerElement.textContent = '平局';
                gameStatusElement.textContent = '游戏平局';
                this.showVictoryModal(0); // 显示平局弹窗
            } else {
                currentPlayerElement.textContent = this.getPlayerName(this.winner);
                gameStatusElement.textContent = `${this.getPlayerName(this.winner)} 获胜！`;
                this.showVictoryModal(this.winner); // 显示胜利弹窗
            }
        } else {
            currentPlayerElement.textContent = this.getPlayerName(this.currentPlayer);
            gameStatusElement.textContent = '进行中';
        }
    }

    /**
     * 根据玩家值返回名称：1 -> 黑棋，-1 -> 白棋
     * @param {number} player 玩家标识
     * @returns {string}
     */
    getPlayerName(player) {
        return player === 1 ? '黑棋' : '白棋';
    }

    /**
     * 重置游戏：先删除再创建，并提示用户
     */
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

    /**
     * 删除当前游戏：调用DELETE接口，重置本地状态
     * @param {boolean} showMessage 是否显示提示消息
     */
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

    /**
     * 显示提示消息，并在3秒后自动清除
     * @param {string} message 要显示的文本
     * @param {string} type 消息类型: 'success','error','warning','info'
     */
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

    /**
     * 显示胜利弹窗
     * @param {number} winner 胜利者标识: 1=黑棋获胜, -1=白棋获胜, 0=平局
     */
    showVictoryModal(winner) {
        const modal = document.getElementById('victoryModal');
        const titleElement = document.getElementById('victoryTitle');
        const messageElement = document.getElementById('victoryMessage');

        if (winner === 0) {
            titleElement.textContent = '游戏平局';
            messageElement.textContent = '棋盘已满，本局平局！';
        } else if (winner === 1) {
            titleElement.textContent = '黑棋获胜！';
            messageElement.textContent = '恭喜黑方玩家获得胜利！';
        } else {
            titleElement.textContent = '白棋获胜！';
            messageElement.textContent = '恭喜白方玩家获得胜利！';
        }

        modal.classList.add('show');
    }

    /**
     * 隐藏胜利弹窗
     */
    hideVictoryModal() {
        const modal = document.getElementById('victoryModal');
        modal.classList.remove('show');
    }

    /**
     * 开始新游戏：隐藏胜利弹窗，重置游戏状态
     */
    async startNewGame() {
        this.hideVictoryModal();
        await this.resetGame();
    }
}

// 页面加载完成后实例化游戏
document.addEventListener('DOMContentLoaded', () => {
    new GomokuGame();
});
