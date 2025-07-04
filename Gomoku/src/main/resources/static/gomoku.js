// 游戏配置常量
const GAME_CONFIG = {
    // 棋盘尺寸限制
    MIN_BOARD_SIZE: 5,        // 最小棋盘大小（5×5）
    MAX_BOARD_SIZE: 25,       // 最大棋盘大小（25×25）
    DEFAULT_BOARD_SIZE: 9,    // 默认棋盘大小（9×9）

    // 单元格尺寸限制
    MIN_CELL_SIZE: 15,        // 最小单元格尺寸（像素）
    MAX_CELL_SIZE: 50,        // 最大单元格尺寸（像素）

    // 显示限制
    MAX_BOARD_DISPLAY_SIZE: 600,  // 棋盘最大显示尺寸（像素）
    MOBILE_BREAKPOINT: 768,       // 移动端屏幕宽度断点（像素）

    // 交互设置
    MESSAGE_TIMEOUT: 3000,    // 消息显示超时时间（毫秒）
    REFRESH_INTERVAL: 1000,   // 棋盘刷新间隔时间（毫秒）

    // 尺寸比例
    BUTTON_SIZE_RATIO: 0.8,   // 交点按钮相对于单元格的尺寸比例
    PIECE_SIZE_RATIO: 0.7,    // 棋子相对于单元格的尺寸比例

    // 玩家标识
    PLAYER: {
        BLACK: 1,             // 黑棋玩家标识
        WHITE: -1,            // 白棋玩家标识
        DRAW: 2,              // 平局标识
        NONE: 0               // 无玩家/空位标识
    }
};

// GomokuGame 类：负责游戏逻辑与界面交互
class GomokuGame {
    constructor() {
        this.gameId = 1;
        this.boardSize = GAME_CONFIG.DEFAULT_BOARD_SIZE;
        this.nextPlayer = GAME_CONFIG.PLAYER.BLACK;
        this.board = [];
        this.isGameOver = false;
        this.winner = GAME_CONFIG.PLAYER.NONE;
        this.msg = null;
        this.gameExists = false; // 添加游戏存在标志
        this.refreshTimer = null; // 添加刷新定时器

        // 缓存DOM元素
        this.elements = this.cacheElements();

        this.init();
    }

    /**
     * 缓存常用DOM元素
     */
    cacheElements() {
        return {
            gameId: document.getElementById('gameId'),
            boardSize: document.getElementById('boardSize'),
            gameBoard: document.getElementById('gameBoard'),
            currentPlayer: document.getElementById('currentPlayer'),
            gameStatus: document.getElementById('gameStatus'),
            messageBox: document.getElementById('messageBox'),
            victoryModal: document.getElementById('victoryModal'),
            victoryTitle: document.getElementById('victoryTitle'),
            victoryMessage: document.getElementById('victoryMessage')
        };
    }

    /**
     * 初始化方法
     */
    init() {
        this.bindEvents();
        this.createBoard();
        this.showMessage('欢迎来到五子棋游戏！请先创建游戏才能开始对弈。', 'info');

        window.addEventListener('resize', () => this.createBoard());
    }

    /**
     * 绑定所有事件处理函数
     */
    bindEvents() {
        // 游戏控制按钮
        const buttonEvents = [
            ['createGame', () => this.createGame()],
            ['loadGame', () => this.loadGame()],
            ['resetGame', () => this.resetGame()],
            ['deleteGame', () => this.deleteGame()],
            ['newGameBtn', () => this.startNewGame()],
            ['closeModalBtn', () => this.hideVictoryModal()]
        ];

        buttonEvents.forEach(([id, handler]) => {
            document.getElementById(id).addEventListener('click', handler);
        });

        // 输入框事件
        this.elements.gameId.addEventListener('change', (e) => {
            this.gameId = parseInt(e.target.value) || 1;
        });

        this.elements.boardSize.addEventListener('change', (e) => {
            this.updateBoardSize(parseInt(e.target.value) || GAME_CONFIG.DEFAULT_BOARD_SIZE);
        });

        // 弹窗事件
        this.elements.victoryModal.addEventListener('click', (e) => {
            if (e.target.id === 'victoryModal') {
                this.hideVictoryModal();
            }
        });
    }

    /**
     * 更新棋盘大小
     */
    updateBoardSize(size) {
        size = Math.max(GAME_CONFIG.MIN_BOARD_SIZE, Math.min(GAME_CONFIG.MAX_BOARD_SIZE, size));

        this.boardSize = size;
        this.elements.boardSize.value = size;
        this.createBoard();
        this.showMessage(`棋盘大小已设置为 ${size}x${size}`, 'info');
    }

    /**
     * 响应式尺寸计算器
     */
    calculateResponsiveSizes() {
        const boardArea = document.querySelector('.board-area');
        const containerRect = boardArea.getBoundingClientRect();
        const isMobile = window.innerWidth <= GAME_CONFIG.MOBILE_BREAKPOINT;

        // 计算可用空间
        const containerPadding = 80;
        const boardPadding = 48;
        const availableWidth = containerRect.width - containerPadding - boardPadding;

        let cellSize;

        if (isMobile) {
            // 移动端：让棋盘宽度几乎占满屏幕宽度
            const screenWidth = window.innerWidth;
            const mobileMargin = 40; // 两侧留出的边距
            const targetBoardWidth = screenWidth - mobileMargin;

            // 根据目标棋盘宽度计算单元格大小
            cellSize = targetBoardWidth / this.boardSize;

            // 限制单元格大小范围，确保棋盘不会过大或过小
            cellSize = Math.max(15, Math.min(45, cellSize));
        } else {
            // 桌面端尺寸计算
            const availableHeight = Math.min(containerRect.height - containerPadding - boardPadding, availableWidth);
            const maxBoardSize = Math.min(availableWidth, availableHeight, GAME_CONFIG.MAX_BOARD_DISPLAY_SIZE);
            const availableSize = Math.max(200, maxBoardSize);

            cellSize = Math.max(
                GAME_CONFIG.MIN_CELL_SIZE,
                Math.min(GAME_CONFIG.MAX_CELL_SIZE, availableSize / (this.boardSize - 1))
            );
        }

        cellSize = Math.floor(cellSize);

        return {
            cellSize,
            buttonSize: Math.max(12, Math.min(40, Math.floor(cellSize * GAME_CONFIG.BUTTON_SIZE_RATIO))),
            pieceSize: Math.max(10, Math.min(35, Math.floor(cellSize * GAME_CONFIG.PIECE_SIZE_RATIO))),
            boardWidth: (this.boardSize - 1) * cellSize,
            boardHeight: (this.boardSize - 1) * cellSize
        };
    }

    /**
     * 创建棋盘DOM
     */
    createBoard() {
        this.elements.gameBoard.innerHTML = '';

        const grid = document.createElement('div');
        grid.className = 'board-grid responsive';

        const sizes = this.calculateResponsiveSizes();

        // 设置CSS变量和样式
        Object.entries({
            '--cell-size': sizes.cellSize + 'px',
            '--button-size': sizes.buttonSize + 'px',
            '--piece-size': sizes.pieceSize + 'px',
            width: sizes.boardWidth + 'px',
            height: sizes.boardHeight + 'px'
        }).forEach(([property, value]) => {
            grid.style.setProperty(property, value);
        });

        // 创建交点按钮
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                const cell = this.createCell(i, j, sizes);
                grid.appendChild(cell);
            }
        }

        this.elements.gameBoard.appendChild(grid);

        // 重新渲染已有棋子
        if (this.board?.length > 0) {
            this.updateBoard();
        }
    }

    /**
     * 创建单个交点按钮
     */
    createCell(row, col, sizes) {
        const cell = document.createElement('button');
        cell.className = 'cell responsive';
        cell.dataset.row = row.toString();
        cell.dataset.col = col.toString();

        const x = col * sizes.cellSize - sizes.buttonSize / 2;
        const y = row * sizes.cellSize - sizes.buttonSize / 2;

        Object.assign(cell.style, {
            left: x + 'px',
            top: y + 'px',
            width: sizes.buttonSize + 'px',
            height: sizes.buttonSize + 'px'
        });

        cell.addEventListener('click', () => this.makeMove(row, col));
        return cell;
    }

    /**
     * 通用API请求处理
     */
    async apiRequest(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: { 'Content-Type': 'application/json' },
                ...options
            });
            return await response.json();
        } catch (error) {
            throw new Error(`网络请求失败: ${error.message}`);
        }
    }

    /**
     * 开始定时刷新棋盘
     */
    startRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
        }

        this.refreshTimer = setInterval(() => {
            if (this.gameExists) {
                this.loadGame();
            }
        }, GAME_CONFIG.REFRESH_INTERVAL);
    }

    /**
     * 停止定时刷新棋盘
     */
    stopRefresh() {
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }
    }

    /**
     * 创建游戏
     */
    async createGame() {
        try {
            this.gameId = parseInt(this.elements.gameId.value) || 1;

            const result = await this.apiRequest('/api/gomoku', {
                method: 'POST',
                body: JSON.stringify({
                    id: this.gameId,
                    x: this.boardSize,
                    y: this.boardSize
                })
            });

            if (result.code === 0) {
                this.gameExists = true; // 根据code判断游戏创建成功
                this.showMessage(`游戏 ${this.gameId} (${this.boardSize}x${this.boardSize}) 创建成功！`, 'success');
                await this.loadGame();
                this.startRefresh(); // 开始定时刷新
            } else {
                this.gameExists = false; // 根据code判断游戏创建失败
                this.showMessage(`创建游戏失败: ${result.msg}`, 'error');
                this.stopRefresh(); // 停止定时刷新
            }
        } catch (error) {
            this.gameExists = false; // 网络错误时标记游戏不存在
            this.showMessage(`创建游戏时发生错误: ${error.message}`, 'error');
            this.stopRefresh(); // 停止定时刷新
        }
    }

    /**
     * 加载游戏
     */
    async loadGame() {
        try {
            this.gameId = parseInt(this.elements.gameId.value) || 1;

            const result = await this.apiRequest(`/api/gomoku/${this.gameId}?showStatus=true`);

            if (result.code === 0) {
                this.gameExists = true; // 根据code判断游戏加载成功
                this.updateGameState(result);
                // 只在手动加载时显示消息，避免定时刷新时频繁显示
                if (!this.refreshTimer) {
                    this.showMessage(`游戏 ${this.gameId} (${this.boardSize}x${this.boardSize}) 加载成功！`, 'success');
                }
                // 如果游戏存在但没有开始刷新，则开始刷新
                if (!this.refreshTimer && !this.isGameOver) {
                    this.startRefresh();
                }
            } else {
                this.gameExists = false; // 根据code判断游戏不存在
                this.showMessage(`加载游戏失败: ${result.msg || '棋盘不存在，请先创建游戏！'}`, 'error');
                this.stopRefresh(); // 停止定时刷新
            }
        } catch (error) {
            this.gameExists = false; // 网络错误时标记游戏不存在
            // 只在手动加载时显示错误消息，避免定时刷新时频繁显示
            if (!this.refreshTimer) {
                this.showMessage(`加载游戏时发生错误: ${error.message}`, 'error');
            }
            this.stopRefresh(); // 停止定时刷新
        }
    }

    /**
     * 更新游戏状态
     */
    updateGameState(result) {
        this.board = result.board;
        this.nextPlayer = result.nextPlayer || GAME_CONFIG.PLAYER.BLACK;
        this.isGameOver = result.isGameOver || false;
        this.winner = result.winner || GAME_CONFIG.PLAYER.NONE;

        // 自动调整棋盘大小
        if (this.board?.length > 0) {
            const loadedBoardSize = this.board.length;
            if (loadedBoardSize !== this.boardSize) {
                this.boardSize = loadedBoardSize;
                this.elements.boardSize.value = loadedBoardSize;
                this.createBoard();
            }
        }

        this.updateBoard();
        this.updateGameInfo();
    }

    /**
     * 执行落子操作
     */
    async makeMove(row, col) {
        // 检查游戏是否存在
        if (!this.gameExists) {
            this.showMessage('棋盘不存在，请先创建游戏！', 'warning');
            return;
        }

        if (this.isGameOver) {
            this.showMessage('游戏已结束，请重新开始或创建新游戏！', 'warning');
            return;
        }

        if (this.board[row]?.[col] !== 0) {
            this.showMessage('该位置已有棋子，请选择其他位置！', 'warning');
            return;
        }

        try {
            const result = await this.apiRequest(`/api/gomoku/${this.gameId}`, {
                method: 'PUT',
                body: JSON.stringify({
                    x: row,
                    y: col,
                    player: this.nextPlayer
                })
            });

            if (result.code === 0) {
                // 根据code判断落子成功
                await this.loadGame();
                this.handleMoveResult();
            } else {
                // 根据code判断落子失败，可能是棋盘不存在
                if (result.msg && result.msg.includes('棋盘不存在')) {
                    this.gameExists = false;
                    this.showMessage('棋盘不存在，请先创建游戏！', 'error');
                } else {
                    this.showMessage(`落子失败: ${result.msg}`, 'error');
                }
            }
        } catch (error) {
            this.showMessage(`落子时发生错误: ${error.message}`, 'error');
        }
    }

    /**
     * 处理落子结果
     */
    handleMoveResult() {
        if (this.isGameOver) {
            const messages = {
                [GAME_CONFIG.PLAYER.DRAW]: '游戏结束，平局！',
                [GAME_CONFIG.PLAYER.BLACK]: '游戏结束，黑棋获胜！',
                [GAME_CONFIG.PLAYER.WHITE]: '游戏结束，白棋获胜！'
            };
            this.showMessage(messages[this.winner] || '游戏结束！', 'success');
        } else {
            const previousPlayer = this.nextPlayer === GAME_CONFIG.PLAYER.BLACK ? GAME_CONFIG.PLAYER.WHITE : GAME_CONFIG.PLAYER.BLACK;
            this.showMessage(`${this.getPlayerName(previousPlayer)} 落子成功！`, 'success');
        }
    }

    /**
     * 更新棋盘显示
     */
    updateBoard() {
        const cells = document.querySelectorAll('.cell');
        const sizes = this.calculateResponsiveSizes();

        cells.forEach(cell => {
            const row = parseInt(cell.dataset.row);
            const col = parseInt(cell.dataset.col);

            // 清除现有棋子
            cell.querySelector('.piece')?.remove();

            // 添加新棋子
            if (this.board[row]?.[col] !== 0) {
                const piece = this.createPiece(this.board[row][col], sizes.pieceSize);
                cell.appendChild(piece);
                cell.classList.add('disabled');
            } else {
                cell.classList.remove('disabled');
            }
        });
    }

    /**
     * 创建棋子元素
     */
    createPiece(player, size) {
        const piece = document.createElement('div');
        const pieceType = player === GAME_CONFIG.PLAYER.BLACK ? 'black' : 'white';
        piece.className = `piece ${pieceType} responsive`;
        piece.style.width = piece.style.height = size + 'px';
        return piece;
    }

    /**
     * 更新游戏信息面板
     */
    updateGameInfo() {
        if (this.isGameOver) {
            this.stopRefresh();
            this.updateGameEndInfo();
        } else {
            this.elements.currentPlayer.textContent = this.getPlayerName(this.nextPlayer);
            this.elements.gameStatus.textContent = '进行中';
        }
    }

    /**
     * 更新游戏结束信息
     */
    updateGameEndInfo() {
        const winnerMessages = {
            [GAME_CONFIG.PLAYER.DRAW]: { player: '平局', status: '游戏平局' },
            [GAME_CONFIG.PLAYER.BLACK]: { player: '黑棋', status: '黑棋获胜！' },
            [GAME_CONFIG.PLAYER.WHITE]: { player: '白棋', status: '白棋获胜！' }
        };

        const info = winnerMessages[this.winner] || { player: '未知', status: '游戏结束' };
        this.elements.currentPlayer.textContent = info.player;
        this.elements.gameStatus.textContent = info.status;
        this.showVictoryModal(this.winner);
    }

    /**
     * 获取玩家名称
     */
    getPlayerName(player) {
        return player === GAME_CONFIG.PLAYER.BLACK ? '黑棋' : '白棋';
    }

    /**
     * 重置游戏
     */
    async resetGame() {
        try {
            await this.deleteGame(false);
            await this.createGame();
            this.showMessage('游戏重置成功！', 'success');
        } catch (error) {
            this.showMessage(`重置游戏时发生错误: ${error.message}`, 'error');
        }
    }

    /**
     * 删除游戏
     */
    async deleteGame(showMessage = true) {
        try {
            const result = await this.apiRequest(`/api/gomoku/${this.gameId}`, {
                method: 'DELETE'
            });

            if (result.code === 0) {
                this.gameExists = false; // 根据code判断游戏删除成功
                this.stopRefresh(); // 停止定时刷新
                this.resetLocalState();
                if (showMessage) {
                    this.showMessage(`游戏 ${this.gameId} 删除成功！`, 'success');
                }
            } else {
                // 根据code判断删除失败
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
     * 重置本地游戏状态
     */
    resetLocalState() {
        this.board = [];
        this.nextPlayer = GAME_CONFIG.PLAYER.BLACK;
        this.isGameOver = false;
        this.winner = GAME_CONFIG.PLAYER.NONE;
        this.gameExists = false; // 重置游戏存在标志
        this.stopRefresh(); // 停止定时刷新
        this.createBoard();
        this.updateGameInfo();
    }

    /**
     * 显示提示消息
     */
    showMessage(message, type = 'info') {
        this.elements.messageBox.textContent = message;
        this.elements.messageBox.className = `message-box ${type}`;

        setTimeout(() => {
            if (this.elements.messageBox.textContent === message) {
                this.elements.messageBox.textContent = '';
                this.elements.messageBox.className = 'message-box';
            }
        }, GAME_CONFIG.MESSAGE_TIMEOUT);
    }

    /**
     * 显示胜利弹窗
     */
    showVictoryModal(winner) {
        const modalMessages = {
            [GAME_CONFIG.PLAYER.DRAW]: { title: '游戏平局', message: '棋盘已满，本局平局！' },
            [GAME_CONFIG.PLAYER.BLACK]: { title: '黑棋获胜！', message: '恭喜黑方玩家获得胜利！' },
            [GAME_CONFIG.PLAYER.WHITE]: { title: '白棋获胜！', message: '恭喜白方玩家获得胜利！' }
        };

        const info = modalMessages[winner];
        if (info) {
            this.elements.victoryTitle.textContent = info.title;
            this.elements.victoryMessage.textContent = info.message;
            this.elements.victoryModal.classList.add('show');
        }
    }

    /**
     * 隐藏胜利弹窗
     */
    hideVictoryModal() {
        this.elements.victoryModal.classList.remove('show');
    }

    /**
     * 开始新游戏
     */
    async startNewGame() {
        this.hideVictoryModal();
        await this.resetGame();
    }
}

// 页面加载完成后实例化游戏
document.addEventListener('DOMContentLoaded', () => {
    const game = new GomokuGame();

    // 页面卸载时清理定时器
    window.addEventListener('beforeunload', () => {
        game.stopRefresh();
    });
});
