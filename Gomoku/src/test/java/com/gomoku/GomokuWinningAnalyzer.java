package com.gomoku;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

public class GomokuWinningAnalyzer {
    private int[][] board;
    private int boardSize = 9; // 固定为9x9棋盘
    private static final int BLACK = 1;
    private static final int WHITE = -1;
    private static final int EMPTY = 0;

    // 优化的必胜棋型模式库（适应9x9棋盘）
    private static final int[][][] WINNING_PATTERNS = {
            // 活四 (两端开放的四连子)
            {{0, 1, 1, 1, 1, 0}},
            // 冲四 (一端开放的四连子)
            {{0, 1, 1, 1, 1, -1}},
            {{-1, 1, 1, 1, 1, 0}},
            // 双活三（横向和纵向示例，实际需覆盖更多方向）
            {{0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}},
            // 四三组合（简化模式，实际需匹配更多布局）
            {{1, 1, 1, 0}, {0, 1, 1, 1}}
    };

    public GomokuWinningAnalyzer(String jsonBoard) throws JSONException {
        JSONObject jsonObject = new JSONObject(jsonBoard);
        JSONArray jsonBoardArray = jsonObject.getJSONArray("board");
        this.boardSize = jsonBoardArray.length(); // 从JSON获取实际尺寸，确保为9
        this.board = new int[boardSize][boardSize];

        for (int i = 0; i < boardSize; i++) {
            JSONArray row = jsonBoardArray.getJSONArray(i);
            for (int j = 0; j < boardSize; j++) {
                board[i][j] = row.getInt(j);
            }
        }
    }

    // 核心方法：分析必胜局面
    public JSONObject analyzeWinningSituations() throws JSONException {
        JSONObject result = new JSONObject();

        // 分析黑棋必胜策略
        JSONObject blackAnalysis = analyzeForPlayer(BLACK);
        result.put("black_winning_analysis", blackAnalysis);

        // 分析白棋必胜策略
        JSONObject whiteAnalysis = analyzeForPlayer(WHITE);
        result.put("white_winning_analysis", whiteAnalysis);

        return result;
    }

    // 为指定玩家分析必胜策略
    private JSONObject analyzeForPlayer(int player) throws JSONException {
        JSONObject analysis = new JSONObject();

        // 1. 检查立即获胜机会
        List<int[]> immediateWins = findImmediateWins(player);
        analysis.put("immediate_wins", positionsToJSON(immediateWins));

        // 2. 检查必胜模式
        List<int[]> winningPatterns = findWinningPatterns(player);
        analysis.put("winning_patterns", positionsToJSON(winningPatterns));

        // 3. 寻找强制获胜路径（简化版，深度限制为2）
        List<List<int[]>> forcedPaths = findForcedWinPaths(player, 2);
        analysis.put("forced_win_paths", pathsToJSON(forcedPaths));

        // 4. 最佳落子推荐
        int[] bestMove = recommendBestMove(player);
        analysis.put("recommended_move", positionToJSON(bestMove));

        return analysis;
    }

    // 寻找立即获胜的机会
    private List<int[]> findImmediateWins(int player) {
        List<int[]> winningMoves = new ArrayList<>();

        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (board[i][j] == EMPTY) {
                    if (isWinningMove(i, j, player)) {
                        winningMoves.add(new int[]{i, j});
                    }
                }
            }
        }

        return winningMoves;
    }

    // 检查落子后是否获胜（适配9x9边界）
    private boolean isWinningMove(int row, int col, int player) {
        int[][] directions = {{1, 0}, {0, 1}, {1, 1}, {1, -1}}; // 四个方向

        for (int[] dir : directions) {
            int count = 1; // 当前位置

            // 正方向计数
            for (int k = 1; k < 5; k++) {
                int r = row + dir[0] * k;
                int c = col + dir[1] * k;
                if (r < 0 || r >= boardSize || c < 0 || c >= boardSize) break;
                if (board[r][c] != player) break;
                count++;
            }

            // 反方向计数
            for (int k = 1; k < 5; k++) {
                int r = row - dir[0] * k;
                int c = col - dir[1] * k;
                if (r < 0 || r >= boardSize || c < 0 || c >= boardSize) break;
                if (board[r][c] != player) break;
                count++;
            }

            if (count >= 5) {
                return true;
            }
        }

        return false;
    }

    // 寻找匹配必胜模式的位置（适配9x9棋盘边界）
    private List<int[]> findWinningPatterns(int player) {
        List<int[]> patternMoves = new ArrayList<>();

        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (board[i][j] == EMPTY) {
                    for (int[][] pattern : WINNING_PATTERNS) {
                        if (matchesPattern(i, j, player, pattern)) {
                            patternMoves.add(new int[]{i, j});
                            break; // 找到一种模式即可
                        }
                    }
                }
            }
        }

        return patternMoves;
    }

    // 模式匹配（适配9x9棋盘尺寸检查）
    private boolean matchesPattern(int row, int col, int player, int[][] pattern) {
        int patternHeight = pattern.length;
        int patternWidth = pattern[0].length;

        // 检查边界：确保模式能完整放置在9x9棋盘中
        if (row < 0 || row + patternHeight > boardSize ||
                col < 0 || col + patternWidth > boardSize) {
            return false;
        }

        // 检查模式匹配
        for (int i = 0; i < patternHeight; i++) {
            for (int j = 0; j < patternWidth; j++) {
                int r = row + i;
                int c = col + j;
                int cellValue = board[r][c];
                int patternValue = pattern[i][j];

                // 模式匹配规则：
                // 0 = 忽略（可以是任意值）
                // 1 = 当前玩家棋子
                // -1 = 对手棋子
                if (patternValue == 0) {
                    continue;
                } else if (patternValue == 1) {
                    if (cellValue != player) {
                        return false;
                    }
                } else if (patternValue == -1) {
                    if (cellValue != -player) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        return true;
    }

    // 寻找强制获胜路径（优化版，避免递归过深）
    private List<List<int[]>> findForcedWinPaths(int player, int maxDepth) {
        List<List<int[]>> allPaths = new ArrayList<>();

        // 只检查可能的位置，避免全棋盘扫描
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (board[i][j] == EMPTY) {
                    // 尝试在此位置落子
                    board[i][j] = player;

                    // 检查是否立即获胜
                    if (isWinningMove(i, j, player)) {
                        List<int[]> path = new ArrayList<>();
                        path.add(new int[]{i, j});
                        allPaths.add(path);
                    } else if (maxDepth > 1) {
                        // 简化搜索：只找下一步能赢的位置
                        int[] nextMove = findImmediateWin(player);
                        if (nextMove != null) {
                            List<int[]> path = new ArrayList<>();
                            path.add(new int[]{i, j});
                            path.add(nextMove);
                            allPaths.add(path);
                        }
                    }

                    // 恢复棋盘
                    board[i][j] = EMPTY;
                }
            }
        }

        return allPaths;
    }

    // 查找任何立即获胜的位置
    private int[] findImmediateWin(int player) {
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (board[i][j] == EMPTY && isWinningMove(i, j, player)) {
                    return new int[]{i, j};
                }
            }
        }
        return null;
    }

    // 推荐最佳落子（适配9x9棋盘中心逻辑）
    private int[] recommendBestMove(int player) {
        // 1. 优先选择立即获胜的位置
        int[] immediateWin = findImmediateWin(player);
        if (immediateWin != null) {
            return immediateWin;
        }

        // 2. 其次选择能阻止对手获胜的位置
        int[] opponentWin = findImmediateWin(-player);
        if (opponentWin != null) {
            return opponentWin;
        }

        // 3. 选择能形成必胜模式的位置
        List<int[]> winningPatterns = findWinningPatterns(player);
        if (!winningPatterns.isEmpty()) {
            return winningPatterns.get(0);
        }

        // 4. 最后选择9x9棋盘的中心区域（4,4）
        return findCenterMove();
    }

    // 寻找9x9棋盘的中心位置
    private int[] findCenterMove() {
        int center = boardSize / 2; // 对于9x9，center=4

        // 优先选择中心点(4,4)
        if (board[center][center] == EMPTY) {
            return new int[]{center, center};
        }

        // 其次选择中心周围的点（3-5行，3-5列）
        for (int r = center - 1; r <= center + 1; r++) {
            for (int c = center - 1; c <= center + 1; c++) {
                if (r >= 0 && r < boardSize && c >= 0 && c < boardSize &&
                        board[r][c] == EMPTY) {
                    return new int[]{r, c};
                }
            }
        }

        // 如果中心区域已满，选择第一个空位
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                if (board[i][j] == EMPTY) {
                    return new int[]{i, j};
                }
            }
        }

        return new int[]{-1, -1}; // 无位置可下
    }

    // ====== 数组输出方法（保持原有逻辑，适配9x9坐标范围） ======

    // 位置转换为JSON
    private JSONArray positionToJSON(int[] position) {
        JSONArray json = new JSONArray();
        if (position != null && position.length >= 2) {
            json.put(position[0]);
            json.put(position[1]);
        } else {
            json.put(JSONObject.NULL);
            json.put(JSONObject.NULL);
        }
        return json;
    }

    // 位置列表转换为JSON
    private JSONArray positionsToJSON(List<int[]> positions) {
        JSONArray jsonArray = new JSONArray();
        if (positions != null && !positions.isEmpty()) {
            for (int[] pos : positions) {
                jsonArray.put(positionToJSON(pos));
            }
        }
        return jsonArray;
    }

    // 路径列表转换为JSON
    private JSONArray pathsToJSON(List<List<int[]>> paths) {
        JSONArray jsonArray = new JSONArray();
        if (paths != null && !paths.isEmpty()) {
            for (List<int[]> path : paths) {
                JSONArray pathArray = new JSONArray();
                if (path != null && !path.isEmpty()) {
                    for (int[] pos : path) {
                        pathArray.put(positionToJSON(pos));
                    }
                }
                jsonArray.put(pathArray);
            }
        }
        return jsonArray;
    }

    // 导出当前9x9棋盘为JSON
    public String exportBoardToJson() throws JSONException {
        JSONObject json = new JSONObject();
        JSONArray boardArray = new JSONArray();

        for (int i = 0; i < boardSize; i++) {
            JSONArray row = new JSONArray();
            for (int j = 0; j < boardSize; j++) {
                row.put(board[i][j]);
            }
            boardArray.put(row);
        }

        json.put("board", boardArray);
        return json.toString(); // 缩进2个空格
    }

    public static void main(String[] args) {
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
        // 示例：9x9棋盘棋局（中心区域有棋子）
        String jsonBoard = "{\"board\":[" +
                "[0,0,0,0,0,0,0,0,0]," +
                "[0,0,0,0,0,0,0,0,0]," +
                "[0,0,0,1,0,0,0,0,0]," +
                "[0,0,-1,1,-1,0,0,0,0]," +
                "[0,0,0,1,0,1,0,0,0]," +
                "[0,0,0,0,1,0,0,0,0]," +
                "[0,0,0,0,0,0,0,0,0]," +
                "[0,0,0,0,0,0,0,0,0]," +
                "[0,0,0,0,0,0,0,0,0]" +
                "]}";

        try {
            GomokuWinningAnalyzer analyzer = new GomokuWinningAnalyzer(jsonBoard);
            JSONObject analysis = analyzer.analyzeWinningSituations();
            System.out.println("9x9棋盘分析结果:\n" + analysis.toString());
            System.out.println("\n当前9x9棋盘:\n" + analyzer.exportBoardToJson());

            // 测试空位置处理
            System.out.println("\n测试空位置输出:");
            JSONArray emptyPos = analyzer.positionToJSON(null);
            System.out.println("空位置: " + emptyPos);

            // 测试空路径处理
            JSONArray emptyPaths = analyzer.pathsToJSON(new ArrayList<>());
            System.out.println("空路径: " + emptyPaths);
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}