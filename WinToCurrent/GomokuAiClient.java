package com.gomoku;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONArray;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Scanner;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * 五子棋AI客户端
 * 可配置角色（黑子/白子）和棋盘ID，定时轮询API并自动下子
 */
public class GomokuAiClient {

    private static final String BASE_URL = "https://gomoku.handywote.site/";
    private static final String API_PATH = "api/gomoku";

    private final HttpClient httpClient;
    private final ScheduledExecutorService scheduler;

    // 配置参数
    private int boardId;
    private int aiPlayer; // 1为黑子，-1为白子
    private int pollInterval = 2; // 轮询间隔（秒）

    // 状态变量
    private boolean isRunning = false;
    private boolean boardExists = false;
    private JSONObject lastBoardState = null;
    private boolean debugMode = true; // 添加调试模式

    public GomokuAiClient() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        this.scheduler = Executors.newScheduledThreadPool(1);
    }

    /**
     * 配置AI参数
     */
    public void configure(int boardId, int aiPlayer, int pollInterval) {
        this.boardId = boardId;
        this.aiPlayer = aiPlayer;
        this.pollInterval = pollInterval;

        String role = aiPlayer == 1 ? "黑子" : "白子";
        System.out.println("AI配置完成:");
        System.out.println("- 棋盘ID: " + boardId);
        System.out.println("- AI角色: " + role);
        System.out.println("- 轮询间隔: " + pollInterval + "秒");
        System.out.println("- 调试模式: " + (debugMode ? "开启" : "关闭"));
    }

    /**
     * 启动AI客户端
     */
    public void start() {
        if (isRunning) {
            System.out.println("AI客户端已经在运行中...");
            return;
        }

        isRunning = true;
        System.out.println("AI客户端启动，开始监控棋盘...");

        // 定时轮询棋盘状态
        scheduler.scheduleAtFixedRate(this::pollAndPlay, 0, pollInterval, TimeUnit.SECONDS);
    }

    /**
     * 停止AI客户端
     */
    public void stop() {
        if (!isRunning) {
            return;
        }

        isRunning = false;
        scheduler.shutdown();
        System.out.println("AI客户端已停止");
    }

    /**
     * 轮询并判断是否需要下子
     */
    private void pollAndPlay() {
        try {
            if (debugMode) {
                System.out.println("开始轮询棋盘状态...");
            }

            // 获取棋盘状态
            JSONObject boardState = getBoardState();

            if (boardState == null) {
                handleBoardNotExists();
                return;
            }

            // 标记棋盘存在
            if (!boardExists) {
                boardExists = true;
                System.out.println("检测到棋盘已创建，开始游戏监控...");
            }

            if (debugMode) {
                System.out.println("收到棋盘状态: " + boardState.toString());
            }

            // 检查是否轮到AI下子
            if (isAiTurn(boardState)) {
                System.out.println("检测到轮到AI下子，准备分析最佳落子位置...");
                makeAiMove(boardState);
            } else {
                if (debugMode) {
                    System.out.println("当前不是AI回合，继续等待...");
                }
            }

            lastBoardState = boardState;

        } catch (Exception e) {
            System.err.println("轮询过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 获取棋盘状态
     */
    private JSONObject getBoardState() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH + "/" + boardId + "?showStatus=true"))
                    .header("Content-Type", "application/json")
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            if (response.statusCode() == 200) {
                return new JSONObject(response.body());
            } else if (response.statusCode() == 404) {
                return null; // 棋盘不存在
            } else {
                System.err.println("获取棋盘状态失败，状态码: " + response.statusCode());
                return null;
            }

        } catch (URISyntaxException | IOException | InterruptedException | JSONException e) {
            System.err.println("获取棋盘状态时发生错误: " + e.getMessage());
            return null;
        }
    }

    /**
     * 处理棋盘不存在的情况
     */
    private void handleBoardNotExists() {
        if (boardExists) {
            boardExists = false;
            System.out.println("棋盘不存在，等待玩家在前端创建棋盘...");
        }
    }

    /**
     * 判断是否轮到AI下子
     */
    private boolean isAiTurn(JSONObject boardState) {
        try {
            // 检查API返回的code字段
            if (!boardState.has("code")) {
                if (debugMode) {
                    System.out.println("API响应中缺少code字段");
                }
                return false;
            }

            int code = boardState.getInt("code");
            if (code != 0) {
                if (debugMode) {
                    System.out.println("API返回错误，code: " + code);
                    if (boardState.has("msg")) {
                        System.out.println("错误信息: " + boardState.getString("msg"));
                    }
                }
                return false;
            }

            // 检查游戏是否结束
            if (boardState.has("isGameOver") && boardState.getBoolean("isGameOver")) {
                if (lastBoardState == null || !lastBoardState.has("isGameOver") ||
                        !lastBoardState.getBoolean("isGameOver")) {
                    // 游戏刚结束，输出结果
                    int winner = boardState.has("winner") ? boardState.getInt("winner") : 0;
                    String result = winner == aiPlayer ? "AI获胜！" :
                            winner == -aiPlayer ? "对手获胜！" :
                                    winner == 2 ? "平局！" : "游戏结束";
                    System.out.println("游戏结束: " + result);
                }
                return false;
            }

            // 检查是否轮到AI
            if (boardState.has("nextPlayer")) {
                int nextPlayer = boardState.getInt("nextPlayer");
                if (debugMode) {
                    System.out.println("当前轮到: " + (nextPlayer == 1 ? "黑子" : "白子") +
                            ", AI是: " + (aiPlayer == 1 ? "黑子" : "白子"));
                }
                return nextPlayer == aiPlayer;
            } else {
                // 如果没有nextPlayer字段，尝试通过棋盘状态推断
                if (debugMode) {
                    System.out.println("API响应中缺少nextPlayer字段，尝试通过棋盘分析当前轮次");
                }
                return inferCurrentPlayer(boardState) == aiPlayer;
            }

        } catch (JSONException e) {
            System.err.println("解析棋盘状态时发生错误: " + e.getMessage());
            e.printStackTrace();
        }

        return false;
    }

    /**
     * 通过棋盘状态推断当前应该轮到哪个玩家
     */
    private int inferCurrentPlayer(JSONObject boardState) {
        try {
            if (!boardState.has("board")) {
                return 1; // 默认黑子先手
            }

            org.json.JSONArray board = boardState.getJSONArray("board");
            int blackCount = 0;
            int whiteCount = 0;

            for (int i = 0; i < board.length(); i++) {
                org.json.JSONArray row = board.getJSONArray(i);
                for (int j = 0; j < row.length(); j++) {
                    int cell = row.getInt(j);
                    if (cell == 1) blackCount++;
                    else if (cell == -1) whiteCount++;
                }
            }

            // 黑子先手，如果黑子数量等于白子数量，轮到黑子
            // 如果黑子数量比白子多1，轮到白子
            if (blackCount == whiteCount) {
                return 1; // 轮到黑子
            } else if (blackCount == whiteCount + 1) {
                return -1; // 轮到白子
            } else {
                return 1; // 异常情况默认黑子
            }

        } catch (JSONException e) {
            System.err.println("推断当前玩家时发生错误: " + e.getMessage());
            return 1; // 默认黑子
        }
    }

    /**
     * AI下子
     */
    private void makeAiMove(JSONObject boardState) {
        try {
            System.out.println("AI开始分析最佳落子位置...");

            // 增强调试信息：打印当前棋盘状态
            if (debugMode && boardState.has("board")) {
                System.out.println("当前棋盘状态:");
                JSONArray boardArray = boardState.getJSONArray("board");
                for (int i = 0; i < boardArray.length(); i++) {
                    System.out.println(boardArray.getJSONArray(i).toString());
                }
            }

            // 使用算法分析并获取最佳落子位置
            int[] bestMove = analyzeAndGetBestMove(boardState);

            if (bestMove != null && bestMove.length >= 2 && bestMove[0] >= 0 && bestMove[1] >= 0) {
                System.out.println("AI分析完成，准备落子: (" + bestMove[0] + "," + bestMove[1] + ")");

                // 执行落子
                boolean success = executeMove(bestMove[0], bestMove[1]);

                if (success) {
                    String role = aiPlayer == 1 ? "黑子" : "白子";
                    System.out.println("AI(" + role + ")落子成功: (" + bestMove[0] + "," + bestMove[1] + ")");
                } else {
                    System.err.println("AI落子失败");
                }
            } else {
                System.err.println("AI无法找到合适的落子位置，分析结果: " +
                        (bestMove == null ? "null" : "(" + bestMove[0] + "," + bestMove[1] + ")"));

                // 尝试简单的落子策略
                int[] simpleMove = findSimpleMove(boardState);
                if (simpleMove != null) {
                    System.out.println("使用简单策略落子: (" + simpleMove[0] + "," + simpleMove[1] + ")");
                    executeMove(simpleMove[0], simpleMove[1]);
                }
            }

        } catch (Exception e) {
            System.err.println("AI下子过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * 执行落子操作
     */
    private boolean executeMove(int x, int y) {
        try {
            String json = "{ \"x\": " + x + ", \"y\": " + y + ", \"player\": " + aiPlayer + " }";

            if (debugMode) {
                System.out.println("发送落子请求: " + json);
            }

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH + "/" + boardId))
                    .header("Content-Type", "application/json")
                    .PUT(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            if (debugMode) {
                System.out.println("落子响应状态码: " + response.statusCode());
                System.out.println("落子响应内容: " + response.body());
            }

            if (response.statusCode() == 200) {
                try {
                    JSONObject result = new JSONObject(response.body());
                    int code = result.getInt("code");

                    if (code == 0) {
                        return true;
                    } else {
                        String msg = result.has("msg") ? result.getString("msg") : "未知错误";
                        System.err.println("落子失败: " + msg);
                        return false;
                    }
                } catch (JSONException e) {
                    System.err.println("解析落子响应失败: " + e.getMessage());
                    return false;
                }
            } else {
                System.err.println("落子请求失败，HTTP状态码: " + response.statusCode());
                return false;
            }

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("执行落子时发生错误: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    /**
     * 使用算法分析并获取最佳落子位置
     */
    private int[] analyzeAndGetBestMove(JSONObject boardState) {
        try {
            if (debugMode) {
                System.out.println("调用算法分析器...");
            }

            // 创建算法分析器
            GomokuWinningAnalyzer analyzer = new GomokuWinningAnalyzer(boardState.toString());

            // 分析必胜策略
            JSONObject analysis = analyzer.analyzeWinningSituations();

            if (debugMode) {
                System.out.println("算法分析结果: " + analysis.toString());
            }

            // 获取针对AI角色的分析结果
            String playerKey = aiPlayer == 1 ? "black_winning_analysis" : "white_winning_analysis";

            if (!analysis.has(playerKey)) {
                if (debugMode) {
                    System.out.println("分析结果中缺少" + playerKey + "字段");
                }
                return null;
            }

            JSONObject playerAnalysis = analysis.getJSONObject(playerKey);

            // 获取推荐落子
            if (playerAnalysis.has("recommended_move")) {
                Object recommendedMove = playerAnalysis.get("recommended_move");
                if (recommendedMove instanceof org.json.JSONArray) {
                    org.json.JSONArray moveArray = (org.json.JSONArray) recommendedMove;
                    if (moveArray.length() >= 2) {
                        int x = moveArray.getInt(0);
                        int y = moveArray.getInt(1);
                        if (debugMode) {
                            System.out.println("算法推荐落子: (" + x + "," + y + ")");
                        }
                        return new int[]{x, y};
                    }
                }
            }

        } catch (Exception e) {
            System.err.println("算法分析时发生错误: " + e.getMessage());
            e.printStackTrace();
        }

        return null;
    }

    /**
     * 简单的落子策略 - 作为备选方案
     */
    private int[] findSimpleMove(JSONObject boardState) {
        try {
            if (!boardState.has("board")) {
                return null;
            }

            org.json.JSONArray board = boardState.getJSONArray("board");
            int size = board.length();

            // 寻找第一个空位
            for (int i = 0; i < size; i++) {
                org.json.JSONArray row = board.getJSONArray(i);
                for (int j = 0; j < row.length(); j++) {
                    if (row.getInt(j) == 0) {
                        return new int[]{i, j};
                    }
                }
            }

            return null;
        } catch (JSONException e) {
            System.err.println("简单落子策略失败: " + e.getMessage());
            return null;
        }
    }

    /**
     * 主函数 - 提供交互式配置和启动
     */
    public static void main(String[] args) {
        GomokuAiClient client = new GomokuAiClient();
        Scanner scanner = new Scanner(System.in);

        try {
            System.out.println("=== 五子棋AI客户端 ===");

            // 配置棋盘ID
            System.out.print("请输入棋盘ID (默认: 1): ");
            String boardIdInput = scanner.nextLine().trim();
            int boardId = boardIdInput.isEmpty() ? 1 : Integer.parseInt(boardIdInput);

            // 配置AI角色
            System.out.print("请选择AI角色 (1=黑子, -1=白子, 默认: 1): ");
            String playerInput = scanner.nextLine().trim();
            int aiPlayer = playerInput.isEmpty() ? 1 : Integer.parseInt(playerInput);

            // 配置轮询间隔
            System.out.print("请输入轮询间隔（秒，默认: 2): ");
            String intervalInput = scanner.nextLine().trim();
            int pollInterval = intervalInput.isEmpty() ? 2 : Integer.parseInt(intervalInput);

            // 配置调试模式
            System.out.print("是否开启调试模式 (y/n, 默认: y): ");
            String debugInput = scanner.nextLine().trim().toLowerCase();
            client.debugMode = debugInput.isEmpty() || "y".equals(debugInput) || "yes".equals(debugInput);

            // 配置并启动AI
            client.configure(boardId, aiPlayer, pollInterval);
            client.start();

            // 等待用户输入停止命令
            System.out.println("\n输入 'stop' 停止AI客户端，输入 'status' 查看状态:");
            while (true) {
                String command = scanner.nextLine().trim().toLowerCase();
                if ("stop".equals(command)) {
                    client.stop();
                    break;
                } else if ("status".equals(command)) {
                    System.out.println("AI客户端状态: " + (client.isRunning ? "运行中" : "已停止"));
                    System.out.println("棋盘存在: " + client.boardExists);
                    System.out.println("调试模式: " + (client.debugMode ? "开启" : "关闭"));
                }
            }

        } catch (Exception e) {
            System.err.println("程序运行错误: " + e.getMessage());
            e.printStackTrace();
        } finally {
            scanner.close();
            client.stop();
        }
    }
}