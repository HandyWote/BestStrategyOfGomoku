package com.gomoku;

import java.io.IOException;
import java.io.PrintStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

/**
 * 五子棋API对接测试示例
 * 使用Java官方HTTP包进行API调用
 */
public class GomokuApiTest {

    private static final String BASE_URL = "https://gomoku.handywote.site/";
    private static final String API_PATH = "api/gomoku";
    private final HttpClient httpClient;

    public GomokuApiTest() {
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    /**
     * 测试创建默认9x9棋盘
     */
    public void testCreateDefaultChessboard() {
        try {
            String json = "{ \"id\": 0 }";

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("创建默认棋盘:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("创建默认棋盘失败: " + e.getMessage());
        }
    }

    /**
     * 测试创建自定义大小棋盘
     */
    public void testCreateCustomChessboard() {
        try {
            String json = "{ \"id\": 1, \"x\": 15, \"y\": 15 }";

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("创建15x15棋盘:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("创建自定义棋盘失败: " + e.getMessage());
        }
    }

    /**
     * 测试导入棋盘数据
     */
    public void testCreateChessboardWithData() {
        try {
            String json = "{ \"id\": 2, \"board\": [" +
                    "[0, 0, 0, 0, 0, 0]," +
                    "[0, 0, 1, 0, 0, 0]," +
                    "[0, 0, -1, 1, 0, 0]," +
                    "[0, 0, -1, 0, 1, 0]," +
                    "[0, 0, -1, 0, 0, 1]," +
                    "[0, 0, 0, 0, 0, 0]" +
                    "] }";

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("创建带数据的棋盘:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("创建带数据的棋盘失败: " + e.getMessage());
        }
    }

    /**
     * 测试获取棋盘数据（不显示状态）
     */
    public void testGetChessboard() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH + "/0"))
                    .header("Content-Type", "application/json")
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("获取棋盘数据:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("获取棋盘数据失败: " + e.getMessage());
        }
    }

    /**
     * 测试获取棋盘数据（显示游戏状态）
     */
    public void testGetChessboardWithStatus() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH + "/0?showStatus=true"))
                    .header("Content-Type", "application/json")
                    .GET()
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("获取棋盘数据（含状态）:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("获取棋盘数据（含状态）失败: " + e.getMessage());
        }
    }

    /**
     * 测试通过坐标落子
     */
    public void testUpdateChessbyPosition() {
        try {
            String json = "{ \"x\": 0, \"y\": 0, \"player\": 1 }";

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH + "/0"))
                    .header("Content-Type", "application/json")
                    .PUT(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("通过坐标落子:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("通过坐标落子失败: " + e.getMessage());
        }
    }

    /**
     * 测试通过棋盘数据更新
     */
    public void testUpdateChessboardWithData() {
        try {
            String json = "{ \"board\": [" +
                    "[1, 0, 0, 0, 0, 0, 0, 0, 0]," +
                    "[0, -1, 0, 0, 0, 0, 0, 0, 0]," +
                    "[0, 0, 1, 0, 0, 0, 0, 0, 0]," +
                    "[0, 0, 0, -1, 0, 0, 0, 0, 0]," +
                    "[0, 0, 0, 0, 1, 0, 0, 0, 0]," +
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0]," +
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0]," +
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0]," +
                    "[0, 0, 0, 0, 0, 0, 0, 0, 0]" +
                    "] }";

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH + "/0"))
                    .header("Content-Type", "application/json")
                    .PUT(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("通过棋盘数据更新:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("通过棋盘数据更新失败: " + e.getMessage());
        }
    }

    /**
     * 测试删除棋盘
     */
    public void testDeleteChessboard() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH + "/0"))
                    .header("Content-Type", "application/json")
                    .DELETE()
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("删除棋盘:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("删除棋盘失败: " + e.getMessage());
        }
    }

    /**
     * 测试完整的游戏流程
     */
    public void testCompleteGameFlow() {
        System.out.println("========== 完整游戏流程测试 ==========");

        // 1. 创建棋盘
        testCreateDefaultChessboard();

        // 2. 获取初始状态
        testGetChessboardWithStatus();

        // 3. 黑子落子
        testUpdateChessbyPosition();

        // 4. 查看状态
        testGetChessboardWithStatus();

        // 5. 白子落子
        try {
            String json = "{ \"x\": 1, \"y\": 1, \"player\": -1 }";

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(new URI(BASE_URL + API_PATH + "/0"))
                    .header("Content-Type", "application/json")
                    .PUT(HttpRequest.BodyPublishers.ofString(json))
                    .build();

            HttpResponse<String> response = httpClient.send(request,
                    HttpResponse.BodyHandlers.ofString());

            System.out.println("白子落子:");
            System.out.println("Status: " + response.statusCode());
            System.out.println("Response: " + response.body());
            System.out.println("----------------------------------------");

        } catch (URISyntaxException | IOException | InterruptedException e) {
            System.err.println("白子落子失败: " + e.getMessage());
        }

        // 6. 查看最终状态
        testGetChessboardWithStatus();

        // 7. 清理：删除棋盘
        testDeleteChessboard();
    }

    /**
     * 主函数，运行所有测试
     */
    public static void main(String[] args) {
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
        GomokuApiTest test = new GomokuApiTest();

        System.out.println("五子棋API测试开始...");
        System.out.println("========================================");

        // 运行完整流程测试
        test.testCompleteGameFlow();

        System.out.println("\n========== 单独功能测试 ==========");

        // 运行单独功能测试
        test.testCreateCustomChessboard();
        test.testCreateChessboardWithData();
        test.testUpdateChessboardWithData();

        System.out.println("所有测试完成!");
    }
}

