package com.gomoku;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * 五子棋AI运行器 - 提供预设配置快速启动
 */
public class GomokuAiRunner {

    /**
     * 启动黑子AI
     */
    public static void startBlackAi(int boardId) {
        GomokuAiClient client = new GomokuAiClient();
        client.configure(boardId, 1, 2); // 黑子，2秒轮询
        client.start();

        System.out.println("黑子AI已启动，棋盘ID: " + boardId);
        System.out.println("按回车键停止...");

        try {
            System.in.read();
        } catch (Exception e) {
            // 忽略
        }

        client.stop();
    }

    /**
     * 启动白子AI
     */
    public static void startWhiteAi(int boardId) {
        GomokuAiClient client = new GomokuAiClient();
        client.configure(boardId, -1, 2); // 白子，2秒轮询
        client.start();

        System.out.println("白子AI已启动，棋盘ID: " + boardId);
        System.out.println("按回车键停止...");

        try {
            System.in.read();
        } catch (Exception e) {
            // 忽略
        }

        client.stop();
    }

    /**
     * 演示用法
     */
    public static void main(String[] args) {
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
        if (args.length >= 2) {
            int boardId = Integer.parseInt(args[0]);
            String role = args[1].toLowerCase();

            if ("black".equals(role) || "1".equals(role)) {
                startBlackAi(boardId);
            } else if ("white".equals(role) || "-1".equals(role)) {
                startWhiteAi(boardId);
            } else {
                System.err.println("无效角色，请使用 'black' 或 'white'");
            }
        } else {
            System.out.println("用法: java GomokuAiRunner <棋盘ID> <角色>");
            System.out.println("角色: black/1 (黑子) 或 white/-1 (白子)");
            System.out.println("示例: java GomokuAiRunner 1 black");

            // 默认启动黑子AI，棋盘ID为1
            startBlackAi(1);
        }
    }
}
