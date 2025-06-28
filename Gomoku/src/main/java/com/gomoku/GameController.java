package com.gomoku;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/gomoku")
public class GameController {

    private final GameService gameService;

    public GameController(GameService gameService) {
        this.gameService = gameService;
    }

    /**
     * 获取棋盘信息
     * @param id 棋盘ID
     * @param showStatus 是否显示游戏状态，默认为false
     * @return 棋盘信息或错误信息
     */
    @GetMapping("/{id}")
    public ResponseEntity<Map<String, Object>> getChessboard(
            @PathVariable int id,
            @RequestParam(required = false, defaultValue = "false") boolean showStatus) {
        try {
            Gomoku gomoku = gameService.getGomoku(id);
            if (gomoku == null) {
                return ResponseEntity.ok(createErrorResponse("棋盘不存在"));
            }
            Map<String, Object> response = createSuccessResponse();
            if (showStatus) {
                response.put("nextPlayer", gomoku.checkPlayerNow());
                response.put("isGameOver", gomoku.isGameOver());
                if (gomoku.isGameOver()) {
                    response.put("winner", gomoku.getWinner());
                }
            }
            response.put("board", gomoku.getChessboard());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.ok(createErrorResponse("获取棋盘失败: " + e.getMessage()));
        }
    }

    /**
     * 创建新棋盘
     * @param body 请求体
     * @return 创建结果
     */
    @PostMapping
    public ResponseEntity<Map<String, Object>> createChessboard(@RequestBody GomokuRequest body) {
        try {
            if (!body.containId()) {
                return ResponseEntity.ok(createErrorResponse("请求体中缺少棋盘ID"));
            }
            int id = body.getId();
            if (gameService.exists(id)) {
                return ResponseEntity.ok(createErrorResponse("棋盘已存在"));
            }
            Gomoku gomoku = createGomokuFromRequest(body);
            gameService.saveGomoku(id, gomoku);

            return ResponseEntity.ok(createSuccessResponse("棋盘创建成功"));
        } catch (Exception e) {
            return ResponseEntity.ok(createErrorResponse("创建棋盘失败: " + e.getMessage()));
        }
    }

    /**
     * 更新棋盘
     * @param id 棋盘ID
     * @param body 更新内容
     * @return 更新结果
     */
    @PutMapping("/{id}")
    public ResponseEntity<Map<String, Object>> updateChessboard(
            @PathVariable int id,
            @RequestBody GomokuRequest body) {
        try {
            Gomoku gomoku = gameService.getGomoku(id);
            if (gomoku == null) {
                return ResponseEntity.ok(createErrorResponse("棋盘不存在，无法更新"));
            }
            boolean updateResult;
            if (body.containBoard()) { // 如果请求体中包含棋盘数组
                updateResult = gomoku.updateChess(body.getBoard());
            } else {
                int x = body.getX();
                int y = body.getY();
                int player = body.getPlayer();
                updateResult = gomoku.updateChess(x, y, player);
            }
            if (!updateResult) {
                return ResponseEntity.ok(createErrorResponse("更新棋盘失败:" + gomoku.getWrongMessage()));
            }
            gameService.saveGomoku(id, gomoku);
            return ResponseEntity.ok(createSuccessResponse("棋盘更新成功"));

        } catch (Exception e) {
            return ResponseEntity.ok(createErrorResponse("更新棋盘失败: " + e.getMessage()));
        }
    }

    /**
     * 删除棋盘
     * @param id 棋盘ID
     * @return 删除结果
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Map<String, Object>> deleteChessboard(@PathVariable int id) {
        try {
            if (!gameService.exists(id)) {
                return ResponseEntity.ok(createErrorResponse("棋盘不存在"));
            }
            gameService.removeGomoku(id);
            return ResponseEntity.ok(createSuccessResponse("棋盘删除成功"));
        } catch (Exception e) {
            return ResponseEntity.ok(createErrorResponse("删除棋盘失败: " + e.getMessage()));
        }
    }

    /**
     * 根据请求体创建五子棋实例
     */
    private Gomoku createGomokuFromRequest(GomokuRequest body) {
        if (body.containBoard()) { // 如果请求体中包含棋盘数组
            return new Gomoku(body.getBoard());
        } else if (body.containX() && body.containY()) { // 如果请求体中包含坐标x,y
            return new Gomoku(body.getX(), body.getY());
        } else { // 创建一个默认9x9的棋盘
            return new Gomoku();
        }
    }

    /**
     * 创建成功响应
     */
    private Map<String, Object> createSuccessResponse() {
        return createSuccessResponse(null);
    }

    /**
     * 创建成功响应
     */
    private Map<String, Object> createSuccessResponse(String message) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("code", 0);
        if (message != null) {
            response.put("msg", message);
        }
        return response;
    }

    /**
     * 创建错误响应
     */
    private Map<String, Object> createErrorResponse(String message) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("code", -1);
        response.put("msg", message);
        return response;
    }

}