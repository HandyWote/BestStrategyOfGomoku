package com.gomoku;

import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/gomoku")
public class GameController {
    private final Map<Integer, Gomoku> gomokus = new HashMap<>();

    @GetMapping("/{id}")
    public Object getChessboard(@PathVariable int id, @RequestParam(required = false) boolean showStatus) { //showStatus参数用于控制棋盘获取成功时是否显示游戏状态，默认为false
        if(gomokus.containsKey(id)) {
            Gomoku gomoku = gomokus.get(id);
            LinkedHashMap<String, Object> response = new LinkedHashMap<>();
            if(showStatus){
                response.put("code", 0);
                response.put("nextPlayer", gomoku.checkPlayerNow());
                response.put("isGameOver", gomoku.isGameOver());
                if(gomoku.isGameOver()) response.put("winner", gomoku.getWinner());
            }
            response.put("board", gomoku.getChessboard());
            return response;
        }else {
            return createResponse(-1, "棋盘不存在");
        }
    }

    @PostMapping
    public Object createChessboard(@RequestBody GomokuRequest body) {
        if(body.containId()) {
            int id =  body.getId();
            if (gomokus.containsKey(id)) {
                return createResponse(-1, "棋盘已存在");
            }
            if(body.containBoard()){
                int[][] board =  body.getBoard();
                Gomoku gomoku = new Gomoku(board);
                gomokus.put(id, gomoku);
                return createResponse(0, "棋盘创建成功");
            }else if(body.containX() && body.containY()) {
                int x = body.getX();
                int y = body.getX();
                Gomoku gomoku = new Gomoku(x, y);
                gomokus.put(id, gomoku);
                return createResponse(0, "棋盘创建成功");
            }else{
                Gomoku gomoku = new Gomoku();
                gomokus.put(id, gomoku);
                return createResponse(0, "棋盘创建成功");
            }
        }else{
            return createResponse(-1, "请求体中缺少棋盘ID");
        }
    }

    @PutMapping("/{id}")
    public Object updateChessboard(@PathVariable int id, @RequestBody GomokuRequest body) {
        Gomoku gomoku = gomokus.get(id);
        if (!gomokus.containsKey(id)) {
            return createResponse(-1, "棋盘不存在，无法更新");
        }else if(body.containBoard()) {
            int[][] newBoard = body.getBoard();
            int[][] board = gomoku.getChessboard();
            int x=-1,y=-1,count=0; // 记录修改的坐标和修改次数
            for (int i = 0; i < board.length; i++) {
                for (int j = 0; j < board[i].length; j++) {
                    if(board[i][j] != newBoard[i][j]) {
                        x = i;
                        y = j;
                        count++;
                    }
                }
            }
            if (count != 1) {
                return createResponse(-1,"更新失败，棋盘修改超过2处或少于1处");
            }else{
                if (!gomoku.updateChess(x, y, newBoard[x][y])){
                    return createResponse(-1,"更新失败，可能是因为位置已被占用或坐标不合法");
                }else {
                    gomokus.put(id, gomoku);
                    return createResponse(0, "棋盘更新成功");
                }
            }
        }else{
            int x = body.getX();
            int y = body.getY();
            int player = body.getPlayer(); // 默认玩家为1,即黑子
            if (!gomoku.updateChess(x, y, player)) {
                return createResponse(-1,"更新失败，可能是因为位置已被占用或坐标不合法");
            }else{
                gomokus.put(id, gomoku);
                return createResponse(0, "棋盘更新成功");
            }
        }
    }

    @DeleteMapping("/{id}")
    public Object deleteChessboard(@PathVariable int id) {
        if (gomokus.containsKey(id)) {
            gomokus.remove(id);
            return createResponse(0, "棋盘删除成功");
        }else{
            return createResponse(-1, "棋盘不存在");
        }
    }

    private Object createResponse(int code, String msg) {
        LinkedHashMap<String, Object> response = new LinkedHashMap<>();
        response.put("code", code);
        response.put("msg", msg);
        return response;
    }

}