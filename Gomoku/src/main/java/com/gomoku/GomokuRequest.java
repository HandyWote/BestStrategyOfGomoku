// GomokuRequest.java
package com.gomoku;

public class GomokuRequest {
    private Integer id;
    private int[][] board;
    private Integer x;
    private Integer y;
    private Integer player;

    public boolean containId() {
        return id != null;
    }

    public Integer getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public boolean containBoard() {
        return board != null;
    }

    public int[][] getBoard() {
        return board;
    }

    public void setBoard(int[][] board) {
        this.board = board;
    }

    public boolean containX() {
        return x != null;
    }

    public Integer getX() {
        return x;
    }

    public void setX(int x) {
        this.x = x;
    }

    public boolean containY() {
        return y != null;
    }

    public Integer getY() {
        return y;
    }

    public void setY(int y) {
        this.y = y;
    }

    public boolean containPlayer() {
        return player != null;
    }

    public Integer getPlayer() {
        return player;
    }

    public void setPlayer(int player) {
        this.player = player;
    }
}