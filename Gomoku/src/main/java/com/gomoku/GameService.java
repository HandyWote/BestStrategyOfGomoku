package com.gomoku;

import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class GameService {
    private final Map<Integer, Gomoku> gomokus = new HashMap<>();

    /**
     * 获取指定ID的Gomoku游戏实例
     * @param id 游戏ID
     * @return Gomoku实例，如果不存在则返回null
     */
    public Gomoku getGomoku(int id) {
        return gomokus.get(id);
    }

    /**
     * 保存Gomoku游戏实例
     * @param id 游戏ID
     * @param gomoku Gomoku实例
     */
    public void saveGomoku(int id, Gomoku gomoku) {
        gomokus.put(id, gomoku);
    }

    /**
     * 检查指定ID的游戏是否存在
     * @param id 游戏ID
     * @return 如果存在返回true，否则返回false
     */
    public boolean exists(int id) {
        return gomokus.containsKey(id);
    }

    /**
     * 删除指定ID的游戏
     * @param id 游戏ID
     * @return 如果删除成功返回true，否则返回false
     */
    public boolean removeGomoku(int id) {
        return gomokus.remove(id) != null;
    }

    /**
     * 获取所有游戏的数量
     * @return 游戏总数
     */
    public int getGameCount() {
        return gomokus.size();
    }

    /**
     * 清空所有游戏
     */
    public void clearAll() {
        gomokus.clear();
    }
}
