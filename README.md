# 五子棋最优策略项目

## 项目简介

本项目致力于实现**9x9五子棋**的最优策略AI，支持多种算法并行对比，具备完整的前后端系统和标准化文档体系。项目已结项，适合算法研究、AI对战、教学演示等多场景。

## 项目结构

- **前端**：HTML/CSS/JavaScript 实现棋盘交互与对弈界面
- **后端**：Java SpringBoot 提供RESTful API，管理棋局与对弈逻辑
- **AI算法**：多语言混合（Python/C#/Java），实现五种主流AI策略
    - 启发式算法（C#）
    - 深度学习（PyTorch）
    - 当前局面到必胜推导（Python）
    - 必胜局面逆推（Java）
    - 大模型API方案（Python/Prompt）

## 主要功能

- 支持9x9棋盘的五子棋对弈与AI自动落子
- 多算法切换与对比，支持启发式、深度学习、必胜推导等
- RESTful API接口，前后端分离，便于集成和扩展
- 支持在线体验：[gomoku.handywote.site](https://gomoku.handywote.site)
- 完善的开发文档与算法报告

## 技术栈

- **前端**：HTML5、CSS3、JavaScript
- **后端**：Java 17、SpringBoot 3.x
- **AI算法**：
    - Python 3.x（PyTorch、Numpy）
    - C#（.NET 6/8，Newtonsoft.Json）
    - Java
- **协作与文档**：Gitea+Git、Markdown

## 运行与使用

### 1. 克隆项目

```bash
git clone https://git.handywote.site/BestGameStrategyGroup/BestStrategyOfGomoku.git
```

### 2. 后端服务启动

- 进入后端目录（如 `Gomoku`），使用Gradle或IDE运行SpringBoot主程序
- 或用Docker Compose一键部署（如有docker-compose.yml）

### 3. 前端访问

- 访问本地或服务器部署的前端页面，或直接体验[在线演示](https://gomoku.handywote.site)

### 4. 算法模块使用

#### 启发式算法（C#）

- 进入 `Heuristics` 目录，编译并运行 `Test.cs` 或相关主程序
- 支持与网页API对接，自动获取棋盘数据并落子

#### 深度学习（Python）

- 进入 `DeepLearning` 目录
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```
- 训练模型：
  ```bash
  python main.py --train
  ```
- AI对战：
  ```bash
  python main.py --play --model gomoku_model_final.pth
  ```

#### 当前到必胜推导（Python）

- 进入 `CurrentToWin` 目录
- 静态模式：运行 `main.py`，输入棋盘JSON或文件，输出最优落子
- 动态模式：运行 `Web_main.py`，输入API地址和棋局ID，支持与网页联动

#### 必胜局面逆推（Java）

- 进入 `WinToCurrent` 目录
- 编译并运行 `GomokuAiRunner.java`，按提示输入棋盘ID和角色（black/white）

### 5. API接口示例

- 创建棋盘：`POST /api/gomoku`
- 获取棋盘：`GET /api/gomoku/{id}`
- 落子操作：`PUT /api/gomoku/{id}`

详见 `/实践报告.md` 和各模块 `使用说明.md`。

## 参考文档

- [实践报告.md](./实践报告.md)：完整项目总结与技术细节
- [深度学习报告.md](./DeepLearning/深度学习报告.md)
- [启发式实验报告.md](./Heuristics/启发式实验报告.md)
- [现在到必胜算法报告.md](./CurrentToWin/现在到必胜算法报告.md)
- [五子棋必胜局面到当前局面算法.md](./WinToCurrent/五子棋必胜局面到当前局面算法.md)

---

如需二次开发或算法集成，建议先阅读各模块的说明文档和代码注释。欢迎参考本项目进行五子棋AI相关研究与实践！
