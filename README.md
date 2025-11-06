# 我的文章网站

一个零后端、纯静态的文章网站（可选启用轻量后端用于评论跨设备保存）。支持：

- 文章列表、搜索
- 基于 hash 的前端路由
- Markdown 渲染（marked）与 XSS 保护（DOMPurify）
- 响应式与暗色模式

## 使用

1. 直接用浏览器打开 `index.html` 即可。若浏览器阻止本地文件 `fetch`，建议启一个本地 HTTP 服务：
   - PowerShell:
     ```powershell
     npx http-server -p 8080 --no-cache
     ```
     然后访问 `http://localhost:8080/`。

2. Windows 一键脚本（推荐）：
   - PowerShell 运行 `start.ps1`，或 CMD 双击 `start_server.bat`
   - 若检测到 Node.js，会自动以 Node 启动后端和静态文件：
     ```bash
     npm install
     npm start
     ```
     评论将保存在 `data/comments.db.json`，跨浏览器/设备有效（同机）。
   - 若无 Node.js，将回退为 Python 静态服务，评论仅存于浏览器 localStorage（仅本机）。

2. 搜索：在右上角输入关键词，按标题/标签/摘要过滤。

## 新增文章

1. 在 `posts/` 目录新增一个 Markdown 文件，例如：`posts/my-post.md`。
2. 打开 `posts/posts.json`，按如下结构新增一项：
   ```json
   {
     "slug": "my-post",
     "title": "我的新文章",
     "date": "2025-11-04",
     "tags": ["随笔", "生活"],
     "summary": "一句话摘要。",
     "file": "./posts/my-post.md"
   }
   ```
   - `slug`: URL 中使用的唯一标识（与路由 `#/post/slug` 对应）
   - `date`: 建议使用 `YYYY-MM-DD`，会用于排序与显示
   - `file`: 指向你刚创建的 Markdown 文件路径

3. 刷新页面即可在列表看到新文章，点击进入阅读。

## 自定义

- 标题与站点名：修改 `index.html` 中的 `<title>` 与页头文字。
- 样式：编辑 `styles.css`。
- 行为：编辑 `script.js`（列表渲染、搜索、路由、渲染逻辑、评论存储）。

## 启用 Giscus（GitHub Discussions 评论）

1. 打开 `https://giscus.app/`，登录后选择你的仓库（需开启 Discussions，例如 `WenxuanWu123/WenxuanWu123.github.io`）。
2. 选择一个分类（如 `General`），在页面右侧会显示 `repo-id` 与 `category-id`。
3. 将这些值填入 `script.js` 中的 `GISCUS_CONFIG`：
   - `repo`: `用户名/仓库名`
   - `repoId`: 从 giscus.app 复制
   - `category`: 选择的分类名
   - `categoryId`: 从 giscus.app 复制
4. 提交并部署后，文章页底部将显示 Giscus 评论。我们按文章 slug 作为讨论话题（不同文章各有独立讨论）。


## 许可

个人项目，随意修改与分发。


