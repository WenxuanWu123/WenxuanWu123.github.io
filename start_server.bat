@echo off
setlocal
cd /d %~dp0
rem 尝试使用 py，否则退回到 python
where node >nul 2>nul && (set "HAS_NODE=1") || (set "HAS_NODE=")
where py >nul 2>nul && (set "PY=py") || (set "PY=python")

rem 打开浏览器
start "" http://127.0.0.1:8080/

if defined HAS_NODE (
  echo 检测到 Node.js，使用 Node 启动后端与静态文件：http://127.0.0.1:8080/
  echo 首次运行需执行：npm install
  node server.js
) else (
  echo 未检测到 Node.js，回退為 Python 静态服务（评论使用本地存储）：http://127.0.0.1:8080/
  echo 按 Ctrl+C 停止服务。
  %PY% -m http.server 8080 --bind 127.0.0.1
)
endlocal


