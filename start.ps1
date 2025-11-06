param(
  [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$node = Get-Command node -ErrorAction SilentlyContinue
$py = Get-Command py -ErrorAction SilentlyContinue
if(-not $py){ $py = Get-Command python -ErrorAction SilentlyContinue }

Start-Process "http://127.0.0.1:$Port/"
if($node){
  Write-Host "检测到 Node.js，使用 Node 启动后端与静态文件: http://127.0.0.1:$Port/" -ForegroundColor Green
  Write-Host "若首次运行，请先执行 npm install" -ForegroundColor Yellow
  node .\server.js
} elseif($py) {
  Write-Host "未检测到 Node.js，回退为 Python 静态服务（评论使用本地存储）: http://127.0.0.1:$Port/" -ForegroundColor Green
  Write-Host "停止：在此窗口按 Ctrl+C" -ForegroundColor Yellow
  & $py.Source -m http.server $Port --bind 127.0.0.1
} else {
  Write-Host "未找到 Node 或 Python，无法启动服务器" -ForegroundColor Red
}


