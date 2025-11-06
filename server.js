// 简易评论后端：Node + Express + 文件存储
const path = require('path')
const fs = require('fs')
const express = require('express')

const app = express()
const PORT = process.env.PORT || 8080
const HOST = process.env.HOST || '127.0.0.1'
const DATA_DIR = path.join(__dirname, 'data')
const DB_FILE = path.join(DATA_DIR, 'comments.db.json')

app.use(express.json({ limit: '256kb' }))

// CORS（本地开发用）
app.use((req,res,next)=>{
  res.setHeader('Access-Control-Allow-Origin','*')
  res.setHeader('Access-Control-Allow-Methods','GET,POST,DELETE,OPTIONS')
  res.setHeader('Access-Control-Allow-Headers','Content-Type')
  if(req.method==='OPTIONS') return res.sendStatus(204)
  next()
})

// 静态资源
app.use(express.static(__dirname))

function ensureDb(){
  if(!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR)
  if(!fs.existsSync(DB_FILE)) fs.writeFileSync(DB_FILE, JSON.stringify({}), 'utf8')
}
function readDb(){
  ensureDb()
  try{
    return JSON.parse(fs.readFileSync(DB_FILE,'utf8')||'{}')
  }catch{
    return {}
  }
}
function writeDb(db){
  ensureDb()
  fs.writeFileSync(DB_FILE, JSON.stringify(db, null, 2), 'utf8')
}
function generateId(){
  return `${Date.now()}_${Math.random().toString(36).slice(2,8)}`
}

app.get('/api/health', (req,res)=>{
  res.json({ ok: true })
})

// 获取某篇文章的评论
app.get('/api/comments/:slug', (req,res)=>{
  const slug = String(req.params.slug||'')
  const db = readDb()
  const list = Array.isArray(db[slug]) ? db[slug] : []
  res.json(list)
})

// 新增评论
app.post('/api/comments/:slug', (req,res)=>{
  const slug = String(req.params.slug||'')
  const { author, content } = req.body || {}
  const safeAuthor = String((author||'').trim()||'匿名').slice(0,50)
  const safeContent = String((content||'').trim()).slice(0,2000)
  if(!safeContent){ return res.status(400).json({ error: '内容不能为空' }) }
  const db = readDb()
  const list = Array.isArray(db[slug]) ? db[slug] : []
  const item = { id: generateId(), author: safeAuthor, content: safeContent, createdAt: Date.now(), likes: 0 }
  list.push(item)
  db[slug] = list
  writeDb(db)
  res.status(201).json(item)
})

// 点赞
app.post('/api/comments/:slug/:id/like', (req,res)=>{
  const slug = String(req.params.slug||'')
  const id = String(req.params.id||'')
  const db = readDb()
  const list = Array.isArray(db[slug]) ? db[slug] : []
  const idx = list.findIndex(x=> String(x.id)===id)
  if(idx<0) return res.status(404).json({ error: '未找到评论' })
  list[idx].likes = (list[idx].likes||0)+1
  db[slug] = list
  writeDb(db)
  res.json({ likes: list[idx].likes })
})

// 删除
app.delete('/api/comments/:slug/:id', (req,res)=>{
  const slug = String(req.params.slug||'')
  const id = String(req.params.id||'')
  const db = readDb()
  const list = Array.isArray(db[slug]) ? db[slug] : []
  const idx = list.findIndex(x=> String(x.id)===id)
  if(idx<0) return res.status(404).json({ error: '未找到评论' })
  list.splice(idx,1)
  db[slug] = list
  writeDb(db)
  res.json({ ok: true })
})

app.listen(PORT, HOST, ()=>{
  console.log(`Server running at http://${HOST}:${PORT}/`)
})


