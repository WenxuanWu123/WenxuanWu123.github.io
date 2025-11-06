const els = {
  year: document.getElementById('year'),
  listView: document.getElementById('listView'),
  postView: document.getElementById('postView'),
  postList: document.getElementById('postList'),
  emptyState: document.getElementById('emptyState'),
  searchInput: document.getElementById('searchInput'),
  themeToggle: document.getElementById('themeToggle'),
  postTitle: document.getElementById('postTitle'),
  postMeta: document.getElementById('postMeta'),
  postContent: document.getElementById('postContent'),
  giscusSection: document.getElementById('giscusSection'),
  giscusContainer: document.getElementById('giscusContainer'),
  giscusHint: document.getElementById('giscusHint')
}

els.year.textContent = new Date().getFullYear()

// 主题切换
const THEME_KEY = 'site-theme'
function applyTheme(theme){
  if(theme === 'dark'){
    document.documentElement.classList.add('theme-dark')
  }else{
    document.documentElement.classList.remove('theme-dark')
  }
}
applyTheme(localStorage.getItem(THEME_KEY)||'')
els.themeToggle.addEventListener('click',()=>{
  const next = document.documentElement.classList.contains('theme-dark') ? '' : 'dark'
  localStorage.setItem(THEME_KEY,next)
  applyTheme(next)
})

// 数据加载
let allPosts = []
async function loadPosts(){
  try{
    const res = await fetch('./posts/posts.json', { cache: 'no-cache' })
    if(!res.ok) throw new Error('无法加载 posts.json')
    const data = await res.json()
    // 按日期倒序
    allPosts = (data||[]).sort((a,b)=> (b.date||'').localeCompare(a.date||''))
  }catch(err){
    console.error(err)
    allPosts = []
  }
}

function renderList(filterText=''){
  const q = filterText.trim().toLowerCase()
  const items = !q ? allPosts : allPosts.filter(p=>{
    const hay = [p.title,p.summary,(p.tags||[]).join(',')].join(' ').toLowerCase()
    return hay.includes(q)
  })
  els.postList.innerHTML = ''
  els.emptyState.hidden = items.length>0
  for(const p of items){
    const li = document.createElement('li')
    li.innerHTML = `
      <a href="#/post/${p.slug}">${escapeHtml(p.title)}</a>
      <div class="meta">${formatDate(p.date)} · ${(p.tags||[]).map(t=>`<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>
      ${p.summary?`<div class="summary">${escapeHtml(p.summary)}</div>`:''}
    `
    els.postList.appendChild(li)
  }
}

function formatDate(s){
  if(!s) return ''
  try{const d = new Date(s); if(!isNaN(d)) return d.toISOString().slice(0,10)}catch(_){}
  return s
}

function escapeHtml(str){
  return String(str||'').replace(/[&<>"]|'/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[ch]))
}

// 前端路由
window.addEventListener('hashchange', handleRoute)
let USE_BACKEND = false
window.addEventListener('DOMContentLoaded', async ()=>{
  await loadPosts()
  bindSearch()
  handleRoute()
})

function bindSearch(){
  els.searchInput.addEventListener('input', ()=>{
    if(location.hash === '' || location.hash === '#/' ){
      renderList(els.searchInput.value)
    }
  })
}

function handleRoute(){
  const hash = location.hash || '#/'
  const match = hash.match(/^#\/post\/([^?#]+)/)
  if(match){
    const slug = decodeURIComponent(match[1])
    showPost(slug)
  }else{
    showList()
  }
}

function showList(){
  els.postView.hidden = true
  els.listView.hidden = false
  renderList(els.searchInput.value)
}

async function showPost(slug){
  const post = allPosts.find(p=>p.slug===slug)
  if(!post){
    els.postTitle.textContent = '未找到文章'
    els.postMeta.textContent = ''
    els.postContent.innerHTML = ''
    els.listView.hidden = true
    els.postView.hidden = false
    return
  }
  els.postTitle.textContent = post.title
  // 同步页面标题以利于第三方组件按标题识别
  try{ document.title = `${post.title} - 我的文章` }catch(_){ }
  els.postMeta.innerHTML = `${formatDate(post.date)} · ${(post.tags||[]).map(t=>`<span class="tag">${escapeHtml(t)}</span>`).join('')}`
  els.listView.hidden = true
  els.postView.hidden = false
  try{
    const res = await fetch(post.file, { cache: 'no-cache' })
    if(!res.ok) throw new Error('加载文章失败')
    const md = await res.text()
    // 使用 marked 渲染 + DOMPurify 清理
    const html = DOMPurify.sanitize(marked.parse(md))
    els.postContent.innerHTML = html
  }catch(err){
    console.error(err)
    els.postContent.textContent = '内容加载失败'
  }

  // 动态加载 Giscus（基于 slug 作为话题 term）
  renderGiscus({ slug, title: post.title })
}

function formatDateTime(ts){
  try{
    const d = new Date(ts)
    if(!isNaN(d)){
      const yyyy = d.getFullYear()
      const mm = String(d.getMonth()+1).padStart(2,'0')
      const dd = String(d.getDate()).padStart(2,'0')
      const hh = String(d.getHours()).padStart(2,'0')
      const mi = String(d.getMinutes()).padStart(2,'0')
      return `${yyyy}-${mm}-${dd} ${hh}:${mi}`
    }
  }catch(_){ }
  return ''
}


// ======= Giscus 动态渲染 =======
const GISCUS_CONFIG = {
  // 必填：你的仓库，例如 'WenxuanWu123/WenxuanWu123.github.io'
  repo: 'WenxuanWu123/WenxuanWu123.github.io',
  // 必填：在 giscus.app 复制得到的 repo-id 与 category/category-id
  repoId: 'R_kgDOQPBL6w',        // e.g. 'R_kgDO...'
  category: 'General',
  categoryId: 'DIC_kwDOQPBL684CxfqA'     // e.g. 'DIC_kwDO...'
}

function renderGiscus({ slug, title }){
  if(!els.giscusSection || !els.giscusContainer) return
  // 清空旧的 iframe/script
  els.giscusContainer.innerHTML = ''
  const hasConfig = !!(GISCUS_CONFIG.repo && GISCUS_CONFIG.repoId && GISCUS_CONFIG.category && GISCUS_CONFIG.categoryId)
  els.giscusHint.hidden = hasConfig
  if(!hasConfig) return
  const script = document.createElement('script')
  script.src = 'https://giscus.app/client.js'
  script.async = true
  script.crossOrigin = 'anonymous'
  script.setAttribute('data-repo', GISCUS_CONFIG.repo)
  script.setAttribute('data-repo-id', GISCUS_CONFIG.repoId)
  script.setAttribute('data-category', GISCUS_CONFIG.category)
  script.setAttribute('data-category-id', GISCUS_CONFIG.categoryId)
  // 使用 slug 作为特定话题 term
  script.setAttribute('data-mapping', 'specific')
  script.setAttribute('data-term', slug)
  script.setAttribute('data-reactions-enabled', '1')
  script.setAttribute('data-emit-metadata', '0')
  script.setAttribute('data-input-position', 'bottom')
  script.setAttribute('data-theme', document.documentElement.classList.contains('theme-dark') ? 'dark' : 'light')
  script.setAttribute('data-lang', 'zh-CN')
  els.giscusContainer.appendChild(script)
}
