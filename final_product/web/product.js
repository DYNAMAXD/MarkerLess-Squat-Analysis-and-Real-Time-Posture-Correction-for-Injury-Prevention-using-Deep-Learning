// product.js -- the final product page (final_product/product_app.py). Views: #/ new analysis, #/session/<id>,
// #/history (camera calibration lives in the new-analysis form). The 3D body is played with three.js r128 (same orbit convention as the mesh viewer and the
// saliency painter: theta = azimuth, 0 = in front of the body; phi = elevation).
'use strict';

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const tkey = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const store = {get(k){ try{ return localStorage.getItem(k); }catch(e){ return null; } }, set(k, v){ try{ localStorage.setItem(k, v); }catch(e){} }};
// Static viewer mode (?viewer=<URL of viewer_index.json>): a finished session's results read from plain files (the
// ZeroGPU Space, hf_space/app.py, serves them; there is no server API). The /api/... URLs are mapped onto those files
// and everything that would change data is off.
const VIEWER = new URLSearchParams(location.search).get('viewer');
let IDX = null;
function U(url){
  if (!IDX) return url;
  let m;
  if (url === '/api/content') return IDX.content;
  if (url === '/api/faces.bin') return IDX.faces;
  if ((m = url.match(/^\/api\/templates\/(.+)$/))) return IDX.templates[decodeURIComponent(m[1])];
  if ((m = url.match(/^\/api\/sessions\/([^/]+)\/(data|download|reps\.csv|file\/(.+))$/))){
    const base = IDX.root + decodeURIComponent(m[1]) + '/';
    return base + ({data: 'viewer_data.json', download: 'download/results.zip', 'reps.csv': 'download/reps.csv'}[m[2]] || m[3]);
  }
  return url;
}
async function viewerApi(url, opt){
  if (opt && opt.method && opt.method !== 'GET') throw new Error('Not available in this viewer.');
  if (url === '/api/info') return IDX.info;
  if (url === '/api/templates') return {templates: Object.fromEntries(Object.keys(IDX.templates).map(k => [k, k]))};
  if (url === '/api/cameras') return [];
  const list = async () => (await fetch(IDX.sessions, {cache: 'no-store'})).json();
  if (url === '/api/sessions') return list();
  const m = url.match(/^\/api\/sessions\/([^/]+)$/);
  if (m){ const row = (await list()).find(r => r.id === decodeURIComponent(m[1])); if (!row) throw new Error('Session not found'); return row; }
  const r = await fetch(U(url));
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
async function api(url, opt){
  if (IDX) return viewerApi(url, opt);
  const r = await fetch(url, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `${r.status} ${r.statusText}`);
  return j;
}
const post = (url, body) => api(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body || {})});
function upload(url, form, bar){
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open('POST', url);
    bar.classList.remove('hidden'); bar.firstElementChild.style.width = '0';
    x.upload.onprogress = e => { if (e.lengthComputable) bar.firstElementChild.style.width = (100 * e.loaded / e.total).toFixed(1) + '%'; };
    x.onload = () => { bar.classList.add('hidden'); let j = {}; try{ j = JSON.parse(x.responseText); }catch(e){}
      x.status < 300 ? resolve(j) : reject(new Error(j.error || `${x.status} ${x.statusText}`)); };
    x.onerror = () => { bar.classList.add('hidden'); reject(new Error('Upload failed (is the app still running?)')); };
    x.send(form);
  });
}
const NV = 6890;
const BASE = [0.74, 0.76, 0.80];   // grey mannequin
let C = null;                        // product_content.json
let INFO = {};
let TPL = {};                        // template key -> file name
const tplCache = {};                 // template key -> Float32Array 0..1
const errInfo = c => (C && C.errors[c]) || {name: 'Not analysed', short: '', color: '#8d94b0', fix: [], what: '', why: ''};
const STAGE_LABEL = {check: 'Checking the videos', sync: 'Finding the clap', calibration: 'Calibrating the cameras',
  extract: 'Cutting the videos', pose: 'Finding the body keypoints', triangulation: 'Building the 3D skeleton',
  angles: 'Joint angles and reps', mesh: 'Building the 3D body', model: 'Detecting errors', finish: 'Finishing',
  intrinsic: 'Intrinsic calibration', extrinsic: 'Fallback calibration'};
const STATE_LABEL = {queued: 'Waiting', running: 'Running', done: 'Done', error: 'Failed', interrupted: 'Interrupted', cancelled: 'Stopped'};
function stateChip(state, extra){
  return `<span class="state ${esc(state)}"><span class="dot"></span>${esc(STATE_LABEL[state] || state || '')}${extra ? ' · ' + esc(extra) : ''}</span>`;
}
const fmtDate = s => { if (!s) return ''; const d = new Date(s.replace(' ', 'T')); return isNaN(d) ? s : d.toLocaleString(undefined, {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'}); };
const fmtDay = s => { const d = new Date((s || '').replace(' ', 'T')); return isNaN(d) ? '' : d.toLocaleDateString(undefined, {day: 'numeric', month: 'short'}); };

// action buttons (class "ub"): label + an icon block that slides over it on hover; data-ic names the icon
const ICON = {
  delete: 'M24 20.188l-8.315-8.209 8.2-8.282-3.697-3.697-8.212 8.318-8.31-8.203-3.666 3.666 8.321 8.24-8.206 8.313 3.666 3.666 8.237-8.318 8.285 8.203z',
  download: 'M10 2h4v9h4.5L12 17.5 5.5 11H10zM3 20h18v3H3z',
  add: 'M10.5 3h3v7.5H21v3h-7.5V21h-3v-7.5H3v-3h7.5z',
  play: 'M6 3l15 9-15 9z',
  stop: 'M5 5h14v14H5z',
  retry: 'M17.65 6.35A7.96 7.96 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4z',
  compare: 'M3 4h7v16H3zm11 0h7v16h-7zM11.5 2h1v20h-1z',
  video: 'M17 10.5V7a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-3.5l4 4v-11z',
  file: 'M6 2h8l6 6v14H6zm7 1.5V9h5.5zM8.5 13h9v1.6h-9zm0 3.5h9v1.6h-9z',
  check: 'M9 16.2l-4.2-4.2-1.4 1.4L9 19 21 7l-1.4-1.4z',
};
const svgIcon = (name, fill) => `<svg viewBox="0 0 24 24" aria-hidden="true"${fill ? ` fill="${fill}"` : ''}><path d="${ICON[name] || ICON.add}"></path></svg>`;
function makeUb(b){
  if (b.querySelector('.ub-t')) return;
  const label = b.textContent.trim();
  b.innerHTML = `<span class="ub-t">${esc(label)}</span><span class="ub-i">${svgIcon(b.dataset.ic)}</span>`;
  if (!b.getAttribute('aria-label')) b.setAttribute('aria-label', label);
}
const ub = (label, icon, cls, attrs) => `<button type="button" class="ub ${cls || ''}" ${attrs || ''} aria-label="${esc(label)}"><span class="ub-t">${esc(label)}</span><span class="ub-i">${svgIcon(icon)}</span></button>`;
document.querySelectorAll('.ub[data-ic]').forEach(makeUb);

// drop zones: drag & drop a file, click to choose one, or paste / type the path of a file on this PC (videos
// already on the PC are then used in place, no upload). Each zone holds <input type=file name=X> and X_path.
let ALLOW_PATHS = true;   // false on a hosted server (Hugging Face Space): uploads only
const VIDEO_EXT = /\.(mp4|mov|m4v|avi|mkv|mts|m2ts|webm)$/i;
function initDrop(z){
  const name = z.dataset.name, kind = z.dataset.kind || 'video', canPath = kind === 'video';
  z.tabIndex = 0; z.setAttribute('role', 'button');
  z.innerHTML = `<input type="file" name="${name}" accept="${kind === 'toml' ? '.toml' : 'video/*'}" hidden>` +
    (canPath ? `<input type="hidden" name="${name}_path">` : '') +
    `<div class="dic">${svgIcon(kind === 'toml' ? 'file' : 'video')}</div>
     <div class="dm"><div class="dt">${esc(z.dataset.title)}</div>
       <div class="dn">Drop the ${esc(z.dataset.hint)} here</div>
       <div class="dh">${z.classList.contains('sm') ? 'or click' : 'or click to choose'}${canPath ? '<span class="pathopt"> · <button type="button" class="lnk">type a path</button>' + (z.classList.contains('sm') ? '' : ' or paste one (Ctrl+V)') + '</span>' : ''}</div>
       ${canPath ? '<input type="text" class="dpath hidden" placeholder="C:\\videos\\' + esc(name) + '.mp4  (Enter)">' : ''}</div>
     <button type="button" class="dx hidden" title="Remove" aria-label="Remove">&times;</button>` +
    ub(kind === 'toml' ? 'Add file' : 'Add video', 'add', 'sm', 'tabindex="-1"');
  const inp = z.querySelector('input[type=file]'), path = z.querySelector(`input[name="${name}_path"]`);
  const dn = z.querySelector('.dn'), dx = z.querySelector('.dx'), dp = z.querySelector('.dpath'), dic = z.querySelector('.dic');
  const okFile = f => kind === 'toml' ? /\.toml$/i.test(f.name) : (/^video\//.test(f.type) || VIDEO_EXT.test(f.name));
  const show = () => {
    const f = inp.files[0], p = path && path.value;
    z.classList.toggle('has', !!(f || p)); dx.classList.toggle('hidden', !(f || p));
    dic.innerHTML = svgIcon(f || p ? 'check' : kind === 'toml' ? 'file' : 'video');
    dn.textContent = f ? `${f.name} · ${(f.size / 1e6).toFixed(f.size < 1e7 ? 1 : 0)} MB` : p ? `${p}  (on this PC, no upload)` : `Drop the ${z.dataset.hint} here`;
    dn.title = dn.textContent;
  };
  const setFile = f => {
    if (!okFile(f)) return zoneMsg(z, kind === 'toml' ? `${f.name} is not a .toml file.` : `${f.name} does not look like a video.`);
    const dt = new DataTransfer(); dt.items.add(f); inp.files = dt.files; if (path) path.value = ''; show(); zoneMsg(z, '');
  };
  const setPath = s => {
    s = (s || '').trim().replace(/^["']|["']$/g, '');
    if (!s || !ALLOW_PATHS) return;
    if (!VIDEO_EXT.test(s)) return zoneMsg(z, 'That path does not end in a video file name (.mp4, .mov, ...).');
    path.value = s; inp.value = ''; show(); zoneMsg(z, '');
  };
  z.reset = () => { inp.value = ''; if (path) path.value = ''; if (dp){ dp.value = ''; dp.classList.add('hidden'); } show(); zoneMsg(z, ''); };
  z.has = () => !!(inp.files[0] || (path && path.value));
  z.addEventListener('click', e => {
    if (e.target === inp) return;
    if (e.target.closest('.dx')){ e.stopPropagation(); z.reset(); return; }
    if (e.target.closest('.lnk')){ e.stopPropagation(); dp.classList.remove('hidden'); dp.focus(); return; }
    if (e.target.closest('.dpath')) return;
    inp.click();
  });
  z.addEventListener('keydown', e => { if ((e.key === 'Enter' || e.key === ' ') && e.target === z){ e.preventDefault(); inp.click(); } });
  inp.onchange = () => { if (inp.files[0]) setFile(inp.files[0]); };
  if (dp){
    dp.addEventListener('keydown', e => { if (e.key === 'Enter'){ e.preventDefault(); setPath(dp.value); dp.classList.add('hidden'); }
      if (e.key === 'Escape'){ dp.classList.add('hidden'); } });
    dp.addEventListener('blur', () => { if (dp.value.trim()) setPath(dp.value); dp.classList.add('hidden'); });
  }
  ['dragenter', 'dragover'].forEach(ev => z.addEventListener(ev, e => { e.preventDefault(); z.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => z.addEventListener(ev, e => { if (ev === 'dragleave' && z.contains(e.relatedTarget)) return; z.classList.remove('over'); }));
  z.addEventListener('drop', e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) return setFile(f);
    if (canPath) setPath(e.dataTransfer.getData('text/plain'));
  });
  z.addEventListener('paste', e => {
    if (e.target.closest('.dpath')) return;
    const f = e.clipboardData.files[0];
    if (f){ e.preventDefault(); return setFile(f); }
    if (canPath){ e.preventDefault(); setPath(e.clipboardData.getData('text/plain')); }
  });
  show();
}
function zoneMsg(z, text){
  let m = z.nextElementSibling && z.nextElementSibling.classList.contains('zmsg') ? z.nextElementSibling : null;
  if (!text){ if (m) m.remove(); return; }
  if (!m){ m = document.createElement('div'); m.className = 'msg err zmsg'; z.after(m); }
  m.textContent = text;
}
document.querySelectorAll('.drop').forEach(initDrop);
// a file dropped next to a zone must not open the video in the browser tab
['dragover', 'drop'].forEach(ev => window.addEventListener(ev, e => { if (!e.target.closest || !e.target.closest('.drop')) e.preventDefault(); }));
function resetFiles(form){ form.querySelectorAll('.drop').forEach(z => z.reset()); }
function msg(el, text, err){ el.textContent = text || ''; el.classList.toggle('hidden', !text); el.classList.toggle('err', !!err); }

// ====================================================================== router
let pollTimer = null;
function stopPolling(){ clearTimeout(pollTimer); pollTimer = null; }
function route(){
  stopPolling(); leaveResults(); closeModals();
  const [v, id] = location.hash.replace(/^#\/?/, '').split('/');
  const view = {session: 'vSession', history: 'vHistory'}[v] || 'vHome';
  document.querySelectorAll('.view').forEach(s => s.classList.toggle('hidden', s.id !== view));
  document.querySelectorAll('nav a').forEach(a => a.classList.toggle('on', a.dataset.nav === ({vHome: 'home', vHistory: 'history'}[view])));
  window.scrollTo(0, 0);
  if (view === 'vSession') openSession(decodeURIComponent(id || ''));
  else if (view === 'vHistory') loadHistory();
  else loadHome();
}
window.addEventListener('hashchange', route);

// ====================================================================== home
let CAMS = [];
let calMode = null;          // saved | videos | import  (null = not chosen yet: picked from the saved profiles)
const camUsable = c => c.ready || ['queued', 'running'].includes((c.status || {}).state);
function setCalMode(m){
  calMode = m;
  document.querySelectorAll('#calMode button').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
  $('calSaved').classList.toggle('hidden', m !== 'saved');
  $('calVideos').classList.toggle('hidden', m !== 'videos');
  $('calImport').classList.toggle('hidden', m !== 'import');
  $('calName').classList.toggle('hidden', m === 'saved');
  $('calBoard').classList.toggle('hidden', m === 'saved');
}
document.querySelectorAll('#calMode button').forEach(b => b.onclick = () => setCalMode(b.dataset.mode));
function boardSummary(){
  const f = $('newForm'), v = n => f.querySelector(`[name=${n}]`).value.trim();
  const same = v('intr_corners') === v('ext_corners') && v('intr_square') === v('ext_square');
  const c = s => s.replace(/\s*,\s*/, '×');
  $('calBoardSum').textContent = `Calibration board: ${c(v('ext_corners'))} inner corners, ${v('ext_square')} mm squares` +
    (same ? '' : ` (intrinsic videos: ${c(v('intr_corners'))}, ${v('intr_square')} mm)`) +
    `, ${f.querySelector('[name=board_position]').value === 'horizontal' ? 'on the floor' : 'held upright'} (change)`;
}
$('calBoard').addEventListener('input', boardSummary);
$('calBoard').addEventListener('change', boardSummary);
async function loadHome(){
  try{
    const [cams, sessions] = await Promise.all([api('/api/cameras'), api('/api/sessions')]);
    CAMS = cams;
    const sel = $('nCam'), keep = sel.value || store.get('camera');
    const usable = cams.filter(camUsable);
    sel.innerHTML = usable.length ? usable.map(c => `<option value="${esc(c.id)}">${esc(c.name)}${c.ready ? '' : ' (calibrating…)'}</option>`).join('')
      : '<option value="">No saved calibration yet</option>';
    if (keep && usable.some(c => c.id === keep)) sel.value = keep; else if (usable[0]) sel.value = usable[0].id;
    $('calMode').querySelector('[data-mode=saved]').disabled = !usable.length;
    if (calMode === null || (calMode === 'saved' && !usable.length)) setCalMode(usable.length ? 'saved' : 'videos');
    renderCamCard();
    renderHeroStats(sessions);
    renderSessionList($('recent'), sessions.slice(0, 8));
    if (sessions.some(s => s.state === 'running' || s.state === 'queued') || cams.some(c => ['running', 'queued'].includes((c.status || {}).state)))
      pollTimer = setTimeout(loadHome, 3000);
  }catch(e){ $('recent').innerHTML = `<div class="errbox"><span class="ic">!</span>${esc(e.message)}</div>`; }
}
function renderHeroStats(sessions){
  const done = sessions.filter(s => s.state === 'done' && s.summary), withErr = done.filter(s => s.summary.analysed);
  $('hsSessions').textContent = done.length;
  $('hsReps').textContent = done.reduce((a, s) => a + (s.summary.reps || 0), 0);
  $('hsCorrect').textContent = withErr.length ? Math.round(withErr.reduce((a, s) => a + (s.summary.pct_correct || 0), 0) / withErr.length) + '%' : '–';
}
function renderSessionList(el, rows){
  el.innerHTML = rows.length ? rows.map(r => {
    const s = r.summary || {}, top = s.top_error ? errInfo(s.top_error) : null;
    const extra = r.state === 'queued' && r.queue ? `#${r.queue} in line` : r.state === 'running' ? (STAGE_LABEL[r.stage] || '') : '';
    return `<a class="item" href="#/session/${encodeURIComponent(r.id)}">
      <div class="main"><div class="title">${esc(r.name)}</div>
        <div class="row small muted" style="gap:10px;margin-top:3px"><span>${esc(fmtDate(r.created))}</span>${stateChip(r.state, extra)}
        ${s.reps != null ? `<span>${s.reps} reps</span>` : ''}${top ? `<span class="chip"><span class="dot" style="background:${top.color}"></span>${esc(top.name)}</span>` : ''}
        ${s.demo ? '<span class="chip">demo</span>' : ''}</div></div>
      ${s.pct_correct != null ? `<div class="score num" style="--p:${s.pct_correct}" title="reps correct">${s.pct_correct}%</div>` : '<div class="score none">–</div>'}</a>`;
  }).join('') : `<div class="empty">${svgIcon('video', 'currentColor')}No analyses yet. Start one on the left.</div>`;
}
$('nCam').onchange = () => { store.set('camera', $('nCam').value); renderCamCard(); };

// the chosen saved calibration: state, intrinsics, warnings, delete
function renderCamCard(){
  const c = CAMS.find(x => x.id === $('nCam').value), el = $('camCard');
  if (!c){ el.innerHTML = ''; return; }
  const st = c.status || {}, running = ['running', 'queued'].includes(st.state);
  const extra = st.state === 'running' ? (STAGE_LABEL[st.stage] || '') + (st.progress ? ` ${st.progress.done}/${st.progress.total}` : '') : st.state === 'queued' && c.queue ? `#${c.queue} in line` : '';
  el.innerHTML = `<div class="cam"><div class="hd"><span class="name">${esc(c.name)}</span>${c.ready && !running ? '<span class="state done"><span class="dot"></span>Ready</span>' : stateChip(st.state || 'error', extra)}</div>
    <div class="small muted">${esc(fmtDate(c.created))} · ${c.source === 'import' ? 'imported calibration' : 'from intrinsic videos'} · board ${esc((c.board.ext_corners || '').replace(",", "×"))} inner corners, ${esc(c.board.ext_square || '')} mm, ${c.board.board_position === 'horizontal' ? 'on the floor' : 'held upright'} · fallback ${c.fallback ? 'yes' : 'no'}</div>
    ${c.intrinsics.length ? `<div class="tscroll"><table class="t" style="margin-top:8px"><thead><tr><th>Camera</th><th>Size</th><th class="n">fx</th><th class="n">fy</th><th class="n">cx</th><th class="n">cy</th></tr></thead><tbody>${c.intrinsics.map(i => `<tr style="cursor:default"><td>${esc(i.camera)}</td><td>${i.size.join('×')}</td><td class="n">${i.fx}</td><td class="n">${i.fy}</td><td class="n">${i.cx}</td><td class="n">${i.cy}</td></tr>`).join('')}</tbody></table></div>` : ''}
    ${c.intrinsics.flatMap(i => i.warnings.map(w => `<div class="warnbox"><span class="ic">!</span><div>${esc(i.camera)} camera: ${esc(w)} - the intrinsic video may not cover the picture well. Consider recording it again.</div></div>`)).join('')}
    ${running ? '<div class="small muted" style="margin-top:6px">Still calibrating: an analysis started now waits for it.</div>' : ''}
    <div class="row" style="margin-top:10px">${!running ? ub('Delete calibration', 'delete', 'sm red', `data-cdel="${esc(c.id)}"`) : ''}</div></div>`;
  const del = el.querySelector('[data-cdel]');
  if (del) del.onclick = async () => {
    if (!confirm(`Delete the calibration "${c.name}"? Sessions already analysed keep their results.`)) return;
    try{ await api('/api/cameras/' + encodeURIComponent(c.id), {method: 'DELETE'}); stopPolling(); loadHome(); }catch(e){ alert(e.message); }
  };
}

$('newForm').onsubmit = async e => {
  e.preventDefault();
  const form = $('newForm'), fd = new FormData(form);
  const zone = n => form.querySelector(`.drop[data-name="${n}"]`);
  fd.set('cam_mode', calMode);
  const drop = names => names.forEach(n => { fd.delete(n); fd.delete(n + '_path'); });
  if (calMode === 'saved'){
    if (!$('nCam').value) return msg($('nMsg'), 'Choose a saved calibration, or give the intrinsic videos (New intrinsic videos).', true);
    drop(['intr_front', 'intr_side', 'ext_front', 'ext_side', 'calib', 'fallback']);
  } else if (calMode === 'videos'){
    if (!zone('intr_front').has() || !zone('intr_side').has()) return msg($('nMsg'), 'Give both intrinsic videos (front and side camera) in step 1.', true);
    if (zone('ext_front').has() !== zone('ext_side').has()) return msg($('nMsg'), 'The fallback needs both extrinsic videos (or neither).', true);
    drop(['calib', 'fallback']); fd.delete('camera');
  } else {
    if (!zone('calib').has()) return msg($('nMsg'), 'Add the calibration .toml file in step 1.', true);
    drop(['intr_front', 'intr_side', 'ext_front', 'ext_side']); fd.delete('camera');
  }
  if (!zone('front').has() || !zone('side').has()) return msg($('nMsg'), 'Give both squat videos in step 2: front and side.', true);
  msg($('nMsg'), 'Uploading…'); $('nStart').disabled = true;
  try{
    const r = await upload('/api/sessions', fd, $('nBar'));
    form.reset(); resetFiles(form); boardSummary(); msg($('nMsg'), ''); calMode = null;
    location.hash = '#/session/' + encodeURIComponent(r.id);
  }catch(err){ msg($('nMsg'), err.message, true); }
  finally{ $('nStart').disabled = false; }
};

// ====================================================================== session: processing
let S = null;   // current session id
async function openSession(id){
  S = id;
  $('procView').classList.add('hidden'); $('resView').classList.add('hidden');
  let row;
  try{ row = await api('/api/sessions/' + encodeURIComponent(id)); }
  catch(e){ $('procView').classList.remove('hidden'); $('pName').textContent = 'Session not found'; $('pSteps').innerHTML = ''; $('pErr').innerHTML = `<div class="errbox"><span class="ic">!</span>${esc(e.message)}</div>`; return; }
  if (S !== id) return;
  if (row.state === 'done') return loadResults(id);
  showProc(row);
}
function showProc(row){
  $('procView').classList.remove('hidden');
  const st = row.status || {};
  $('pName').textContent = row.name;
  $('pState').innerHTML = stateChip(row.state);
  const cj = row.camera_job, camBusy = cj && ['queued', 'running'].includes(cj.state);
  $('pSub').textContent = `Started ${fmtDate(row.created)}` + (row.state === 'queued' ? (camBusy ? ' · the cameras are calibrated first'
    : ` · waiting for ${row.queue > 1 ? (row.queue - 1) + ' other job(s)' : 'the current job'} to finish`) : '');
  const stages = st.stages || [], done = st.done || [];
  // the session's own camera calibration (new intrinsic videos), while it runs or if it failed
  let camLi = '';
  if (cj && (camBusy || (!cj.ready && cj.state === 'error'))){
    const cls = cj.state === 'running' ? 'cur' : cj.state === 'error' ? 'fail' : '';
    const pg = cj.state === 'running' ? `<span class="pg num">${esc(STAGE_LABEL[cj.stage] || '')}${cj.progress ? ` ${cj.progress.done} / ${cj.progress.total}` : ''}</span>`
      : cj.state === 'queued' ? '<span class="pg">waiting</span>' : '';
    camLi = `<li class="camstep ${cls}"><span class="ic">${cls === 'fail' ? '!' : ''}</span>Calibrating the cameras (${esc(cj.name)})${pg}</li>`;
  } else if (cj && cj.ready && row.state === 'queued' && st.stages && !done.length){
    camLi = `<li class="camstep done"><span class="ic">&#10003;</span>Cameras calibrated (${esc(cj.name)})</li>`;
  }
  $('pSteps').innerHTML = camLi + stages.map(s => {
    const cur = s === st.stage && row.state === 'running', fail = s === st.stage && ['error', 'interrupted'].includes(row.state);
    const cls = done.includes(s) ? 'done' : cur ? 'cur' : fail ? 'fail' : '';
    const pg = cur && st.progress ? `<span class="pg num">${st.progress.done} / ${st.progress.total}</span>` : '';
    return `<li class="${cls}"><span class="ic">${cls === 'done' ? '&#10003;' : cls === 'fail' ? '!' : ''}</span>${esc(STAGE_LABEL[s] || s)}${pg}</li>`;
  }).join('');
  $('pWarn').innerHTML = (st.warnings || []).map(w => `<div class="warnbox"><span class="ic">!</span><div>${esc(w)}</div></div>`).join('');
  const err = row.error && row.state !== 'running' ? row.error : cj && cj.state === 'error' && !cj.ready ? `The camera calibration failed: ${cj.error || 'see the log'}` : '';
  $('pErr').innerHTML = err ? `<div class="errbox"><span class="ic">!</span><div>${esc(err)}</div></div>` : '';
  const active = row.state === 'running' || row.state === 'queued';
  $('pCancel').classList.toggle('hidden', !active);
  $('pRetry').classList.toggle('hidden', active);
  $('pDelete').classList.toggle('hidden', active);
  $('pTime').textContent = row.elapsed_s != null ? `${Math.floor(row.elapsed_s / 60)} min ${row.elapsed_s % 60} s` : '';
  const log = $('pLog'), atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
  log.textContent = (row.log || []).join('\n');
  if (atEnd) log.scrollTop = log.scrollHeight;
  if (active) pollTimer = setTimeout(() => refreshProc(row.id), 2000);
}
async function refreshProc(id){
  if (S !== id) return;
  try{
    const row = await api('/api/sessions/' + encodeURIComponent(id));
    if (S !== id) return;
    if (row.state === 'done'){ $('procView').classList.add('hidden'); return loadResults(id); }
    showProc(row);
  }catch(e){ pollTimer = setTimeout(() => refreshProc(id), 4000); }
}
$('pCancel').onclick = async () => { if (confirm('Stop this analysis?')){ await post(`/api/sessions/${encodeURIComponent(S)}/cancel`); refreshProc(S); } };
$('pRetry').onclick = async () => { try{ await post(`/api/sessions/${encodeURIComponent(S)}/retry`); stopPolling(); refreshProc(S); }catch(e){ alert(e.message); } };
$('pDelete').onclick = () => deleteSession(S);
async function deleteSession(id){
  if (!confirm('Delete this session and all its files?')) return;
  try{ await api('/api/sessions/' + encodeURIComponent(id), {method: 'DELETE'}); location.hash = '#/'; }catch(e){ alert(e.message); }
}

// ====================================================================== 3D viewer (shared by results + compare)
function makeViewer(canvas){
  const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b1247);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  scene.add(new THREE.HemisphereLight(0xe4eaff, 0x1a2050, 0.85));
  const sun = new THREE.DirectionalLight(0xffffff, 0.7), fill = new THREE.DirectionalLight(0xffffff, 0.25);
  scene.add(sun); scene.add(fill);
  const V = {renderer, scene, camera, canvas, target: new THREE.Vector3(0, 0.9, 0), r: 3.4, theta: 0, phi: 0.12, facing: 0, home: null, grid: null};
  V.place = () => {
    const t = V.target, r = V.r * Math.max(1, 0.9 / camera.aspect);   // tall, narrow views: step back so the body fits
    camera.position.set(t.x + r * Math.sin(V.theta) * Math.cos(V.phi), t.y + r * Math.sin(V.phi), t.z + r * Math.cos(V.theta) * Math.cos(V.phi));
    camera.lookAt(t);
    sun.position.copy(camera.position).add(new THREE.Vector3(0, 2, 0));
    fill.position.set(2 * t.x - camera.position.x, t.y + 1, 2 * t.z - camera.position.z);
  };
  V.draw = () => renderer.render(scene, camera);
  V.setView = v => {
    if (v === 'reset' && V.home){ V.target.copy(V.home.target); V.r = V.home.r; v = V.home.view || 'front'; }
    if (v === 'front'){ V.theta = V.facing; V.phi = 0.12; }
    if (v === 'side'){ V.theta = V.facing + Math.PI / 2; V.phi = 0.12; }
    if (v === 'back'){ V.theta = V.facing + Math.PI; V.phi = 0.12; }
    if (v === 'rear'){ V.theta = V.facing + Math.PI * 0.72; V.phi = 0.2; }   // three-quarter view from behind
    if (v === 'top'){ V.phi = 1.35; }
    V.place(); V.draw();
  };
  V.resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); V.place(); V.draw();
  };
  new ResizeObserver(V.resize).observe(canvas);
  V.setFloor = (y, x, z) => {
    if (V.grid) scene.remove(V.grid);
    V.grid = new THREE.GridHelper(6, 24, 0x4a58b0, 0x1d276a); V.grid.position.set(x, y, z); scene.add(V.grid);
  };
  // left drag = rotate, right / shift drag = pan, wheel / pinch = zoom, double click = reset
  const ptrs = new Map(); let mode = null, pinch0 = 0, r0 = 0;
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointerdown', e => {
    canvas.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, {x: e.clientX, y: e.clientY});
    if (ptrs.size === 2){ const [a, b] = [...ptrs.values()]; pinch0 = Math.hypot(a.x - b.x, a.y - b.y); r0 = V.r; mode = 'pinch'; }
    else mode = (e.button === 2 || e.shiftKey) ? 'pan' : 'rotate';
    canvas.style.cursor = 'grabbing';
  });
  const up = e => { ptrs.delete(e.pointerId); if (!ptrs.size){ mode = null; canvas.style.cursor = 'grab'; } else if (mode === 'pinch') mode = 'rotate'; };
  canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('pointermove', e => {
    const p = ptrs.get(e.pointerId); if (!p || !mode) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
    if (mode === 'pinch'){ const [a, b] = [...ptrs.values()]; const d = Math.hypot(a.x - b.x, a.y - b.y); if (pinch0 > 0) V.r = clamp(r0 * pinch0 / Math.max(d, 1), 0.8, 12); }
    else if (mode === 'rotate'){ V.theta -= dx * 0.008; V.phi = clamp(V.phi + dy * 0.006, -0.3, 1.45); }
    else {
      const s = V.r * 0.0016, right = new THREE.Vector3(Math.cos(V.theta), 0, -Math.sin(V.theta));
      V.target.addScaledVector(right, -dx * s); V.target.y = clamp(V.target.y + dy * s, -1, 3);
    }
    V.place(); V.draw();
  });
  canvas.addEventListener('wheel', e => { e.preventDefault(); V.r = clamp(V.r * Math.pow(1.1, Math.sign(e.deltaY)), 0.8, 12); V.place(); V.draw(); }, {passive: false});
  canvas.addEventListener('dblclick', () => V.setView('reset'));
  V.place();
  return V;
}
function makeBody(V, faces){
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(NV * 3), 3).setUsage(THREE.DynamicDrawUsage));
  geom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(NV * 3), 3).setUsage(THREE.DynamicDrawUsage));
  geom.setIndex(new THREE.BufferAttribute(faces, 1));
  const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({vertexColors: true, roughness: 0.8, metalness: 0, side: THREE.DoubleSide}));
  mesh.frustumCulled = false;   // the vertices move every frame; bounds from the first frame would hide it
  V.scene.add(mesh);
  return {geom, mesh, dispose(){ V.scene.remove(mesh); geom.dispose(); mesh.material.dispose(); }};
}
function jet(t){ return [clamp(1.5 - Math.abs(4 * t - 3), 0, 1), clamp(1.5 - Math.abs(4 * t - 2), 0, 1), clamp(1.5 - Math.abs(4 * t - 1), 0, 1)]; }
// one frame of the body: positions from the int16 (mm) vertex table, colour = template x weight (same blend as the painter)
function paintBody(body, verts, row, sal, w, off){
  const pos = body.geom.attributes.position.array, col = body.geom.attributes.color.array, o = row * NV * 3;
  const ox = off ? off[0] : 0, oz = off ? off[1] : 0;
  for (let v = 0; v < NV; v++){
    pos[v * 3] = verts[o + v * 3] * 0.001 + ox; pos[v * 3 + 1] = verts[o + v * 3 + 1] * 0.001; pos[v * 3 + 2] = verts[o + v * 3 + 2] * 0.001 + oz;
  }
  body.geom.attributes.position.needsUpdate = true; body.geom.computeVertexNormals();
  const [b0, b1, b2] = BASE;
  for (let v = 0; v < NV; v++){
    let c0 = b0, c1 = b1, c2 = b2;
    if (sal && w > 0){ const s = Math.min(1, sal[v] * w); if (s > 0.04){ const j = jet(s), a = Math.min(1, s * 1.6); c0 = b0 * (1 - a) + j[0] * a; c1 = b1 * (1 - a) + j[1] * a; c2 = b2 * (1 - a) + j[2] * a; } }
    col[v * 3] = c0; col[v * 3 + 1] = c1; col[v * 3 + 2] = c2;
  }
  body.geom.attributes.color.needsUpdate = true;
}
let facesP = null;
function loadFaces(){ return facesP = facesP || fetch(U('/api/faces.bin')).then(r => { if (!r.ok) throw new Error('SMPL body model missing'); return r.arrayBuffer(); }).then(b => new Uint16Array(b)); }
async function loadTemplate(label){
  const k = tkey(label);
  if (tplCache[k] !== undefined) return tplCache[k];
  if (!TPL[k]) return tplCache[k] = null;
  try{
    const j = await (await fetch(U('/api/templates/' + encodeURIComponent(TPL[k])))).json();
    const s = Float32Array.from(j.saliency); let mx = 0; for (const x of s) mx = Math.max(mx, x);
    if (mx > 0) for (let i = 0; i < s.length; i++) s[i] /= mx;
    return tplCache[k] = s;
  }catch(e){ return tplCache[k] = null; }
}

// ====================================================================== session: results
let D = null, R = [], I = [], verts = null, idxOf = null, MV = null, body = null, dur = 0, fps = 30;
let curRep = null, paintedKey = '', playing = false, clock = {t0: 0, w0: 0}, raf = 0, loopOn = false, tab = store.get('errTab') || 'all';
const vF = $('vF'), vS = $('vS');
function leaveResults(){
  pause(); D = null; S = null;
  [vF, vS].forEach(v => { v.removeAttribute('src'); v.load(); });
}
async function loadResults(id){
  let d;
  try{ d = await api(`/api/sessions/${encodeURIComponent(id)}/data`); }catch(e){ alert(e.message); return; }
  if (S !== id) return;
  D = d; fps = d.fps || 30;
  const P = d.predictions || {reps: [], intensity: []};
  R = P.reps.map(r => ({...r, bottom_s: r.bottom_frame / fps}));
  I = P.intensity || [];
  $('resView').classList.remove('hidden');
  // header
  const s = d.summary || {};
  $('rName').textContent = d.name;
  $('rSub').textContent = fmtDate(d.created);
  $('rReps').textContent = R.length;
  $('rScore').textContent = s.pct_correct != null ? s.pct_correct + '%' : '–';
  const top = s.top_error ? errInfo(s.top_error) : null;
  $('rTop').innerHTML = top ? `<span class="dot" style="background:${top.color}"></span> ${esc(top.name)}` : (s.analysed ? 'None' : '–');
  $('rZip').href = U(`/api/sessions/${encodeURIComponent(id)}/download`);
  $('rCsv').href = U(`/api/sessions/${encodeURIComponent(id)}/reps.csv`);
  $('rCompare').disabled = R.length < 2;
  const bn = $('rBanner');
  if (P.demo){
    bn.className = 'banner demo';
    bn.innerHTML = `<span class="tag">DEMO</span><span style="flex:1">These errors and attention are made up to try the page. They say nothing about this squat.</span><button class="btn sm" id="demoOff">Remove demo</button>`;
  } else if (!P.model_active){
    bn.className = 'banner';
    bn.innerHTML = `<span style="flex:1">Error detection is not active yet: the trained model is not installed, so the reps are shown without errors.</span><button class="btn sm" id="demoOn">Try demo predictions</button>`;
  } else bn.className = 'banner hidden';
  if ($('demoOn')) $('demoOn').onclick = () => setDemo(true);
  if ($('demoOff')) $('demoOff').onclick = () => setDemo(false);
  const focus = s.top_error ? errInfo(s.top_error).focus : (s.analysed && s.pct_correct === 100 ? 'Every rep was correct - keep it up.' : '');
  $('rFocus').innerHTML = focus ? `Focus next time: <b>${esc(focus)}</b>` : '';
  $('rFocus').classList.toggle('hidden', !focus);
  $('rWarn').innerHTML = (d.warnings || []).map(w => `<div class="warnbox"><span class="ic">!</span><div>${esc(w)}</div></div>`).join('');
  // videos
  [['front', vF], ['side', vS]].forEach(([cam, v]) => {
    const has = d.videos.includes(cam);
    v.closest('.vbox').classList.toggle('hidden', !has);
    if (has) v.src = U(`/api/sessions/${encodeURIComponent(id)}/file/${cam}.mp4`);
  });
  updateDuration();
  [vF, vS].forEach(v => v.onloadedmetadata = () => { updateDuration(); showTime(now(), true); });
  const m = master(); if (m) m.onended = () => pause();
  // templates for the predicted errors
  await Promise.all([...new Set(R.filter(r => r.label && r.label !== 'correct').map(r => r.label))].map(loadTemplate));
  renderErrors(); setTab(tab);
  // 3D body
  verts = null; idxOf = null;
  if (body){ body.dispose(); body = null; }
  if (!MV) MV = makeViewer($('gl'));
  const sm = $('stageMsg');
  if (!d.mesh){
    sm.classList.remove('hidden');
    sm.innerHTML = d.smpl ? '<div><b>3D body not available</b></div><div class="small">The body mesh could not be built for this session (see the warning above). The videos, errors and timeline still work.</div>'
      : '<div><b>3D body not available</b></div><div class="small">The SMPL body model files are missing on this PC (MotionBERT\\data\\mesh\\SMPL_NEUTRAL.pkl).</div>';
  } else {
    sm.classList.remove('hidden'); sm.innerHTML = '<div>Loading the 3D body…</div>';
    try{
      const [faces, buf] = await Promise.all([loadFaces(), fetchProgress(U(`/api/sessions/${encodeURIComponent(id)}/file/mesh/verts.bin`), sm)]);
      if (S !== id) return;
      verts = new Int16Array(buf);
      idxOf = new Map(d.mesh.frames.map((f, i) => [f, i]));
      body = makeBody(MV, faces);
      frameCamera(MV, 0);
      sm.classList.add('hidden');
    }catch(e){ sm.innerHTML = `<div><b>3D body not available</b></div><div class="small">${esc(e.message)}</div>`; }
  }
  updateDuration();
  const firstErr = R.find(r => r.label && r.label !== 'correct');
  MV.home = {target: MV.target.clone(), r: MV.r, view: firstErr ? errInfo(firstErr.label).view : 'front'};
  MV.setView('reset');
  paintedKey = ''; curRep = undefined;
  seek(0);
  drawTrack();
}
async function fetchProgress(url, el){
  const r = await fetch(url);
  if (!r.ok) throw new Error(`3D body: ${r.status}`);
  const total = +r.headers.get('Content-Length') || 0, reader = r.body.getReader(), parts = []; let got = 0;
  for (;;){ const {done, value} = await reader.read(); if (done) break; parts.push(value); got += value.length;
    el.innerHTML = `<div>Loading the 3D body… ${(got / 1e6).toFixed(0)}${total ? ' / ' + (total / 1e6).toFixed(0) : ''} MB</div>`; }
  const out = new Uint8Array(got); let o = 0; for (const p of parts){ out.set(p, o); o += p.length; }
  return out.buffer;
}
function frameCamera(V, row){
  // aim between the floor and the top of the head, at the body's centre; face the body
  let sx = 0, sz = 0, top = -1e9; const o = row * NV * 3;
  for (let v = 0; v < NV; v++){ sx += verts[o + v * 3]; sz += verts[o + v * 3 + 2]; }
  for (let i = 1; i < verts.length; i += 3 * 29) top = Math.max(top, verts[i]);
  const floorY = (D.mesh.floor_y_mm || 0) / 1000;
  V.target.set(sx / NV / 1000, (floorY + top / 1000) / 2, sz / NV / 1000); V.r = 3.2;
  const f = D.mesh.facing_xz || [0, 1]; V.facing = Math.atan2(f[0], f[1]);
  V.setFloor(floorY, V.target.x, V.target.z);
}
async function setDemo(on){
  try{ await post(`/api/sessions/${encodeURIComponent(S)}/demo`, {on}); const id = S; pause(); loadResults(id); }catch(e){ alert(e.message); }
}
$('rDelete').onclick = () => deleteSession(S);

// ---------------------------------------------------------------------- time
function master(){ return D && D.videos.includes('front') ? vF : D && D.videos.includes('side') ? vS : null; }
function updateDuration(){
  if (!D) return;
  const m = master();
  const meshDur = D.mesh ? (D.mesh.frames[D.mesh.frames.length - 1] + 1) / fps : 0;
  dur = m && isFinite(m.duration) && m.duration > 0 ? m.duration : Math.max(D.n_frames / fps, meshDur);
  $('scrub').max = dur;
}
function now(){ const m = master(); return m ? m.currentTime : clock.t0 + (playing ? (performance.now() - clock.w0) / 1000 * +$('tSpeed').value : 0); }
const repAt = f => R.find(r => f >= r.start_frame && f <= r.end_frame) || null;
function showTime(t, force){
  if (!D) return;
  const f = Math.round(t * fps), rep = repAt(f);
  if (verts && body){
    let row = idxOf.get(f);
    if (row === undefined) row = f < D.mesh.frames[0] ? 0 : D.mesh.frames.length - 1;
    const err = rep && rep.label && rep.label !== 'correct';
    const sal = err ? tplCache[tkey(rep.label)] : null;
    const w = sal ? (I[f] || 0) * (rep.uncertain ? 0.5 : 1) : 0;
    const key = `${row}|${rep ? rep.rep : ''}|${w.toFixed(3)}`;
    if (force || key !== paintedKey){ paintedKey = key; paintBody(body, verts, row, sal, w); MV.draw(); }
    $('legend').classList.toggle('hidden', !sal);
    $('tplNote').textContent = err && !sal ? `No painted map for ${errInfo(rep.label).name.toLowerCase()} yet` : '';
  }
  if (rep !== curRep){
    curRep = rep;
    const lb = $('repLabel');
    if (rep){
      const e = rep.label ? errInfo(rep.label) : null;
      lb.innerHTML = `Rep ${rep.rep}` + (e ? ` · <span class="dot" style="background:${e.color}"></span>${esc(e.name)} ${Math.round(rep.confidence * 100)}%${rep.uncertain ? ' (uncertain)' : ''}` : '');
    }
    lb.classList.toggle('hidden', !rep);
    document.querySelectorAll('[data-jump]').forEach(b => b.classList.toggle('on', rep && +b.dataset.jump === rep.rep));
    if (tab === 'rep') renderRep();
  }
  $('tTime').textContent = `${t.toFixed(2)} / ${dur.toFixed(2)} s`;
  $('scrub').value = t;
  drawTrack(t);
}
function seek(t){
  t = clamp(t, 0, dur || 0);
  [vF, vS].forEach(v => { if (v.getAttribute('src')) v.currentTime = t; });
  clock = {t0: t, w0: performance.now()};
  showTime(t);
}
function tick(){
  if (!playing) return;
  let t = now();
  const lr = loopOn && curRep;
  if (lr && t >= lr.end_frame / fps){ seek(lr.start_frame / fps); t = lr.start_frame / fps; }
  if (t >= dur - 0.02){ pause(); t = dur; }
  showTime(t); raf = requestAnimationFrame(tick);
}
function play(){
  if (!D) return;
  if (now() >= dur - 0.05) seek(0);
  playing = true; $('tPlay').innerHTML = '&#10074;&#10074;'; $('tPlay').setAttribute('aria-label', 'Pause');
  [vF, vS].forEach(v => { if (v.getAttribute('src')){ v.playbackRate = +$('tSpeed').value; v.play().catch(() => {}); } });
  clock = {t0: now(), w0: performance.now()}; tick();
}
function pause(){
  playing = false; cancelAnimationFrame(raf);
  $('tPlay').innerHTML = '&#9654;'; $('tPlay').setAttribute('aria-label', 'Play');
  [vF, vS].forEach(v => v.pause());
}
const step = n => { pause(); seek(now() + n / fps); };
function jumpRep(dir){
  if (!R.length) return;
  const f = Math.round(now() * fps);
  const r = dir > 0 ? R.find(x => x.start_frame > f) : [...R].reverse().find(x => x.start_frame < f - 2);
  if (r){ pause(); seek(r.start_frame / fps); }
}
$('tPlay').onclick = () => playing ? pause() : play();
$('tBack').onclick = () => step(-1); $('tFwd').onclick = () => step(1);
$('tPrevRep').onclick = () => jumpRep(-1); $('tNextRep').onclick = () => jumpRep(1);
$('tLoop').onclick = () => { loopOn = !loopOn; $('tLoop').classList.toggle('primary', loopOn); };
$('tSpeed').onchange = () => { [vF, vS].forEach(v => v.playbackRate = +$('tSpeed').value); clock = {t0: now(), w0: performance.now()}; };
$('scrub').oninput = () => { pause(); seek(+$('scrub').value); };
document.querySelectorAll('[data-view]').forEach(b => b.onclick = () => MV && MV.setView(b.dataset.view));
document.addEventListener('keydown', e => {
  if (!D || $('resView').classList.contains('hidden') || ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (!$('cmpModal').classList.contains('hidden') || !$('vidModal').classList.contains('hidden')) return;
  if (e.code === 'Space'){ e.preventDefault(); $('tPlay').click(); }
  else if (e.key === 'ArrowLeft'){ e.preventDefault(); step(-1); }
  else if (e.key === 'ArrowRight'){ e.preventDefault(); step(1); }
  else if (e.key === 'n' || e.key === 'N'){ jumpRep(1); }
  else if (e.key === 'p' || e.key === 'P'){ jumpRep(-1); }
  else if (e.key === 'l' || e.key === 'L'){ $('tLoop').click(); }
});

// ---------------------------------------------------------------------- timeline track
const TR = {repY: 16, repH: 24, attY: 50, attH: 22};
function drawTrack(t){
  const c = $('track'), dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  if (!D || !W || !dur) return;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)){ c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const x = s => s / dur * W;
  g.font = '500 10px Poppins,system-ui,Segoe UI,sans-serif'; g.fillStyle = '#8d94b0'; g.textBaseline = 'alphabetic';
  g.fillText('REPS', 4, TR.repY - 4); g.fillText('MODEL ATTENTION', 4, TR.attY - 3);
  g.strokeStyle = '#dce2f2'; g.lineWidth = 1; g.beginPath(); g.moveTo(0, TR.attY + TR.attH + 0.5); g.lineTo(W, TR.attY + TR.attH + 0.5); g.stroke();
  for (const r of R){
    const x0 = x(r.start_frame / fps) + 1, x1 = x((r.end_frame + 1) / fps) - 1, w = Math.max(2, x1 - x0);
    const e = errInfo(r.label);
    g.globalAlpha = r.label ? 1 : 0.55; g.fillStyle = e.color;
    roundRect(g, x0, TR.repY, w, TR.repH, 4); g.fill(); g.globalAlpha = 1;
    if (r.uncertain){ g.save(); roundRect(g, x0, TR.repY, w, TR.repH, 4); g.clip(); g.strokeStyle = 'rgba(255,255,255,.5)'; g.lineWidth = 3;
      for (let k = -TR.repH; k < w; k += 8){ g.beginPath(); g.moveTo(x0 + k, TR.repY + TR.repH); g.lineTo(x0 + k + TR.repH, TR.repY); g.stroke(); } g.restore(); }
    if (curRep === r){ g.strokeStyle = '#001b94'; g.lineWidth = 2; roundRect(g, x0 - 1, TR.repY - 1, w + 2, TR.repH + 2, 5); g.stroke(); }
    const lbl = r.label ? `${r.rep} ${e.short}` : `${r.rep}`;
    g.font = '600 11px Poppins,system-ui,Segoe UI,sans-serif';
    if (g.measureText(lbl).width < w - 6){ g.fillStyle = '#fff'; g.fillText(lbl, x0 + 5, TR.repY + 16); }
    else if (g.measureText(String(r.rep)).width < w - 4){ g.fillStyle = '#fff'; g.fillText(String(r.rep), x0 + 3, TR.repY + 16); }
    // attention (smoothed alpha x confidence) under the rep, in the rep's colour
    if (r.label && r.label !== 'correct' && I.length){
      g.fillStyle = e.color; g.globalAlpha = r.uncertain ? 0.5 : 0.9;
      g.beginPath(); g.moveTo(x(r.start_frame / fps), TR.attY + TR.attH);
      for (let f = r.start_frame; f <= r.end_frame; f++) g.lineTo(x(f / fps), TR.attY + TR.attH - (I[f] || 0) * TR.attH);
      g.lineTo(x(r.end_frame / fps), TR.attY + TR.attH); g.closePath(); g.fill(); g.globalAlpha = 1;
    }
  }
  if (!R.length){ g.fillStyle = '#5a6386'; g.font = '12px Poppins,system-ui,sans-serif'; g.fillText('No squat reps were found.', 8, TR.repY + 16); }
  const tt = t ?? now(), px = x(tt);
  g.fillStyle = '#001b94'; g.fillRect(px - 1, 6, 2, H - 8);
}
function roundRect(g, x, y, w, h, r){ r = Math.min(r, w / 2, h / 2); g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
new ResizeObserver(() => drawTrack()).observe($('track'));
(function trackInput(){
  const c = $('track'); let drag = false;
  const tAt = e => { const b = c.getBoundingClientRect(); return clamp((e.clientX - b.left) / b.width, 0, 1) * dur; };
  const repUnder = e => { const b = c.getBoundingClientRect(), y = e.clientY - b.top, f = Math.round(tAt(e) * fps);
    return y >= TR.repY - 2 && y <= TR.repY + TR.repH + 2 ? repAt(f) : null; };
  c.addEventListener('pointerdown', e => {
    if (!D) return; pause();
    const r = repUnder(e);
    if (r){ seek(r.start_frame / fps); return; }
    drag = true; c.setPointerCapture(e.pointerId); seek(tAt(e));
  });
  c.addEventListener('pointermove', e => {
    if (!D) return;
    if (drag){ seek(tAt(e)); return; }
    const r = repUnder(e);
    if (r){
      const er = errInfo(r.label);
      showTip(e, `<b>Rep ${r.rep}</b> · ${((r.end_frame - r.start_frame) / fps).toFixed(1)} s<br>` + (r.label ? `<span class="dot" style="background:${er.color}"></span> ${esc(er.name)} ${Math.round(r.confidence * 100)}%${r.uncertain ? ' (uncertain)' : ''}` : 'not analysed') + '<br><span class="faint">click to jump</span>');
    } else hideTip();
  });
  c.addEventListener('pointerup', () => drag = false);
  c.addEventListener('pointerleave', hideTip);
})();
function showTip(e, html){ const t = $('tip'); t.innerHTML = html; t.classList.remove('hidden');
  const w = t.offsetWidth, h = t.offsetHeight; t.style.left = clamp(e.clientX + 12, 4, innerWidth - w - 4) + 'px'; t.style.top = clamp(e.clientY - h - 12, 4, innerHeight - h - 4) + 'px'; }
function hideTip(){ $('tip').classList.add('hidden'); }

// ---------------------------------------------------------------------- errors and fixes
function setTab(t){
  tab = t; store.set('errTab', t);
  document.querySelectorAll('[data-tab]').forEach(b => { b.classList.toggle('on', b.dataset.tab === t); b.setAttribute('aria-selected', b.dataset.tab === t); });
  t === 'rep' ? renderRep() : renderErrors();
}
document.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => setTab(b.dataset.tab));
const repBtn = r => `<button data-jump="${r.rep}" class="${curRep === r ? 'on' : ''}" title="Jump to rep ${r.rep}">Rep ${r.rep}${r.uncertain ? '?' : ''}</button>`;
function bindJumps(){
  document.querySelectorAll('#errBody [data-jump]').forEach(b => b.onclick = () => {
    const r = R.find(x => x.rep === +b.dataset.jump);
    if (!r) return;
    pause(); seek(r.start_frame / fps);
    if (MV && r.label && r.label !== 'correct') MV.setView(errInfo(r.label).view);   // the angle that shows this error best
  });
}
function renderErrors(){
  if (tab !== 'all') return;
  const el = $('errBody'), P = D.predictions || {};
  if (!R.length){ el.innerHTML = '<div class="empty">No squat reps were found in this recording.</div>'; return; }
  const analysed = R.filter(r => r.label);
  if (!analysed.length){
    el.innerHTML = `<div class="card"><div class="hd">Error detection is not active yet</div><p>The error-detection model is not installed, so the reps below are shown without errors.${P.demo ? '' : ' Use <b>Try demo predictions</b> above to see how the results will look.'}</p></div>
      <div class="card"><div class="hd">${R.length} reps found</div><div class="reps">${R.map(repBtn).join('')}</div>
      <p>${R.map(r => `Rep ${r.rep}: ${((r.end_frame - r.start_frame) / fps).toFixed(1)} s`).join(' · ')}</p></div>`;
    bindJumps(); return;
  }
  const by = {}; analysed.forEach(r => (by[r.label] = by[r.label] || []).push(r));
  const unc = analysed.filter(r => r.uncertain).length;
  const parts = Object.entries(by).sort((a, b) => b[1].length - a[1].length).map(([k, v]) => `${v.length} ${errInfo(k).name.toLowerCase()}`);
  const order = (C ? C.classes : Object.keys(by)).filter(k => by[k]).sort((a, b) => (a === 'correct') - (b === 'correct') || by[b].length - by[a].length);
  el.innerHTML = `<p style="margin:0 0 10px">${R.length} reps: ${esc(parts.join(', '))}${unc ? `; ${unc} uncertain` : ''}.</p>` +
    order.map(k => { const e = errInfo(k), reps = by[k];
      return `<div class="card"><div class="hd"><span class="dot" style="background:${e.color}"></span>${esc(e.name)}<span class="muted small" style="margin-left:auto">${reps.length} rep${reps.length > 1 ? 's' : ''}</span></div>
        <div class="reps">${reps.map(repBtn).join('')}</div>
        ${k === 'correct' ? `<p>${esc(e.what)}</p>` : `<p><b>What:</b> ${esc(e.what)}</p><p><b>Why it matters:</b> ${esc(e.why)}</p><p><b>How to fix:</b></p><ul>${e.fix.map(f => `<li>${esc(f)}</li>`).join('')}</ul>`}</div>`; }).join('');
  bindJumps();
}
function renderRep(){
  if (tab !== 'rep' || !D) return;
  const el = $('errBody'), r = curRep;
  if (!r){ el.innerHTML = '<div class="empty">Move the slider into a rep (or press N) to see its details.</div>'; return; }
  const head = `<div class="hd" style="margin-bottom:2px">Rep ${r.rep}<span class="muted small" style="margin-left:auto">${(r.start_frame / fps).toFixed(1)}–${(r.end_frame / fps).toFixed(1)} s · ${((r.end_frame - r.start_frame) / fps).toFixed(1)} s</span></div>`;
  if (!r.label){ el.innerHTML = `<div class="card">${head}<p>Not analysed: the error-detection model is not installed yet.</p></div>`; return; }
  const e = errInfo(r.label), classes = (D.predictions.classes || []);
  const top = (r.probs || []).map((p, i) => [classes[i], p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
  el.innerHTML = `<div class="card">${head}
      <div class="hd" style="font-size:16px;margin:8px 0 2px"><span class="dot" style="background:${e.color}"></span>${esc(e.name)} <span class="muted num" style="font-weight:400">${Math.round(r.confidence * 100)}%</span></div>
      ${r.uncertain ? `<div class="warnbox" style="margin:6px 0"><span class="ic">?</span><div>The model is not sure about this rep (below ${Math.round((D.predictions.uncertain_below || 0.5) * 100)} %).</div></div>` : ''}
      <div class="probs">${top.map(([k, p]) => `<span>${esc(errInfo(k).name)}</span><span class="pb"><i style="width:${(p * 100).toFixed(1)}%;background:${errInfo(k).color}"></i></span><span class="num muted">${Math.round(p * 100)}%</span>`).join('')}</div>
    </div>
    <div class="card">${r.label === 'correct' ? `<p>${esc(e.what)}</p><p>${esc(e.fix[0] || '')}</p>`
      : `<p><b>What:</b> ${esc(e.what)}</p><p><b>Why it matters:</b> ${esc(e.why)}</p><p><b>How to fix:</b></p><ul>${e.fix.map(f => `<li>${esc(f)}</li>`).join('')}</ul>`}</div>`;
}

// ---------------------------------------------------------------------- help popover, enlarged video
$('helpBtn').onclick = e => { e.stopPropagation(); const p = $('helpPop'), open = p.classList.toggle('hidden'); $('helpBtn').setAttribute('aria-expanded', !open); };
document.addEventListener('click', e => { if (!$('helpPop').contains(e.target)) { $('helpPop').classList.add('hidden'); $('helpBtn').setAttribute('aria-expanded', 'false'); } });
let bigFrom = null;
document.querySelectorAll('.vbox').forEach(b => b.onclick = () => {
  if (bigFrom){ closeBig(); return; }
  bigFrom = {box: b, parent: b.parentElement, next: b.nextElementSibling};
  $('bigVid').appendChild(b); $('vidModal').classList.remove('hidden');
});
function closeBig(){
  if (!bigFrom) return;
  bigFrom.parent.insertBefore(bigFrom.box, bigFrom.next); bigFrom = null; $('vidModal').classList.add('hidden');
}
$('vidModal').addEventListener('click', e => { if (e.target === $('vidModal')) closeBig(); });
function closeModals(){ closeBig(); closeCompare(); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModals(); });

// ---------------------------------------------------------------------- compare two reps
let CV = null, cmpBodies = [], cmpPlaying = false, cmpRaf = 0, cmpLast = 0;
$('rCompare').onclick = openCompare;
$('cmpClose').onclick = closeCompare;
$('cmpModal').addEventListener('click', e => { if (e.target === $('cmpModal')) closeCompare(); });
function repName(r){ return `Rep ${r.rep}` + (r.label ? ` · ${errInfo(r.label).name} ${Math.round(r.confidence * 100)}%` : ''); }
async function openCompare(){
  if (!D || R.length < 2) return;
  pause();
  $('cmpModal').classList.remove('hidden');
  const opts = R.map((r, i) => `<option value="${i}">${esc(repName(r))}</option>`).join('');
  $('cmpA').innerHTML = opts; $('cmpB').innerHTML = opts;
  let a = 0, b = 1;
  if (R.some(r => r.probs)){   // best (most "correct") vs worst
    const pc = r => r.probs ? r.probs[0] : -1, idx = R.map((r, i) => i).filter(i => R[i].probs);
    a = idx.reduce((m, i) => pc(R[i]) > pc(R[m]) ? i : m, idx[0]); b = idx.reduce((m, i) => pc(R[i]) < pc(R[m]) ? i : m, idx[0]);
    if (a === b) b = (a + 1) % R.length;
  }
  $('cmpA').value = a; $('cmpB').value = b;
  if (!verts){ $('cmpMsg').classList.remove('hidden'); $('cmpMsg').innerHTML = '<div>The 3D body is not available for this session.</div>'; return; }
  $('cmpMsg').classList.add('hidden');
  if (!CV){ CV = makeViewer($('cmpGl')); }
  const faces = await loadFaces();
  cmpBodies.forEach(x => x.dispose());
  cmpBodies = [makeBody(CV, faces), makeBody(CV, faces)];
  // place the two bodies side by side, across the person's facing direction
  const f = D.mesh.facing_xz || [0, 1]; CV.facing = Math.atan2(f[0], f[1]);
  const lat = [f[1], -f[0]];
  const floorY = (D.mesh.floor_y_mm || 0) / 1000;
  CV.target.set(0, floorY + 0.85, 0); CV.r = 4.4; CV.home = {target: CV.target.clone(), r: CV.r, view: 'front'};
  CV.setFloor(floorY, 0, 0);
  CV.lat = lat;
  CV.resize(); CV.setView('front');
  $('cmpPhase').value = 0; drawCompare();
}
function closeCompare(){ cmpPlaying = false; cancelAnimationFrame(cmpRaf); $('cmpModal').classList.add('hidden'); $('cmpPlay').innerHTML = '&#9654;'; }
function phaseFrame(r, p){
  const a = r.start_frame, b = r.bottom_frame, e = r.end_frame;
  return Math.round(p < 0.5 ? a + (b - a) * p / 0.5 : b + (e - b) * (p - 0.5) / 0.5);
}
function repCentre(r){ const row = rowOf(r.start_frame), o = row * NV * 3; let sx = 0, sz = 0;
  for (let v = 0; v < NV; v++){ sx += verts[o + v * 3]; sz += verts[o + v * 3 + 2]; } return [sx / NV / 1000, sz / NV / 1000]; }
function rowOf(f){ let row = idxOf.get(f); if (row === undefined) row = f < D.mesh.frames[0] ? 0 : D.mesh.frames.length - 1; return row; }
function drawCompare(){
  if (!CV || !verts || cmpBodies.length < 2) return;
  const p = +$('cmpPhase').value, reps = [R[+$('cmpA').value], R[+$('cmpB').value]];
  reps.forEach((r, i) => {
    const f = phaseFrame(r, p), c = repCentre(r), side = i ? 0.55 : -0.55;
    const sal = r.label && r.label !== 'correct' ? tplCache[tkey(r.label)] : null;
    const w = sal ? (I[f] || 0) * (r.uncertain ? 0.5 : 1) : 0;
    paintBody(cmpBodies[i], verts, rowOf(f), sal, w, [-c[0] + CV.lat[0] * side, -c[1] + CV.lat[1] * side]);
  });
  CV.draw();
  const lab = (r, el) => { const e = r.label ? errInfo(r.label) : null;
    el.innerHTML = e ? `<span class="dot" style="background:${e.color}"></span><b>Rep ${r.rep}</b> ${esc(e.name)} ${Math.round(r.confidence * 100)}%` : `<b>Rep ${r.rep}</b> <span class="muted">not analysed</span>`; };
  lab(reps[0], $('cmpLA')); lab(reps[1], $('cmpLB'));
  $('cmpPh').textContent = p < 0.02 ? 'start' : p < 0.48 ? `going down ${Math.round(p / 0.5 * 100)}%` : p <= 0.52 ? 'bottom' : p < 0.98 ? `coming up ${Math.round((p - 0.5) / 0.5 * 100)}%` : 'end';
}
['cmpA', 'cmpB'].forEach(k => $(k).onchange = drawCompare);
$('cmpPhase').oninput = () => { cmpPlaying = false; $('cmpPlay').innerHTML = '&#9654;'; drawCompare(); };
document.querySelectorAll('[data-cview]').forEach(b => b.onclick = () => CV && CV.setView(b.dataset.cview));
$('cmpPlay').onclick = () => {
  cmpPlaying = !cmpPlaying; $('cmpPlay').innerHTML = cmpPlaying ? '&#10074;&#10074;' : '&#9654;';
  if (!cmpPlaying) return;
  if (+$('cmpPhase').value >= 0.999) $('cmpPhase').value = 0;
  cmpLast = performance.now();
  const reps = () => [R[+$('cmpA').value], R[+$('cmpB').value]];
  const loop = () => {
    if (!cmpPlaying) return;
    const t = performance.now(), longest = Math.max(...reps().map(r => (r.end_frame - r.start_frame) / fps), 0.5);
    let p = +$('cmpPhase').value + (t - cmpLast) / 1000 / longest; cmpLast = t;
    if (p >= 1) p = 0;
    $('cmpPhase').value = p; drawCompare(); cmpRaf = requestAnimationFrame(loop);
  };
  loop();
};

// ====================================================================== history
async function loadHistory(){
  let rows;
  try{ rows = await api('/api/sessions'); }catch(e){ $('charts').innerHTML = `<div class="errbox"><span class="ic">!</span>${esc(e.message)}</div>`; return; }
  $('histTable').innerHTML = `<thead><tr><th>Date</th><th>Name</th><th class="n">Reps</th><th class="n">Correct</th><th>Most frequent error</th><th class="n">Depth (°)</th><th class="n">Rep time (s)</th><th>Status</th></tr></thead><tbody>` +
    (rows.length ? rows.map(r => { const s = r.summary || {}, top = s.top_error ? errInfo(s.top_error) : null;
      return `<tr data-id="${esc(r.id)}"><td>${esc(fmtDate(r.created))}</td><td>${esc(r.name)}${s.demo ? ' <span class="chip">demo</span>' : ''}</td><td class="n">${s.reps ?? ''}</td><td class="n">${s.pct_correct != null ? s.pct_correct + '%' : '–'}</td>
        <td>${top ? `<span class="dot" style="background:${top.color}"></span> ${esc(top.name)}` : (s.analysed ? 'none' : '–')}</td><td class="n">${s.mean_depth_deg ?? ''}</td><td class="n">${s.mean_rep_s ?? ''}</td><td>${stateChip(r.state)}</td></tr>`; }).join('')
      : '<tr><td colspan="8" class="empty">No analyses yet.</td></tr>') + '</tbody>';
  document.querySelectorAll('#histTable tbody tr[data-id]').forEach(tr => tr.onclick = () => location.hash = '#/session/' + encodeURIComponent(tr.dataset.id));
  const done = rows.filter(r => r.state === 'done' && r.summary).reverse();   // oldest first
  const withErr = done.filter(r => r.summary.analysed);
  const pts = (list, f) => list.map(r => ({label: fmtDay(r.created), name: r.name, y: f(r.summary)})).filter(p => p.y != null);
  $('charts').innerHTML = '';
  chartLine('Correct reps', 'Share of reps without an error, per session', pts(withErr, s => s.pct_correct), '%', 0, 100,
    'Appears once errors are detected (trained model, or demo predictions).');
  chartStack('Errors per session', 'Reps with each error', withErr);
  chartLine('Squat depth', 'Average knee bend at the bottom of the reps (degrees, higher = deeper)', pts(done, s => s.mean_depth_deg), '°');
  chartLine('Rep duration', 'Average time per rep (seconds)', pts(done, s => s.mean_rep_s), ' s');
}
function chartBox(title, sub){
  const d = document.createElement('div'); d.className = 'panel chart';
  d.innerHTML = `<h3>${esc(title)}</h3><div class="sub">${esc(sub)}</div>`;
  $('charts').appendChild(d); return d;
}
const CW = 520, CH = 200, PAD = {l: 38, r: 12, t: 10, b: 26};
function niceMax(v){ if (v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))); return Math.ceil(v / p * 1.1) * p; }
function axes(n, yMin, yMax, unit, labels){
  const ih = CH - PAD.t - PAD.b, iw = CW - PAD.l - PAD.r;
  let s = '<g class="axis">';
  for (let k = 0; k <= 4; k++){ const v = yMin + (yMax - yMin) * k / 4, y = PAD.t + ih - ih * k / 4;
    s += `<line x1="${PAD.l}" x2="${CW - PAD.r}" y1="${y}" y2="${y}"/><text x="${PAD.l - 6}" y="${y + 3}" text-anchor="end">${+v.toFixed(1)}${unit === '%' ? '%' : ''}</text>`; }
  const every = Math.max(1, Math.ceil(n / 8));
  labels.forEach((l, i) => { if (i % every === 0) s += `<text x="${PAD.l + (n === 1 ? iw / 2 : iw * (i + 0.5) / n)}" y="${CH - 8}" text-anchor="middle">${esc(l)}</text>`; });
  return s + '</g>';
}
function hoverCols(svg, n, html){
  const iw = CW - PAD.l - PAD.r, cross = svg.querySelector('.cross');
  for (let i = 0; i < n; i++){
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('x', PAD.l + iw * i / n); r.setAttribute('y', 0); r.setAttribute('width', iw / n); r.setAttribute('height', CH);
    r.setAttribute('fill', 'transparent');
    r.addEventListener('pointermove', e => { showTip(e, html(i)); if (cross){ const cx = PAD.l + iw * (i + 0.5) / n; cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.style.display = ''; } });
    r.addEventListener('pointerleave', () => { hideTip(); if (cross) cross.style.display = 'none'; });
    svg.appendChild(r);
  }
}
function chartLine(title, sub, pts, unit, yMin, yMax, emptyText){
  const box = chartBox(title, sub);
  if (!pts.length){ box.insertAdjacentHTML('beforeend', `<div class="empty">${esc(emptyText || 'No finished sessions yet.')}</div>`); return; }
  const n = pts.length, lo = yMin ?? 0, hi = yMax ?? niceMax(Math.max(...pts.map(p => p.y)));
  const iw = CW - PAD.l - PAD.r, ih = CH - PAD.t - PAD.b;
  const X = i => PAD.l + (n === 1 ? iw / 2 : iw * (i + 0.5) / n), Y = v => PAD.t + ih - (v - lo) / (hi - lo) * ih;
  let s = `<svg viewBox="0 0 ${CW} ${CH}" role="img" aria-label="${esc(title)}">` + axes(n, lo, hi, unit, pts.map(p => p.label));
  s += `<line class="cross" y1="${PAD.t}" y2="${PAD.t + ih}" stroke="#8d94b0" stroke-dasharray="3 3" style="display:none"/>`;
  s += `<path d="${pts.map((p, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(p.y).toFixed(1)).join('')}" fill="none" stroke="#001b94" stroke-width="2.5" stroke-linejoin="round"/>`;
  s += pts.map((p, i) => `<circle cx="${X(i)}" cy="${Y(p.y)}" r="4.5" fill="#001b94" stroke="#fff" stroke-width="2"/>`).join('') + '</svg>';
  box.insertAdjacentHTML('beforeend', s);
  hoverCols(box.querySelector('svg'), n, i => `<b>${esc(pts[i].name)}</b><br>${esc(pts[i].label)}: <b>${pts[i].y}${unit}</b>`);
}
function chartStack(title, sub, rows){
  const box = chartBox(title, sub);
  if (!rows.length){ box.insertAdjacentHTML('beforeend', '<div class="empty">Appears once errors are detected (trained model, or demo predictions).</div>'); return; }
  const classes = (C ? C.classes : []).filter(c => c !== 'correct');
  const n = rows.length, mx = Math.max(1, ...rows.map(r => classes.reduce((a, c) => a + (r.summary.counts[c] || 0), 0)));
  const hi = Math.max(4, Math.ceil(mx / 4) * 4);   // whole-number ticks (4 steps)
  const iw = CW - PAD.l - PAD.r, ih = CH - PAD.t - PAD.b, bw = Math.min(36, iw / n * 0.6);
  let s = `<svg viewBox="0 0 ${CW} ${CH}" role="img" aria-label="${esc(title)}">` + axes(n, 0, hi, '', rows.map(r => fmtDay(r.created)));
  rows.forEach((r, i) => {
    let y = PAD.t + ih; const cx = PAD.l + iw * (i + 0.5) / n;
    classes.forEach(c => { const v = r.summary.counts[c] || 0; if (!v) return; const h = v / hi * ih;
      s += `<rect x="${cx - bw / 2}" y="${y - h + 1}" width="${bw}" height="${Math.max(1, h - 2)}" rx="2" fill="${errInfo(c).color}"/>`; y -= h; });
  });
  s += '</svg>';
  box.insertAdjacentHTML('beforeend', s);
  const used = classes.filter(c => rows.some(r => r.summary.counts[c]));
  box.insertAdjacentHTML('beforeend', `<div class="legend">${used.map(c => `<span><span class="dot" style="background:${errInfo(c).color}"></span>${esc(errInfo(c).name)}</span>`).join('') || '<span>No errors in these sessions.</span>'}</div>`);
  hoverCols(box.querySelector('svg'), n, i => `<b>${esc(rows[i].name)}</b> · ${esc(fmtDay(rows[i].created))}<br>` +
    (classes.filter(c => rows[i].summary.counts[c]).map(c => `<span class="dot" style="background:${errInfo(c).color}"></span> ${esc(errInfo(c).name)}: <b>${rows[i].summary.counts[c]}</b>`).join('<br>') || 'no errors'));
}

// ====================================================================== start
(async function init(){
  try{
    if (VIEWER){ IDX = await (await fetch(VIEWER, {cache: 'no-store'})).json(); document.body.classList.add('viewer'); }
    [C, INFO] = await Promise.all([api('/api/content'), api('/api/info')]);
    TPL = (await api('/api/templates')).templates || {};
  }catch(e){ C = C || {errors: {}, classes: []}; }
  ALLOW_PATHS = INFO.allow_paths !== false;
  document.body.classList.toggle('no-paths', !ALLOW_PATHS);
  $('helpPop').textContent = (C && C.saliency_help) || '';
  $('classList').innerHTML = (C.classes || []).map(c => { const e = errInfo(c);
    return `<div class="cls"><span class="sw" style="background:${e.color}">${esc(e.short)}</span><div><b>${esc(e.name)}</b><span>${esc(e.what || '')}</span></div></div>`; }).join('');
  $('modelState').querySelector('span').textContent = INFO.model_active ? 'Error model active' : 'Error model not installed';
  $('modelState').classList.toggle('off', !INFO.model_active);
  route();
})();
