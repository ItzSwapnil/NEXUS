const $=id=>document.getElementById(id);
const money=v=>v==null?'—':`$${Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}`;
let marketsLoaded=false;
let lastBacktestSnapshot='';
let followResearchMarket=true;
let lastResearchTrial=null;
let lastCandleData=[];
const followResearchButton = $('followResearchMarket');
if (followResearchButton) {
  followResearchButton.onclick = () => {
    followResearchMarket = !followResearchMarket;
    followResearchButton.textContent = `Follow: ${followResearchMarket ? 'on' : 'off'}`;
  };
}
const note = t => { $('notice').textContent = t; };

/* ============================================================
   LIVE FORMULA FEED
   ============================================================ */
const formulaFeedLog=[];
const MAX_FEED=60;
function formulaTimestamp(){const n=new Date();return `${String(n.getHours()).padStart(2,'0')}:${String(n.getMinutes()).padStart(2,'0')}:${String(n.getSeconds()).padStart(2,'0')}`}
function pushFormulaEntry(entry){
  // entry: {asset, formula, features, stage, status, win_rate, threshold, fold}
  // status: 'active' | 'promoted' | 'rejected' | 'shadow'
  formulaFeedLog.unshift({...entry, ts: formulaTimestamp()});
  if(formulaFeedLog.length>MAX_FEED)formulaFeedLog.length=MAX_FEED;
  renderFormulaFeed();
}
function renderFormulaFeed(){
  const feed=$('formulaFeed');if(!feed)return;
  const pulse=$('formulaPulse');
  if(!formulaFeedLog.length){feed.innerHTML='<div class="feed-empty">Waiting for research or continuous learning to start…</div>';if(pulse)pulse.className='ops-pulse';return}
  if(pulse)pulse.className='ops-pulse active';
  feed.innerHTML=formulaFeedLog.slice(0,MAX_FEED).map(e=>{
    const statusClass=e.status==='active'?'active':e.status==='promoted'?'promoted':e.status==='rejected'?'rejected':'';
    const tag=e.status==='active'?'<span class="formula-tag active-tag">TESTING</span>':
               e.status==='promoted'?'<span class="formula-tag promote-tag">PROMOTED</span>':
               e.status==='rejected'?'<span class="formula-tag reject-tag">REJECTED</span>':
               '<span class="formula-tag shadow-tag">SHADOW</span>';
    const formulaExpr=e.formula||(e.features||[]).join(' + ')||'candidate feature set';
    const metaParts=[];
    if(e.fold)metaParts.push(`fold ${e.fold}`);
    if(e.win_rate!=null)metaParts.push(`win ${(Number(e.win_rate)*100).toFixed(1)}%`);
    if(e.threshold!=null)metaParts.push(`threshold ${(Number(e.threshold)*100).toFixed(0)}%`);
    if(e.stage)metaParts.push(String(e.stage).replaceAll('_',' '));
    return `<div class="formula-entry ${statusClass}">
      <div class="formula-entry-head"><span class="formula-entry-asset">${esc(e.asset||'—')}${tag}</span><span class="formula-entry-time">${e.ts}</span></div>
      <div class="formula-expr">${esc(formulaExpr)}</div>
      ${metaParts.length?`<div class="formula-meta">${esc(metaParts.join(' · '))}</div>`:''}
    </div>`}).join('');
}

let _lastFormulaKey='';
function syncFormulaFeed(s){
  const t=s.research_training||{},p=s.research_progress||{},d=(t.strategy_discovery||s.strategy_discovery)||{};
  const cl=s.continual_learning||{};
  const active=d.active_trial||null;
  const history=d.trial_history||[];
  // Build a key from current state to avoid redundant pushes
  const key=JSON.stringify([active?.asset,active?.fold,active?.stage,active?.indicators,cl.last_asset,cl.last_indicators,cl.ticks]);
  if(key===_lastFormulaKey)return;
  _lastFormulaKey=key;
  // Active research trial
  if(active?.asset){
    const formulas=active.indicators?.formulas||{};
    const formulaName=Object.keys(formulas)[0]||null;
    const formulaExpr=formulaName?`${formulaName} = ${formulas[formulaName]}`:null;
    pushFormulaEntry({
      asset:active.asset,
      formula:formulaExpr,
      features:active.indicators?.features||[],
      stage:active.stage,
      status:'active',
      win_rate:active.win_rate,
      threshold:active.threshold,
      fold:active.fold!=null?`${active.fold}/${active.total_folds||'—'}`:null
    });
  }
  // Latest trial in history (may be promoted or rejected)
  if(history.length){
    const last=history[history.length-1];
    const lastKey=JSON.stringify([last.asset,last.fold,last.status]);
    if(lastKey!==_lastHistoryKey){
      _lastHistoryKey=lastKey;
      pushFormulaEntry({
        asset:last.asset,
        features:last.indicators?.features||[],
        stage:last.status,
        status:last.status==='candidate_validated'?'promoted':'rejected',
        win_rate:last.win_rate,
        threshold:last.threshold,
        fold:last.fold
      });
    }
  }
  // Continuous learning shadow observation
  if(cl.running&&cl.last_asset&&!active?.asset){
    const shadowKey=`${cl.last_asset}-${cl.ticks}`;
    if(shadowKey!==_lastShadowKey){
      _lastShadowKey=shadowKey;
      pushFormulaEntry({
        asset:cl.last_asset,
        features:cl.last_indicators||[],
        stage:`regime: ${cl.last_regime||'—'} · policy: ${cl.last_action||'—'}`,
        status:'shadow'
      });
    }
  }
}
let _lastHistoryKey='';
let _lastShadowKey='';

function render(s){
  if(window._allMarkets && workspaceState.count > 0){
    renderEightMarketWorkspace(window._allMarkets);
  } else if (workspaceState.count === 0) {
    $('workspaceSection')?.classList.add('hidden');
  }
  $('mode').textContent=s.mode||'—';$('balance').textContent=money(s.balance);$('active').textContent=s.active_trades??'—';$('pnl').textContent=money(s.performance?.total_profit);
  $('mode').textContent=s.mode||'—';$('balance').textContent=money(s.balance);$('active').textContent=s.active_trades??'—';$('pnl').textContent=money(s.performance?.total_profit);
  const c=s.controls||{};
  $('modeSelect').value=s.mode==='REAL'?'REAL':'PRACTICE';
  $('pauseButton').textContent=c.trading_paused?'Resume trading':'Pause trading';
  $('aiTimeframe').checked=!!c.ai_select_timeframe;
  $('confidence').value=Math.round((c.min_confidence??.7)*100);
  $('confidenceOut').value=`${$('confidence').value}%`;
  $('payout').value=c.payout_threshold??80;
  $('maxOpen').value=c.max_open_trades??3;
  // Auto-trade controls
  const autoChk=$('autoTradeEnabled');
  if(autoChk)autoChk.checked=!!c.auto_trade_enabled;
  const autoInt=$('autoTradeInterval');
  if(autoInt&&document.activeElement!==autoInt)autoInt.value=c.auto_trade_interval_seconds??30;
  const autoStatus=$('autoTradeStatus');
  if(autoStatus){
    if(c.auto_trade_enabled&&!c.trading_paused){
      const last=c.auto_trade_last_fired?new Date(c.auto_trade_last_fired*1000).toLocaleTimeString():null;
      autoStatus.textContent=`AUTO-TRADE ACTIVE · ${c.auto_trade_count||0} trades fired${last?` · last: ${last}`:''}`;autoStatus.className='pill online';
    }else{
      autoStatus.textContent=c.auto_trade_enabled?'PAUSED':'DISABLED';autoStatus.className='pill offline';
    }
  }
  const rows=s.trades||[];$('trades').innerHTML=rows.length?rows.map(t=>`<tr><td>${t.asset||'—'}</td><td>${t.direction||'—'}</td><td>${money(t.amount)}</td><td>${t.status||'—'}</td><td>${t.outcome||'UNVERIFIED'}</td><td>${money(t.profit)}</td><td>${t.broker_order_id||'—'}</td></tr>`).join(''):'<tr><td colspan="7" class="empty">No trades recorded.</td></tr>';
  const backtestSnapshot=JSON.stringify([s.backtest,s.backtest_markets||[]]);if(backtestSnapshot!==lastBacktestSnapshot){lastBacktestSnapshot=backtestSnapshot;renderBacktestStable(s.backtest,s.backtest_markets||[])}
  const task=s.backtest||{};const active=['fetching','learning','evaluating','queued'].includes(task.status);const compute=s.compute||{};const accelerator=compute.cuda_available?`${compute.device_name} / CUDA`:compute.device_name||'CPU';const cl=s.continual_learning||{};$('opsPulse').className=`ops-pulse ${active||cl.running?'active':''}`;$('opsHeadline').textContent=active?`${String(task.status).toUpperCase()} · ${task.asset||'Multiple markets'}${task.stage?` · ${task.stage}`:''}`:'Waiting for an operation.';$('opsProgress').value=Number(task.progress||0);$('opsMeta').textContent=active?`Elapsed: ${Number(s.backtest_elapsed_seconds||0).toFixed(1)}s · ${task.detail||'Working'} · broker orders: ${task.orders_placed??0} · compute: ${accelerator}`:`Last task: ${task.status||'idle'} · elapsed: ${Number(s.backtest_elapsed_seconds||0).toFixed(1)}s · broker orders: ${task.orders_placed??0} · compute: ${accelerator}`;$('continualMeta').textContent=`Continuous learning: ${cl.running?'RUNNING':'PAUSED'} · ${cl.ticks||0} ticks · observations ${cl.observations||0} · settled ${cl.resolved_outcomes||0} · pending ${cl.pending_outcomes||0} · shadow replay ${cl.shadow_updates||0} (${cl.candidate_status||'waiting'}) · production mutation: ${cl.production_model_mutated===false?'disabled':'unknown'}${cl.last_error?` · ${cl.last_error}`:''}`;renderResearchConsole(s);
  syncFormulaFeed(s);
}
function renderResearchConsole(s){
  const t=s.research_training||{},r=s.research_report||{},p=s.research_progress||{},d=t.strategy_discovery||{},cl=s.continual_learning||{};
  const running=['starting','running'].includes(t.status)||p.status==='running',progress=Number(p.progress??t.progress??0);
  $('researchTrainStatus').textContent=running?'SEARCHING':r.available?'READY':'NO REPORT';$('researchTrainStatus').className=`pill ${running?'online':r.available?'online':'offline'}`;
  $('startResearchTrain').disabled=running;$('stopResearchTrain').disabled=!running;
  $('researchTrainProgress').textContent=running?`${p.asset||t.asset||'All markets'} · market ${p.market_index||t.market_index||0}/${p.markets_total||t.markets_total||'—'} · ${Math.round(progress*100)}% · ${p.stage||t.detail||'working'} · orders: 0`:`${String(t.status||p.status||'idle').toUpperCase()} · report updated ${r.report_updated||'—'} · orders: ${t.orders_placed??r.orders_placed??0}`;
  $('researchLiveProgress').value=Math.max(0,Math.min(1,progress));$('researchLiveDetail').textContent=p.available?`${String(p.stage||'working').replaceAll('_',' ')} · ${p.detail||'working'} · completed ${p.completed_markets||0}/${p.markets_total||0} · errors ${p.error_markets||0} · updated ${p.updated_at_text||'—'}`:'No live progress available.';
  renderStrategyDiscovery(d,running,cl);
  $('reportMarkets').textContent=r.available?`${r.completed_markets||0}/${r.markets||0}`:'—';$('reportTrades').textContent=r.available?Number(r.oos_trades||0).toLocaleString():'—';$('reportFolds').textContent=r.available?Number(r.walk_forward_splits||0).toLocaleString():'—';$('reportPromoted').textContent=r.available?`${(r.promoted_markets||[]).length}`:'—';$('reportMonotonic').textContent=r.available?`${r.monotonic_confidence_markets||0}/${r.evaluated_confidence_markets||0}`:'—';$('reportMaxConfidence').textContent=r.available?`${(Number(r.max_confidence||0)*100).toFixed(2)}%`:'—';$('reportHighConfidence').textContent=r.available?Number(r.high_confidence_oos_trades||0).toLocaleString():'—';$('reportExact100').textContent=r.available?`${r.exact_100_confidence||0}`:'—';$('reportOrders').textContent=r.available?`${r.orders_placed||0}`:'—';$('researchTrainNote').textContent=r.available?`Schema ${r.schema||'unknown'} · ${Number(r.sequence_model_rows||0).toLocaleString()} sequence rows · under-50 confidence records: ${r.under_50_confidence||0} · errors: ${(r.error_markets||[]).join(', ')||'none'} · promotion remains gated until walk-forward and cost checks pass.`:(r.detail||'Research report unavailable.');
}
function renderStrategyDiscovery(d,running,continual={}){
  const activeResearch=running?d.active_trial:null,shadowTrial=!activeResearch?.asset&&continual.running&&continual.last_asset?{asset:continual.last_asset,stage:'continuous_shadow_observation',fold:'live',total_folds:'live',threshold:0,indicators:{features:continual.last_indicators||[]},win_rate:0,expectancy:0}:{};
  const active=activeResearch||shadowTrial,champion=d.champion||{},history=d.trial_history||[],features=d.feature_importance||active.indicators?.features||[],markets=d.market_results||[];
  lastResearchTrial=active.asset?active:null;
  if(running&&followResearchMarket&&active.asset&&$('marketSelect')?.value!==active.asset&&[...$('marketSelect').options].some(option=>option.value===active.asset)){$('marketSelect').value=active.asset;loadChart()}else if(lastCandleData.length){drawCandles(lastCandleData)}
  $('researchDiscoveryState').textContent=running?`Live telemetry · ${history.length} candidate folds observed`:active.asset?`Continuous shadow observation · tick ${continual.ticks||0} · ${continual.candidate_status||'collecting evidence'}`:'Waiting for a strategy search.';
  $('researchTrialName').textContent=active.asset?`${active.asset} · fold ${active.fold||'—'}/${active.total_folds||'—'}`:'—';
  const formulaNames=Object.keys(active.indicators?.formulas||{}), formulaText=formulaNames.length?`${formulaNames[0]} = ${active.indicators.formulas[formulaNames[0]]}`:''  ;
  $('researchTrialDetail').textContent=active.asset?`${active.stage||'evaluating'} · ${formulaText||active.indicators?.features?.join(', ')||'model feature set'}${active.stage==='continuous_shadow_observation'?` · regime ${continual.last_regime||'—'} · policy ${continual.last_action||'—'} · shadow only`: ` · threshold ${(Number(active.threshold||0)*100).toFixed(0)}% · validation win ${(Number(active.win_rate||0)*100).toFixed(1)}% · expectancy ${Number(active.expectancy||0).toFixed(4)}`}`:'Start a run to see actual candidate folds, features, thresholds, and risk settings.';
  $('researchChampion').textContent=champion.asset?`${champion.asset} · ${(Number(champion.win_rate||0)*100).toFixed(1)}% validation win`:'No validated leader';
  $('researchChampionDetail').textContent=champion.asset?`Fold ${champion.fold||'—'} · threshold ${(Number(champion.threshold||0)*100).toFixed(0)}% · expected edge ${Number(champion.expectancy||0).toFixed(4)}. Promotion still requires held-out and cost gates.`:'Candidates must earn a positive validation edge before they can lead.';
  $('researchFeatureList').innerHTML=features.length?features.slice(0,12).map(x=>`<b>${esc(typeof x==='string'?x:(x.name||x.feature||'signal'))}</b>`).join(''):'<em>Features will appear from the trainer.</em>';
  $('researchTrialCount').textContent=`${history.length} tested`;
  $('researchTrialBars').innerHTML=history.length?history.slice(-24).map((x,i)=>{const score=Math.max(0,Math.min(1,Number(x.win_rate||0))),status=x.status==='candidate_validated'?'valid':'rejected';return `<i class="trial-bar ${status}" style="height:${Math.max(9,Math.round(score*58))}px" title="${esc(x.asset||'candidate')} fold ${x.fold||'—'}: ${(score*100).toFixed(1)}% validation win"><i>${i+1}</i></i>`}).join(''):'<span class="muted small">No candidate telemetry yet.</span>';
  $('researchMarketResults').innerHTML=markets.length?markets.slice(-8).reverse().map(x=>`<div class="market-result ${x.promotion==='promote'?'promote':'reject'}"><b>${esc(x.asset||'market')}</b> · ${(Number(x.win_rate||0)*100).toFixed(1)}% · ${Number(x.trades||0)} OOS · ${esc(x.promotion||'pending')}</div>`).join(''):''  ;
}
async function api(url,opts={}){const r=await fetch(url,{headers:{'Content-Type':'application/json'},...opts});const data=await r.json().catch(()=>({}));if(!r.ok)throw Error(data.detail||`Request failed (${r.status})`);return data}
async function refresh(){try{const s=await api('/api/status');render(s);$('connection').textContent='CONNECTED';$('connection').className='pill online';if(!marketsLoaded){await loadMarkets();marketsLoaded=Boolean($('marketSelect').value&&$('marketSelect').value!=='No live markets');if(!marketsLoaded)setTimeout(refresh,5000)}}catch(e){$('connection').textContent='OFFLINE';$('connection').className='pill offline';note(e.message);setTimeout(refresh,5000)}}
function updateBacktestMarketOptions(){if(!window._allMarkets)return;const minP=Number($('backtestMinPayout')?.value||0);const scope=$('backtestScope').value;const eligible=minP>0?window._allMarkets.filter(m=>m.payout>=minP):window._allMarkets;const options=eligible.map(m=>`<option value="${m.symbol}">${m.symbol} (${m.payout.toFixed(0)}%)</option>`).join('');$('backtestAsset').innerHTML=options||`<option value="">No markets meeting ≥ ${minP}% payout</option>`;if(scope==='all'){[...$('backtestAsset').options].forEach(o=>o.selected=true);$('backtestStatus').textContent=`${eligible.length} market(s) with ≥${minP}% payout ready`;}else if(scope==='single'){if($('backtestAsset').options.length)$('backtestAsset').options[0].selected=true;}}

/* ============================================================
   WORKSPACE STATE & HELPERS
   ============================================================ */
let workspaceState = { count: 0, assets: [] };
try {
  const saved = localStorage.getItem('nexus-workspace');
  workspaceState = saved ? JSON.parse(saved) : { count: 0, assets: [] };
} catch (_) {
  workspaceState = { count: 0, assets: [] };
}
function saveWorkspace() {
  localStorage.setItem('nexus-workspace', JSON.stringify(workspaceState));
}
function isValidMarketSymbol(value) {
  if (value == null) return false;
  const text = String(value).trim();
  if (!text) return false;
  const normalized = text.toLowerCase();
  if (['undefined', 'null', '—', '–', '...', 'loading markets…', 'loading', 'unknown'].includes(normalized)) return false;
  if (/[\u2014\u2013]/.test(text)) return false;
  if (text.includes(' ')) return false;
  return true;
}
function workspaceAssets(markets) {
  const available = (markets || []).map(m => m.symbol).filter(isValidMarketSymbol);
  const requested = (workspaceState.assets || []).filter(asset => isValidMarketSymbol(asset) && available.includes(asset));
  // User's explicit picks: return all of them (up to 12), never capped by count
  if (requested.length) return requested.slice(0, 12);
  // Auto-fill from available up to grid count
  return available.slice(0, Math.max(1, workspaceState.count || 1));
}
function sanitizeWorkspaceState() {
  if (!workspaceState || typeof workspaceState !== 'object') workspaceState = { count: 0, assets: [] };
  const count = Number(workspaceState.count);
  workspaceState.count = Number.isFinite(count) ? Math.min(12, Math.max(0, Math.round(count))) : 0;
  const assets = Array.isArray(workspaceState.assets) ? workspaceState.assets.filter(isValidMarketSymbol) : [];
  if (!Array.isArray(workspaceState.assets) || assets.length !== workspaceState.assets.length) {
    workspaceState.assets = assets;
    saveWorkspace();
  }
}
sanitizeWorkspaceState();
function workspaceColumns(count) {
  if (count <= 1) return '1fr';
  if (count === 2) return 'repeat(2, minmax(0, 1fr))';
  if (count === 3) return 'repeat(3, minmax(0, 1fr))';
  if (count <= 4) return 'repeat(2, minmax(0, 1fr))';
  if (count <= 6) return 'repeat(3, minmax(0, 1fr))';
  return 'repeat(4, minmax(0, 1fr))';
}

let _drawingLock = false;
function drawMiniCandles(data, canvas) {
  if (_drawingLock || !canvas || !canvas.isConnected || !Array.isArray(data) || !data.length) return false;
  _drawingLock = true;
  try {
  const ctx = canvas.getContext('2d'), dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!ctx || !w || !h) return false;
  const targetW = Math.max(1, Math.floor(w * dpr));
  const targetH = Math.max(1, Math.floor(h * dpr));
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const candles = data.slice(-35);
  const highs = candles.map(x => Number(x.high)), lows = candles.map(x => Number(x.low));
  const lo = Math.min(...lows), hi = Math.max(...highs);
  const range = hi - lo || 1;
  const pad = { x: 8, y: 8 };
  const y = v => h - pad.y - (v - lo) / range * (h - pad.y * 2);
  const chartW = w - pad.x * 2;
  const step = chartW / candles.length;
  const bodyW = Math.max(2, Math.floor(step - 2));

  ctx.strokeStyle = '#182432';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.x, Math.round(h / 2));
  ctx.lineTo(w - pad.x, Math.round(h / 2));
  ctx.stroke();

  candles.forEach((c, i) => {
    const x = pad.x + step * i + step / 2;
    const o = Number(c.open), cl = Number(c.close);
    const up = cl >= o;
    const color = up ? '#6df2ad' : '#ff7777';
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(x, y(Number(c.high)));
    ctx.lineTo(x, y(Number(c.low)));
    ctx.stroke();

    const top = Math.min(y(o), y(cl));
    const bodyH = Math.max(2, Math.abs(y(o) - y(cl)));
    ctx.fillRect(Math.round(x - bodyW / 2), Math.round(top), Math.round(bodyW), Math.round(bodyH));
  });

  const lastClose = Number(candles[candles.length - 1].close);
  const isUp = lastClose >= Number(candles[0].open);
  ctx.fillStyle = isUp ? '#6df2ad' : '#ff7777';
  ctx.font = '10px ui-monospace,monospace';
  ctx.textAlign = 'right';
  ctx.fillText(lastClose.toFixed(lastClose > 10 ? 2 : 4), w - 6, 12);
  ctx.textAlign = 'left';
  return true;
  } finally { _drawingLock = false; }
}

/* ResizeObserver to keep canvas pixel dimensions in sync with card layout */
const _canvasObserver = new ResizeObserver(entries => {
  for (const entry of entries) {
    const canvas = entry.target;
    const w = Math.floor(entry.contentRect.width);
    const h = Math.floor(entry.contentRect.height);
    if (!w || !h) continue;
    if (canvas.id === 'chart' && lastCandleData.length) {
      drawCandles(lastCandleData, canvas);
    } else {
      const card = canvas.closest('.workspace-card');
      if (card) {
        const asset = card.dataset.asset || card.querySelector('b')?.textContent;
        if (asset && window._workspaceCanvasData?.[asset]) {
          drawMiniCandles(window._workspaceCanvasData[asset], canvas);
        }
      }
    }
  }
});
window._workspaceCanvasData = {};

async function refreshWorkspaceCharts() {
  if (workspaceState.count === 0 || document.visibilityState === 'hidden') return;
  const root = document.getElementById('marketWorkspace');
  if (!root || $('workspaceSection')?.classList.contains('hidden')) return;
  const tf = Number($('chartTimeframe')?.value || 60);
  for (const card of root.querySelectorAll('.workspace-card')) {
    const asset = card.dataset.asset || card.querySelector('b')?.textContent;
    const canvas = card.querySelector('canvas');
    if (!isValidMarketSymbol(asset) || !canvas || card.dataset.candleRequest === 'pending' || Date.now() < Number(card.dataset.candleNextRetry || 0)) continue;
    try {
      card.dataset.candleRequest = 'pending';
      card.dataset.candleNextRetry = String(Date.now() + 25000);
      const data = await api(`/api/markets/${encodeURIComponent(asset)}/candles?timeframe=${tf}&limit=50`);
      if (!Array.isArray(data) || !data.length) continue;
      window._workspaceCanvasData[asset] = data;
      drawMiniCandles(data, canvas);
    } catch (_) {
    } finally {
      delete card.dataset.candleRequest;
    }
    await new Promise(r => setTimeout(r, 60));
  }
}

async function loadWorkspaceCandles(cards) {
  if (workspaceState.count === 0) return;
  for (const { asset, canvas, card, label } of cards) {
    if (!isValidMarketSymbol(asset) || !canvas.isConnected) continue;
    try {
      const data = await Promise.race([
        api(`/api/markets/${encodeURIComponent(asset)}/candles?timeframe=60&limit=50`),
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 8000))
      ]);
      if (Array.isArray(data) && data.length) {
        window._workspaceCanvasData[asset] = data;
        drawMiniCandles(data, canvas);
        const status = card.querySelector('.workspace-status');
        if (status) status.textContent = `${label}% payout`;
      }
    } catch (_) {
      const status = card.querySelector('.workspace-status');
      if (status) status.textContent = `${label}% payout`;
    } finally {
      delete card.dataset.candleRequest;
    }
    await new Promise(r => setTimeout(r, 60));
  }
}

function normalizeMarket(m) {
  const raw = String(m?.symbol ?? m?.asset ?? m?.name ?? '').trim();
  const symbol = isValidMarketSymbol(raw) ? raw : '';
  const payout = Number(m?.payout ?? m?.market_payout ?? 0);
  return { symbol, payout: Number.isFinite(payout) ? payout : 0 };
}

function populateWorkspaceChips(markets, selected) {
  const container = $('workspaceMarketChips');
  if (!container) return;
  const normalized = [...markets].map(normalizeMarket).filter(m => m.symbol);
  const selectedSymbols = new Set((selected || []).map(s => typeof s === 'string' ? s : s?.symbol));

  const sorted = [...normalized].sort((a, b) => {
    const aSel = selectedSymbols.has(a.symbol) ? 1 : 0;
    const bSel = selectedSymbols.has(b.symbol) ? 1 : 0;
    if (aSel !== bSel) return bSel - aSel;
    return (b.payout || 0) - (a.payout || 0);
  });

  const filterText = ($('wsSearchInput')?.value || '').trim().toLowerCase();
  container.innerHTML = '';
  if (!sorted.length) {
    container.innerHTML = '<span class="muted small">No live markets loaded.</span>';
    return;
  }

  for (const m of sorted) {
    const isSel = selectedSymbols.has(m.symbol);
    const chip = document.createElement('div');
    chip.className = `market-chip ${isSel ? 'selected' : ''}`;
    chip.dataset.symbol = m.symbol;
    chip.innerHTML = `<span>${esc(m.symbol)}</span><span class="chip-payout">${Number(m.payout || 0).toFixed(0)}%</span>`;

    if (filterText && !m.symbol.toLowerCase().includes(filterText)) {
      chip.style.display = 'none';
    }

    chip.onclick = () => {
      let current = new Set(workspaceState.assets || []);
      if (current.has(m.symbol)) {
        current.delete(m.symbol);
      } else {
        if (current.size >= 12) {
          note('Maximum 12 markets can be displayed in the grid.');
          return;
        }
        current.add(m.symbol);
      }
      workspaceState.assets = [...current];
      workspaceState.count = Math.max(1, workspaceState.assets.length);
      const wsSelect = $('workspaceCount');
      if (wsSelect) wsSelect.value = String(workspaceState.count);
      saveWorkspace();
      renderEightMarketWorkspace(markets);
      syncAutoTradeMarkets();
    };

    container.appendChild(chip);
  }

  const displayedMarketsSelect = $('workspaceAssets');
  if (displayedMarketsSelect) {
    displayedMarketsSelect.innerHTML = sorted.map(m =>
      `<option value="${esc(m.symbol)}" ${selectedSymbols.has(m.symbol) ? 'selected' : ''}>${esc(m.symbol)}</option>`
    ).join('');
  }
}

async function syncAutoTradeMarkets() {
  const autoChk = $('autoTradeEnabled');
  if (!autoChk || !autoChk.checked) return;
  const targetMode = $('autoTradeTarget')?.value || 'grid';
  let targetAssets = [];
  if (targetMode === 'current') {
    const curr = $('marketSelect')?.value;
    if (isValidMarketSymbol(curr)) targetAssets = [curr];
  } else if (targetMode === 'grid') {
    targetAssets = workspaceAssets(window._allMarkets || []);
  }
  try {
    await api('/api/control', {
      method: 'POST',
      body: JSON.stringify({ auto_trade_assets: targetAssets }),
    });
  } catch (_) {}
}

async function renderEightMarketWorkspace(markets) {
  const wsSection = $('workspaceSection');
  const wsSelect = $('workspaceCount');
  if (wsSelect && wsSelect.value !== String(workspaceState.count)) {
    wsSelect.value = String(workspaceState.count);
  }

  if (workspaceState.count === 0) {
    if (wsSection) wsSection.classList.add('hidden');
    return;
  }

  if (wsSection) wsSection.classList.remove('hidden');
  const root = document.getElementById('marketWorkspace');
  if (!root) return;

  const normalizedMarkets = [...markets].map(normalizeMarket).filter(m => m.symbol);
  const selected = workspaceAssets(normalizedMarkets);
  const fallback = normalizedMarkets.slice(0, Math.max(1, workspaceState.count));
  const visibleMarkets = selected.length
    ? selected.map(symbol => normalizedMarkets.find(market => market.symbol === symbol)).filter(Boolean)
    : fallback;

  const countBadge = $('workspaceCountBadge');
  if (countBadge) {
    countBadge.textContent = `${visibleMarkets.length} market${visibleMarkets.length === 1 ? '' : 's'}`;
  }

  const key = JSON.stringify(visibleMarkets.map(m => m.symbol));
  root.style.gridTemplateColumns = workspaceColumns(workspaceState.count);

  populateWorkspaceChips(normalizedMarkets, visibleMarkets);

  const applyScope = () => {
    const assets = workspaceAssets(markets);
    $('researchMarketScope').value = 'selected';
    $('researchRunAssets').disabled = false;
    [...$('researchRunAssets').options].forEach(o => o.selected = assets.includes(o.value));
    return assets;
  };
  const btnTrain = $('workspaceTrain');
  if (btnTrain) btnTrain.onclick = () => {
    const assets = applyScope();
    note(`Research scoped to ${assets.length} displayed market(s). Click "Start strategy search" to begin.`);
  };
  const btnBacktest = $('workspaceBacktest');
  if (btnBacktest) btnBacktest.onclick = () => {
    const assets = workspaceAssets(markets);
    $('backtestScope').value = assets.length === 1 ? 'single' : 'multiple';
    updateBacktestMarketOptions();
    [...$('backtestAsset').options].forEach(o => o.selected = assets.includes(o.value));
    note(`Backtest scoped to ${assets.length} displayed market(s).`);
  };
  const btnPractice = $('workspacePractice');
  if (btnPractice) btnPractice.onclick = async () => {
    const assets = workspaceAssets(markets);
    if (prompt(`Type RUN PRACTICE WORKSPACE to scan ${assets.length} displayed markets:`) !== 'RUN PRACTICE WORKSPACE') return;
    try {
      const result = await api('/api/workspace/practice-scan', { method: 'POST', body: JSON.stringify({ assets, confirmation: 'RUN PRACTICE WORKSPACE' }) });
      note(`${result.confirmed_orders} broker-confirmed PRACTICE order(s). ${result.results.map(x => `${x.asset}: ${x.status}`).join(' · ')}`);
      refresh();
    } catch (e) { note(e.message); }
  };

  /* --- DOM diffing: only add/remove cards that actually changed --- */
  const currentCards = new Map([...root.children].map(c => [c.dataset.asset, c]));
  const desiredSymbols = new Set(visibleMarkets.map(m => m.symbol).filter(isValidMarketSymbol));

  /* If nothing changed, skip entirely */
  if (root.dataset.key === key && root.children.length === visibleMarkets.length) {
    return;
  }
  root.dataset.key = key;

  /* Remove cards that are no longer in the desired set */
  for (const [asset, card] of currentCards) {
    if (!desiredSymbols.has(asset)) {
      const canvas = card.querySelector('canvas');
      if (canvas) _canvasObserver.unobserve(canvas);
      card.remove();
      currentCards.delete(asset);
    }
  }

  const candleQueue = [];
  for (const market of visibleMarkets) {
    if (!isValidMarketSymbol(market?.symbol)) continue;
    const payout = Number(market?.payout ?? 0);
    const label = Number.isFinite(payout) ? payout.toFixed(0) : 0;
    let card = currentCards.get(market.symbol);

    if (card) {
      /* Update existing card in place */
      const titleEl = card.querySelector('b');
      if (titleEl && titleEl.textContent !== market.symbol) titleEl.textContent = market.symbol;
      const statusEl = card.querySelector('.workspace-status');
      if (statusEl) statusEl.textContent = `${label}% payout`;
      let removeBtn = card.querySelector('.workspace-card-remove');
      if (!removeBtn) {
        const head = card.querySelector('.workspace-card-head');
        if (head) {
          const wrap = document.createElement('div');
          wrap.className = 'workspace-card-actions';
          if (statusEl) wrap.appendChild(statusEl);
          removeBtn = document.createElement('button');
          removeBtn.type = 'button';
          removeBtn.className = 'workspace-card-remove';
          removeBtn.title = 'Remove from grid';
          removeBtn.dataset.asset = market.symbol;
          removeBtn.textContent = '✕';
          wrap.appendChild(removeBtn);
          head.appendChild(wrap);
        }
      }
    } else {
      /* Create new card */
      card = document.createElement('div');
      card.className = 'workspace-card';
      card.dataset.asset = market.symbol;
      card.innerHTML = `<div class="workspace-card-head"><b>${esc(market.symbol)}</b><div class="workspace-card-actions"><span class="workspace-status">${label}% payout</span><button type="button" class="workspace-card-remove" title="Remove from grid" data-asset="${esc(market.symbol)}">✕</button></div></div><canvas></canvas><small class="eligibility">Checking eligibility…</small>`;
      card.onclick = (e) => {
        if (e.target.closest('.workspace-card-remove')) {
          e.stopPropagation();
          const assetToRemove = market.symbol;
          workspaceState.assets = (workspaceState.assets || []).filter(a => a !== assetToRemove);
          workspaceState.count = Math.max(0, workspaceState.assets.length);
          const wsSelect = $('workspaceCount');
          if (wsSelect) wsSelect.value = String(workspaceState.count);
          saveWorkspace();
          renderEightMarketWorkspace(markets);
          syncAutoTradeMarkets();
          return;
        }
        $('marketSelect').value = market.symbol;
        loadChart();
      };
      root.append(card);
      const canvas = card.querySelector('canvas');
      _canvasObserver.observe(canvas);
      card.dataset.candleRequest = 'pending';
      card.dataset.candleNextRetry = String(Date.now() + 20000);
      candleQueue.push({ asset: market.symbol, canvas, card, label });
    }
  }

  /* Reorder cards to match visibleMarkets order */
  const ordered = visibleMarkets.map(m => root.querySelector(`.workspace-card[data-asset="${CSS.escape(m.symbol)}"]`)).filter(Boolean);
  for (const card of ordered) root.append(card);

  if (candleQueue.length) void loadWorkspaceCandles(candleQueue).then(() => hydrateWorkspaceEligibility());
}

async function loadMarkets() {
  try {
    const markets = await api('/api/markets');
    const normalized = [...markets].map(normalizeMarket).filter(m => m.symbol);
    window._allMarkets = normalized;
    const old = $('marketSelect').value;
    const options = normalized.map(m => `<option value="${m.symbol}">${m.symbol} (${Number(m.payout || 0).toFixed(0)}%)</option>`).join('');
    $('marketSelect').innerHTML = options || '<option>No live markets</option>';
    $('researchRunAssets').innerHTML = options || '<option value="">No broker markets available</option>';
    $('researchRunAssets').disabled = !normalized.length || $('researchMarketScope').value !== 'selected';
    updateBacktestMarketOptions();
    if (old && [...$('marketSelect').options].some(o => o.value === old)) $('marketSelect').value = old;
    if ($('marketSelect').value) $('backtestAsset').value = $('marketSelect').value;
    await loadChart();

    const wsSelect = $('workspaceCount');
    if (wsSelect) {
      wsSelect.value = String(workspaceState.count);
      wsSelect.onchange = () => {
        workspaceState.count = Number(wsSelect.value);
        saveWorkspace();
        renderEightMarketWorkspace(normalized);
        if (workspaceState.count > 0) {
          refreshWorkspaceCharts();
          hydrateWorkspaceEligibility();
        }
      };
    }

    if (workspaceState.count > 0) {
      await renderEightMarketWorkspace(normalized);
      hydrateWorkspaceEligibility();
    } else {
      $('workspaceSection')?.classList.add('hidden');
    }
  } catch (e) {
    note(`Market refresh unavailable: ${e.message}`);
    if (!$('marketSelect').options.length || $('marketSelect').value === 'No live markets') $('marketSelect').innerHTML = '<option value="">No live markets — refresh login</option>';
    if (!$('backtestAsset').options.length || $('backtestAsset').value === 'No live markets') $('backtestAsset').innerHTML = '<option value="">No live markets — refresh login</option>';
  }
}

async function hydrateWorkspaceEligibility() {
  if (workspaceState.count === 0) return;
  const wsSection = $('workspaceSection');
  if (!wsSection || wsSection.classList.contains('hidden')) return;
  const cards = [...document.querySelectorAll('.workspace-card')];
  for (const card of cards) {
    const asset = card.dataset.asset || card.querySelector('b')?.textContent;
    if (!isValidMarketSymbol(asset) || card.dataset.eligibility === 'pending' || Date.now() < Number(card.dataset.eligibilityNextRetry || 0)) continue;
    card.dataset.eligibility = 'pending';
    card.dataset.eligibilityNextRetry = String(Date.now() + 25000);
    try {
      const item = await Promise.race([
        api(`/api/markets/${encodeURIComponent(asset)}/eligibility`),
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 8000))
      ]);
      let badge = card.querySelector('.eligibility');
      if (!badge) {
        badge = document.createElement('small');
        badge.className = 'eligibility';
        card.append(badge);
      }
      badge.className = `eligibility ${item.eligible ? 'eligible' : 'blocked'}`;
      badge.textContent = item.eligible ? `✓ Eligible · ${item.signal} ${(Number(item.confidence) * 100).toFixed(0)}%` : `✗ Blocked: ${(item.reasons || [])[0] || 'no edge'}`;
      badge.title = (item.reasons || []).join('\n');
      card.dataset.eligibility = item.eligible ? 'eligible' : 'blocked';
    } catch (_) {
      card.dataset.eligibility = 'error';
      const badge = card.querySelector('.eligibility');
      if (badge) {
        badge.textContent = 'Eligibility queued';
        badge.className = 'eligibility';
      }
    }
    await new Promise(r => setTimeout(r, 60));
  }
}

function drawCandles(data,canvas=$('chart')){
  if(_drawingLock||!canvas||!canvas.isConnected||!Array.isArray(data)||!data.length)return false;
  _drawingLock=true;try{
  const ctx=canvas.getContext('2d'),dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
  if(!ctx||!w||!h){_drawingLock=false;return false}
  canvas.width=Math.max(1,Math.floor(w*dpr));canvas.height=Math.max(1,Math.floor(h*dpr));ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  const highs=data.map(x=>Number(x.high)),lows=data.map(x=>Number(x.low)),lo=Math.min(...lows),hi=Math.max(...highs),range=hi-lo||1,pad={x:28,y:22};
  const y=v=>h-pad.y-(v-lo)/range*(h-pad.y*2), chartW=w-pad.x*2, step=chartW/data.length, body=Math.max(3,step*.58);
  ctx.strokeStyle='#243342';ctx.lineWidth=1;for(let i=1;i<5;i++){const gy=pad.y+(h-pad.y*2)*i/5;ctx.beginPath();ctx.moveTo(pad.x,gy);ctx.lineTo(w-pad.x,gy);ctx.stroke()}
  data.forEach((c,i)=>{const x=pad.x+step*i+step/2,o=Number(c.open),cl=Number(c.close),up=cl>=o,color=up?'#6df2ad':'#ff7777';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,y(Number(c.high)));ctx.lineTo(x,y(Number(c.low)));ctx.stroke();const top=Math.min(y(o),y(cl)),height=Math.max(2,Math.abs(y(o)-y(cl)));ctx.fillRect(x-body/2,top,body,height)});
  const active=lastResearchTrial,showResearch=canvas.id==='chart'&&active&&active.asset===$('marketSelect').value;
  if(showResearch){const ema=span=>{const alpha=2/(span+1);let value=Number(data[0].close);return data.map(c=>value=alpha*Number(c.close)+(1-alpha)*value)};const drawLine=(values,color)=>{ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();values.forEach((value,i)=>i?ctx.lineTo(pad.x+step*i+step/2,y(value)):ctx.moveTo(pad.x+step/2,y(value)));ctx.stroke()};drawLine(ema(8),'#ffd166');drawLine(ema(34),'#b388ff');const activeIndicators=[...new Set(active.indicators?.features||[])].slice(0,6);ctx.fillStyle='#dbeafe';ctx.fillRect(pad.x+5,22,Math.min(w-pad.x-10,560),50);ctx.fillStyle='#0a0e14';ctx.font='10px system-ui';const formula=Object.keys(active.indicators?.formulas||{})[0]||'candidate features';ctx.fillText(`RESEARCH TEST · ${active.stage||'evaluating'} · ${formula} · threshold ${(Number(active.threshold||0)*100).toFixed(0)}% · fold ${active.fold||'—'}/${active.total_folds||'—'}`,pad.x+10,36);ctx.fillStyle='#ffd166';ctx.fillText('PLOTTED: EMA 8',pad.x+10,50);ctx.fillStyle='#b388ff';ctx.fillText('EMA 34',pad.x+104,50);ctx.fillStyle='#0a0e14';ctx.fillText(`TESTING: ${activeIndicators.join(' · ')||'candidate feature set'}`,pad.x+10,64)}
  ctx.fillStyle='#8290a5';ctx.font='11px system-ui';ctx.fillText(hi.toFixed(5),4,pad.y+4);ctx.fillText(lo.toFixed(5),4,h-pad.y);ctx.fillText(showResearch?'LIVE RESEARCH OVERLAY':'LIVE OHLC',w-(showResearch?142:82),16);
  return true;
  }finally{_drawingLock=false}
}
async function loadChart(){const asset=$('marketSelect').value;if(!asset)return;$('analysis').textContent=`Loading live Quotex candles for ${asset}…`;try{const tf=$('chartTimeframe').value;const data=await api(`/api/markets/${encodeURIComponent(asset)}/candles?timeframe=${tf}&limit=120`);if(!Array.isArray(data)||!data.length)throw Error('No candles returned for this market');lastCandleData=data;const canvas=$('chart');if(!canvas)throw Error('Chart canvas is temporarily unavailable; retrying');if(!drawCandles(data,canvas))throw Error('Chart canvas has no drawable size; retrying');const a=await api(`/api/markets/${encodeURIComponent(asset)}/analysis`);const trial=lastResearchTrial?.asset===asset?`<br><span>Research overlay: ${lastResearchTrial.stage||'evaluating'} · threshold ${(Number(lastResearchTrial.threshold||0)*100).toFixed(0)}% · live historical-candle candidate, no orders.</span>`:'';$('analysis').innerHTML=`<b>${asset}</b> · Signal: <b>${String(a.signal||'HOLD').toUpperCase()}</b> · Confidence: ${(Number(a.confidence||0)*100).toFixed(1)}% · AI timeframe: ${a.recommended_expiration||'—'}s<br><span>${a.reasoning||'Live broker analysis'}</span>${trial}`}catch(e){$('analysis').textContent=`Live candle data unavailable for ${asset}: ${e.message}. Choose another market or refresh.`;note(e.message)}}
$('refresh').onclick=refresh;$('confidence').oninput=()=>{$('confidenceOut').value=`${$('confidence').value}%`};$('marketSelect').onchange=loadChart;$('chartTimeframe').onchange=loadChart;
$('pauseButton').onclick=async()=>{try{const c=await api('/api/control',{method:'POST',body:JSON.stringify({pause:$('pauseButton').textContent!=='Resume trading'})});$('pauseButton').textContent=c.trading_paused?'Resume trading':'Pause trading';note('Trading control updated')}catch(e){note(e.message)}};
$('panicButton').onclick=async()=>{try{await api('/api/control',{method:'POST',body:JSON.stringify({pause:true})});note('PANIC STOP active; trading paused')}catch(e){note(e.message)}};
// Wire Workspace Market Quick Action buttons
function addMarketToGrid(symbol) {
  if (!isValidMarketSymbol(symbol)) return;
  const current = new Set(workspaceState.assets || []);
  if (current.size >= 12 && !current.has(symbol)) {
    note('Maximum 12 markets can be displayed in the grid.');
    return;
  }
  current.add(symbol);
  workspaceState.assets = [...current];
  workspaceState.count = Math.max(1, workspaceState.assets.length);
  const wsSelect = $('workspaceCount');
  if (wsSelect) wsSelect.value = String(workspaceState.count);
  saveWorkspace();
  renderEightMarketWorkspace(window._allMarkets || []);
  syncAutoTradeMarkets();
  note(`Added ${symbol} to grid workspace`);
}

const btnAddCurrent = $('addCurrentToGrid');
if (btnAddCurrent) btnAddCurrent.onclick = () => addMarketToGrid($('marketSelect')?.value);
const wsAddActive = $('wsAddActive');
if (wsAddActive) wsAddActive.onclick = () => addMarketToGrid($('marketSelect')?.value);

const wsTop4 = $('wsTop4');
if (wsTop4) wsTop4.onclick = () => {
  const sorted = [...(window._allMarkets || [])].map(normalizeMarket).filter(m => m.symbol).sort((a, b) => (b.payout || 0) - (a.payout || 0));
  workspaceState.assets = sorted.slice(0, 4).map(m => m.symbol);
  workspaceState.count = Math.max(1, workspaceState.assets.length);
  const wsSelect = $('workspaceCount');
  if (wsSelect) wsSelect.value = String(workspaceState.count);
  saveWorkspace();
  renderEightMarketWorkspace(window._allMarkets || []);
  syncAutoTradeMarkets();
  note('Selected top 4 payout markets for grid');
};

const wsTop8 = $('wsTop8');
if (wsTop8) wsTop8.onclick = () => {
  const sorted = [...(window._allMarkets || [])].map(normalizeMarket).filter(m => m.symbol).sort((a, b) => (b.payout || 0) - (a.payout || 0));
  workspaceState.assets = sorted.slice(0, 8).map(m => m.symbol);
  workspaceState.count = Math.max(1, workspaceState.assets.length);
  const wsSelect = $('workspaceCount');
  if (wsSelect) wsSelect.value = String(workspaceState.count);
  saveWorkspace();
  renderEightMarketWorkspace(window._allMarkets || []);
  syncAutoTradeMarkets();
  note('Selected top 8 payout markets for grid');
};

const wsSelectAll = $('wsSelectAll');
if (wsSelectAll) wsSelectAll.onclick = () => {
  const available = [...(window._allMarkets || [])].map(normalizeMarket).filter(m => m.symbol);
  workspaceState.assets = available.slice(0, 12).map(m => m.symbol);
  workspaceState.count = Math.max(1, workspaceState.assets.length);
  const wsSelect = $('workspaceCount');
  if (wsSelect) wsSelect.value = String(workspaceState.count);
  saveWorkspace();
  renderEightMarketWorkspace(window._allMarkets || []);
  syncAutoTradeMarkets();
  note(`Selected all ${workspaceState.assets.length} markets for grid`);
};

const wsClearAll = $('wsClearAll');
if (wsClearAll) wsClearAll.onclick = () => {
  workspaceState.assets = [];
  workspaceState.count = 0;
  const wsSelect = $('workspaceCount');
  if (wsSelect) wsSelect.value = '0';
  saveWorkspace();
  renderEightMarketWorkspace(window._allMarkets || []);
  syncAutoTradeMarkets();
  note('Grid cleared');
};

const wsSearchInput = $('wsSearchInput');
if (wsSearchInput) {
  wsSearchInput.oninput = () => {
    const text = wsSearchInput.value.trim().toLowerCase();
    const container = $('workspaceMarketChips');
    if (!container) return;
    for (const chip of container.querySelectorAll('.market-chip')) {
      const sym = (chip.dataset.symbol || '').toLowerCase();
      chip.style.display = !text || sym.includes(text) ? 'inline-flex' : 'none';
    }
  };
}

$('saveControls').onclick = async () => {
  try {
    const targetMode = $('autoTradeTarget')?.value || 'grid';
    let targetAssets = [];
    if (targetMode === 'current') {
      const curr = $('marketSelect')?.value;
      if (isValidMarketSymbol(curr)) targetAssets = [curr];
    } else if (targetMode === 'grid') {
      targetAssets = workspaceAssets(window._allMarkets || []);
    }
    await api('/api/control', {
      method: 'POST',
      body: JSON.stringify({
        ai_select_timeframe: $('aiTimeframe').checked,
        min_confidence: Number($('confidence').value) / 100,
        payout_threshold: Number($('payout').value),
        max_open_trades: Number($('maxOpen').value),
        base_trade_amount: Number($('baseTradeAmount')?.value || 10),
        auto_trade_interval_seconds: Number($('autoTradeInterval')?.value || 15),
        auto_trade_assets: targetAssets,
      }),
    });
    note('Controls saved');
  } catch (e) {
    note(e.message);
  }
};

const autoChkEl = $('autoTradeEnabled');
if (autoChkEl) {
  autoChkEl.onchange = async () => {
    try {
      const targetMode = $('autoTradeTarget')?.value || 'grid';
      let targetAssets = [];
      if (targetMode === 'current') {
        const curr = $('marketSelect')?.value;
        if (isValidMarketSymbol(curr)) targetAssets = [curr];
      } else if (targetMode === 'grid') {
        targetAssets = workspaceAssets(window._allMarkets || []);
      }
      await api('/api/control', {
        method: 'POST',
        body: JSON.stringify({
          auto_trade_enabled: autoChkEl.checked,
          auto_trade_assets: targetAssets,
          auto_trade_interval_seconds: Number($('autoTradeInterval')?.value || 15),
        }),
      });
      note(
        autoChkEl.checked
          ? 'Auto-trading ENABLED — will execute practice orders when resumed'
          : 'Auto-trading DISABLED'
      );
    } catch (e) {
      note(e.message);
    }
  };
}

const autoTargetEl = $('autoTradeTarget');
if (autoTargetEl) {
  autoTargetEl.onchange = syncAutoTradeMarkets;
}
$('modeButton').onclick=async()=>{const mode=$('modeSelect').value;const confirmation=mode==='REAL'?prompt('Type I UNDERSTAND REAL TRADING to continue:'):'SWITCH TO PRACTICE';if(confirmation===null)return;try{await api('/api/account/mode',{method:'POST',body:JSON.stringify({mode,confirmation})});note(`Switched to ${mode}; trading remains paused`);refresh()}catch(e){note(e.message)}};
$('clearHistory').onclick=async()=>{if(prompt('Type CLEAR HISTORY to delete resolved records:')!=='CLEAR HISTORY')return;try{const d=await api('/api/history/clear',{method:'POST',body:JSON.stringify({confirmation:'CLEAR HISTORY',include_unresolved:false})});note(`Deleted ${d.deleted} resolved records; unresolved records retained`);refresh()}catch(e){note(e.message)}};
$('clearAllHistory').onclick=async()=>{if(prompt('This removes every ledger record, including unresolved orders. Type CLEAR ALL HISTORY:')!=='CLEAR ALL HISTORY')return;try{const d=await api('/api/history/clear',{method:'POST',body:JSON.stringify({confirmation:'CLEAR HISTORY',include_unresolved:true})});note(`Deleted ${d.deleted} records`);refresh()}catch(e){note(e.message)}};
$('researchMarketScope').onchange=()=>{$('researchRunAssets').disabled=$('researchMarketScope').value!=='selected'};
$('startResearchTrain').onclick=async()=>{const marketScope=$('researchMarketScope').value,assets=[...$('researchRunAssets').selectedOptions].map(o=>o.value).filter(Boolean);if(marketScope==='selected'&&!assets.length){note('Select at least one market for a selected-market strategy search.');return}try{const d=await api('/api/research/retrain',{method:'POST',body:JSON.stringify({market_scope:marketScope,assets,min_payout:Number($('researchMinPayout').value),search_depth:$('researchSearchDepth').value,indicator_scope:$('researchIndicatorScope').value,folds:Number($('researchFolds').value)})});note(d.detail||'Robust research retraining started');refresh()}catch(e){note(e.message)}};$('stopResearchTrain').onclick=async()=>{try{await api('/api/research/stop',{method:'POST'});note('Stopping robust research retraining')}catch(e){note(e.message)}};$('refreshLogin').onclick=async()=>{try{await api('/api/login/refresh',{method:'POST'});note('Broker login refresh started; keep this tab open until it completes')}catch(e){note(e.message)}};
const shutdownButton=document.getElementById('shutdownButton')||document.createElement('button');shutdownButton.id='shutdownButton';shutdownButton.className='danger';shutdownButton.textContent='Shutdown server';if(!shutdownButton.parentElement)$('clearAllHistory').after(shutdownButton);shutdownButton.onclick=async()=>{if(prompt('Type SHUTDOWN NEXUS to close the server:')!=='SHUTDOWN NEXUS')return;shutdownButton.disabled=true;note('NEXUS server is shutting down…');try{await api('/api/shutdown',{method:'POST'})}catch(e){note('Server shutdown requested')}setTimeout(()=>{document.body.innerHTML='<main style="padding:3rem;font-family:system-ui"><h1>NEXUS is shut down</h1><p>You can close this tab. Restart the server from the terminal when needed.</p></main>'},700)};
let backtestPoll=null;
function renderBacktest(s,markets=[]){if(!s)return;const progress=s.progress==null?'':` · ${(Number(s.progress)*100).toFixed(0)}%`;const marketProgress=s.market_index&&s.markets_total?` · market ${s.market_index}/${s.markets_total}`:'';const stage=s.stage?` · ${s.stage}`:'';const detail=s.detail?`<br><span>${s.detail}</span>`:'';const active=['fetching','learning','evaluating','queued'].includes(s.status);const recoverable=active||['error','cancelled'].includes(s.status);const lH=s.learning_hours||5,eH=s.evaluation_hours||1,minC=Math.round(Number(s.min_confidence||0.6)*100),maxC=s.max_confidence?Math.round(Number(s.max_confidence)*100):null;const confNote=maxC?` (highest signal reached ${maxC}%, below ${minC}% threshold; lower to 60% for trades)`:` (all signals remained below ${minC}% confidence threshold)`;const label=s.status==='complete'?`Complete · ${s.markets_total||1} market(s) · ${s.trades_scored||0} scored · ${s.wins||0} wins / ${s.losses||0} losses · ${(Number(s.win_rate||0)*100).toFixed(1)}% win rate · ${money(s.net_profit)}`:`${String(s.status||'idle').toUpperCase()}${marketProgress}${stage}${progress} · ${s.orders_placed??0} broker orders · live`;$('backtestStatus').innerHTML=label;$('stopBacktest').disabled=!recoverable;$('stopBacktest').textContent=active?'Stop':s.status==='error'?'Reset':'Clear';$('startBacktest').disabled=active;$('backtestMarkets').innerHTML=markets.length?markets.map(m=>`<div class="market-progress-row"><b>${m.asset}</b><span>${String(m.status||'queued').toUpperCase()}${m.stage?` · ${m.stage}`:''}</span><progress max="1" value="${Number(m.progress||0)}"></progress><small>${m.detail||`${Math.round(Number(m.progress||0)*100)}%`}</small></div>`).join(''):'<span class="muted small">No active market training.</span>';if(s.status==='learning')$('backtestResult').innerHTML=`<b>Training ${s.asset||''}</b>${detail}<br>Learning candles: ${s.learning_candles||'—'} · Evaluation candles queued: ${s.evaluation_candles||'—'} · broker orders: <b>0</b>`;else if(s.status==='complete'){$('backtestResult').innerHTML=s.markets?`<b>Local replay complete:</b> ${lH}h learned + ${eH}h evaluated across ${s.markets_total} markets (${s.decisions||s.markets_total*59} bars analyzed) · <b>${s.trades_scored||0} trades executed</b>${(s.trades_scored||0)===0?confNote:` · win rate ${(Number(s.win_rate||0)*100).toFixed(1)}%`} · net P&amp;L <b>${money(s.net_profit)}</b> · orders placed: <b>0</b>`:`<b>Local replay only:</b> ${lH}h learned + ${eH}h evaluated · ${s.candles} Quotex candles · payout ${Number(s.payout_percent||0).toFixed(1)}% · <b>${s.trades_scored||0} scored</b>${(s.trades_scored||0)===0?confNote:''}<br>AI used expirations: ${(s.timeframes_used||[]).join(', ')||'none'}s · amount range: ${money(s.amount_min)}–${money(s.amount_max)} · online updates: ${s.learning_updates||0} · orders placed: <b>0</b>`}else if(s.status==='error')$('backtestResult').textContent=`Backtest error: ${s.error||'unknown error'}`;if(['complete','error','cancelled'].includes(s.status)&&backtestPoll){clearInterval(backtestPoll);backtestPoll=null}}
async function pollBacktest(){try{renderBacktest(await api('/api/backtest/status'))}catch(e){note(e.message)}}
$('backtestStakeMode').onchange=()=>{$('fixedStakeWrap').classList.toggle('hidden',$('backtestStakeMode').value!=='fixed')};
$('backtestScope').onchange=()=>{
  $('backtestAsset').multiple=$('backtestScope').value!=='single';
  updateBacktestMarketOptions();
};
if($('backtestMinPayout')){$('backtestMinPayout').oninput=updateBacktestMarketOptions;$('backtestMinPayout').onchange=updateBacktestMarketOptions;}
$('backtestAsset').onchange=()=>{const selected=[...$('backtestAsset').selectedOptions];if($('backtestScope').value==='single'&&selected[0]){$('marketSelect').value=selected[0].value;loadChart()}};
$('startBacktest').onclick=async()=>{try{
  const minPayout=Number($('backtestMinPayout')?.value||$('payout')?.value||0);
  const rawConf=Number($('backtestConfidence')?.value||$('confidence')?.value||60);
  const minConf=rawConf>1?rawConf/100:rawConf;
  const selected=[...$('backtestAsset').selectedOptions].map(o=>o.value).filter(Boolean);
  const aiTime=$('backtestTradeTime').value==='ai';
  const body={
    asset:selected[0]||null,
    assets:selected,
    market_scope:$('backtestScope').value,
    learning_hours:Number($('learningHours').value)||10,
    evaluation_hours:Number($('evaluationHours').value)||2,
    candle_timeframe:Number($('backtestTimeframe').value)||60,
    ai_select_timeframe:aiTime,
    trade_expiration:Number(aiTime?60:$('backtestTradeTime').value)||60,
    stake_mode:$('backtestStakeMode').value,
    min_payout:minPayout,
    min_confidence:minConf
  };
  if($('backtestScope').value==='all')body.assets=[];
  if($('backtestStakeMode').value==='fixed')body.stake=Number($('fixedStake').value)||10;
  const result=await api('/api/backtest/start',{method:'POST',body:JSON.stringify(body)});
  renderBacktest(result);
  await pollBacktest();
  note(`Backtest started with ≥${minPayout}% payout filter; no broker orders placed`);
}catch(e){note(e.message)}};
$('stopBacktest').onclick=async()=>{try{await api('/api/backtest/stop',{method:'POST'});note('Stopping local backtest')}catch(e){note(e.message)}};
refresh();
function connectWs(){try{const ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);ws.onopen=()=>{$('connection').textContent='LIVE';$('connection').className='pill online'};ws.onmessage=e=>{try{render(JSON.parse(e.data))}catch(error){note(`Live update error: ${error.message}`)}};ws.onclose=()=>{$('connection').textContent='RECONNECTING';$('connection').className='pill offline';setTimeout(connectWs,3000)}}catch(e){setTimeout(connectWs,3000)}}
connectWs();
setInterval(() => {
  if (workspaceState.count > 0 && document.visibilityState === 'visible') {
    refreshWorkspaceCharts();
    hydrateWorkspaceEligibility();
  }
}, 15000);

let researchLoaded = false;
const esc = value => String(value ?? '—').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
function drawResearchLines(records) {
  const canvas = $('researchEquity'); if (!canvas) return;
  const ctx = canvas.getContext('2d'), dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = Math.max(1, w*dpr); canvas.height = Math.max(1, h*dpr); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
  const ordered = [...records].sort((a,b) => String(a.timestamp).localeCompare(String(b.timestamp))); let equity = 0; const values = ordered.map(r => equity += Number(r.net_pnl || 0));
  if (!values.length) { ctx.fillStyle='#8290a5'; ctx.fillText('No matching trades',12,28); return; }
  const lo=Math.min(0,...values), hi=Math.max(0,...values), range=hi-lo||1, px=24, py=18;
  const x=i=>px+(w-px*2)*(i/Math.max(1,values.length-1)), y=v=>h-py-(v-lo)/range*(h-py*2);
  ctx.strokeStyle='#243342';ctx.beginPath();ctx.moveTo(px,y(0));ctx.lineTo(w-px,y(0));ctx.stroke();ctx.strokeStyle='#6de7f5';ctx.lineWidth=2;ctx.beginPath();values.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));ctx.stroke();ctx.fillStyle='#8290a5';ctx.font='11px system-ui';ctx.fillText(`$${lo.toFixed(2)}`,2,h-py);ctx.fillText(`$${hi.toFixed(2)}`,2,py+4);
}
function drawResearchBuckets(records) {
  const canvas=$('researchConfidenceChart'); if(!canvas)return; const ctx=canvas.getContext('2d'),dpr=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=Math.max(1,w*dpr);canvas.height=Math.max(1,h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  const buckets=Array.from({length:10},(_,i)=>records.filter(r=>Number(r.confidence)>=i/10&&Number(r.confidence)<(i+1)/10)); const px=25, bottom=h-25, maxH=bottom-20, bw=(w-px*2)/10*.7;
  buckets.forEach((b,i)=>{const rate=b.length?b.filter(r=>r.win).length/b.length:0,x=px+i*(w-px*2)/10+(w-px*2)/10*.15,y=bottom-rate*maxH;ctx.fillStyle=b.length?'#6df2ad':'#243342';ctx.fillRect(x,y,bw,bottom-y);ctx.fillStyle='#8290a5';ctx.font='10px system-ui';ctx.fillText(`${i*10}%`,x,bottom+15);if(b.length)ctx.fillText(`${(rate*100).toFixed(0)}%`,x,y-4)});ctx.fillStyle='#8290a5';ctx.font='11px system-ui';ctx.fillText('Win rate',px,14);
}
async function loadResearch() {
  try {
    const first=await api('/api/research/report?limit=1');
    if(!researchLoaded){$('researchAsset').innerHTML='<option value="">All completed markets</option>'+first.markets.filter(m=>m.status==='complete').map(m=>`<option value="${esc(m.asset)}">${esc(m.asset)}</option>`).join('');researchLoaded=true;}
    const params=new URLSearchParams({limit:'5000',sort:$('researchSort').value,descending:'true',min_confidence:String(Number($('researchConfidence').value||0)/100),min_payout:String(Number($('researchPayout').value||0))});if($('researchAsset').value)params.set('asset',$('researchAsset').value);
    const data=await api(`/api/research/report?${params}`), a=data.aggregate||{}; $('researchTrades').textContent=data.matched_trades??0;$('researchPnl').textContent=money(a.net_pnl);$('researchAllocated').textContent=money(a.amount_allocated);$('researchWinRate').textContent=a.win_rate==null?'—':`${(a.win_rate*100).toFixed(1)}%`;$('researchStatus').textContent=data.orders_placed===0?'NO ORDERS':'CHECK';$('researchNote').textContent=`${data.returned_trades} trades shown (of ${data.matched_trades} matched). Amounts are recalculated from filtered trades only. ${data.payout_note} Errors: ${(data.error_markets||[]).join(', ')||'none'}`;$('researchTradesTable').innerHTML=data.trades.length?data.trades.map(t=>`<tr><td>${esc(t.timestamp)}</td><td>${esc(t.asset)}</td><td>${esc(t.regime)}</td><td>${esc(t.direction)}</td><td>${(Number(t.confidence)*100).toFixed(1)}%</td><td>${money(t.original_amount_allocated)}</td><td>${money(t.filtered_trade_amount_before)}</td><td>${money(t.filtered_trade_amount_after)}</td><td>${money(t.gross_pnl)}</td><td>${money(Number(t.fees||0)+Number(t.spread||0)+Number(t.slippage||0))}</td><td>${money(t.net_pnl)}</td><td>${money(t.account_balance_after)}</td><td>${t.payout_percent==null?'not recorded':`${Number(t.payout_percent).toFixed(1)}%`}</td><td>${t.payout_percent==null?'—':money(t.payout_amount)}</td><td>${t.payout_percent==null?'—':money(t.payment_after_payout)}</td></tr>`).join(''):'<tr><td colspan="15" class="empty">No trades match these filters.</td></tr>';drawResearchLines(data.trades);drawResearchBuckets(data.trades);
  } catch(e) { $('researchStatus').textContent='UNAVAILABLE'; $('researchNote').textContent=`Research report unavailable: ${e.message}`; }
}
$('researchApply').onclick=loadResearch;$('researchAsset').onchange=loadResearch;$('researchSort').onchange=loadResearch;loadResearch();
