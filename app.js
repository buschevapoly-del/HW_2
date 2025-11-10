/* =======================================================================
   Titanic Binary Classifier — TensorFlow.js (FINAL)
   - Forces Kaggle CSV settings (',' and '"')
   - Repairs mis-parsed rows (split Name -> joined, shifted fields fixed)
   - Fully client-side: PapaParse + TensorFlow.js
   ======================================================================= */

/* ------------------------------ State --------------------------------- */
const state = {
  rawTrain: [], rawTest: [],
  pre: null,
  xsTrain: null, ysTrain: null,
  xsVal: null,   ysVal: null,
  model: null,
  valProbs: null, testProbs: null,
  thresh: 0.5
};

const SCHEMA = {
  target: 'Survived',
  id: 'PassengerId',
  cols: ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'] // features
};

const el = id => document.getElementById(id);

/* --------------------- Backend: prefer CPU (stable) ------------------- */
(async () => {
  try { await tf.setBackend('cpu'); } catch {}
  await tf.ready();
})();

/* --------------------------- CSV helpers ------------------------------ */
// Force Kaggle defaults: delimiter ',', quote '"'
function parseWithPapa(file, delimiter, quoteChar){
  return new Promise((resolve,reject)=>{
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: 'greedy',
      delimiter,
      quoteChar,
      complete: r => resolve(r.data),
      error: reject
    });
  });
}

const normalizeRow = row => {
  const o = {};
  for (const [k, v] of Object.entries(row)) {
    if (v === '') o[k] = null;
    else if (typeof v === 'string') { const t = v.trim(); o[k] = t === '' ? null : t; }
    else o[k] = v;
  }
  return o;
};

function roughMissingPct(rows){
  if(!rows.length) return 100;
  const cols = Object.keys(rows[0]);
  let miss = 0, total = rows.length * cols.length;
  for(const r of rows){ for(const c of cols){ const v=r[c]; if(v===''||v==null||v===undefined) miss++; } }
  return +(100*miss/total).toFixed(1);
}

function previewTable(rows, limit=8){
  if(!rows.length){ el('previewTable').innerHTML = ''; return; }
  const cols = Object.keys(rows[0]);
  const head = '<thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead>';
  const body = '<tbody>'+rows.slice(0,limit).map(r =>
    '<tr>'+cols.map(c=>`<td>${r[c]??''}</td>`).join('')+'</tr>'
  ).join('')+'</tbody>';
  el('previewTable').innerHTML = `<table>${head}${body}</table>`;
}

/* -------------------- CSV REPAIR HELPERS (FINAL) ---------------------- */
// Detect & fix the “split name / shifted columns” problem seen in your screenshot.

function isGoodSex(v){ return typeof v === 'string' && /^(male|female)$/i.test(v.trim()); }
function numOrNull(v){ const x = Number(v); return Number.isFinite(x) ? x : null; }
function stripQuotes(s){
  if (typeof s !== 'string') return s;
  return s.replace(/^\s*"+/, '').replace(/"+\s*$/, '');
}

// Row is "shifted" when Sex isn't male/female but Age looks like 'male'/'female'
function looksShifted(row){
  const sexBad = !isGoodSex(row.Sex);
  const ageLooksSex = typeof row.Age === 'string' && /^(male|female)$/i.test(row.Age.trim());
  return sexBad && ageLooksSex;
}

// Repair a single shifted row by joining Name pieces and shifting fields right
function repairShiftedRow(row){
  const r = { ...row };

  // 1) Join Name (left) + Sex (right piece) into a single Name
  const left  = stripQuotes(r.Name ?? '');
  const right = stripQuotes((r.Sex ?? '').toString());
  const joinedName = (left ? left : '') + (right ? (left ? ', ' : '') + right : '');
  r.Name = stripQuotes(joinedName);

  // 2) Shift remaining fields back to their rightful place
  r.Sex    = (r.Age ?? '').toString().trim();     // 'male'/'female'
  r.Age    = numOrNull(r.SibSp);
  r.SibSp  = numOrNull(r.Parch);
  r.Parch  = numOrNull(r.Ticket);
  r.Ticket = (r.Fare ?? '').toString();
  r.Fare   = numOrNull(r.Cabin);

  // 3) Embarked may be shoved into __parsed_extra – recover the last token
  if (Array.isArray(r.__parsed_extra) && r.__parsed_extra.length){
    r.Embarked = r.__parsed_extra[r.__parsed_extra.length - 1];
  }
  delete r.__parsed_extra;

  return r;
}

// Clean a batch of rows: repair if needed; strip __parsed_extra otherwise
function repairTitanicRows(rows){
  return rows.map(row => {
    if (looksShifted(row)) {
      try { return repairShiftedRow(row); }
      catch { /* on repair error, fall through to basic cleanup */ }
    }
    const r = { ...row };
    delete r.__parsed_extra;
    return r;
  });
}

/* --------------------------- Preprocessing ---------------------------- */
const median = a => { const b=a.filter(v=>v!=null&&!Number.isNaN(+v)).map(Number).sort((x,y)=>x-y); if(!b.length) return null; const m=Math.floor(b.length/2); return b.length%2?b[m]:(b[m-1]+b[m])/2; };
const mode   = a => { const m=new Map(); let best=null,cnt=0; for(const v of a){ if(v==null||v==='') continue; const c=(m.get(v)||0)+1; m.set(v,c); if(c>cnt){cnt=c;best=v;} } return best; };
const oneHot = (v,cats)=>{ const r=new Array(cats.length).fill(0); const i=cats.indexOf(v); if(i>=0) r[i]=1; return r; };
const mean   = a => { const b=a.filter(Number.isFinite); return b.length? b.reduce((s,x)=>s+x,0)/b.length : 0; };
const sd     = a => { const b=a.filter(Number.isFinite); if(b.length<2) return 0; const m=mean(b); return Math.sqrt(b.reduce((s,x)=>s+(x-m)**2,0)/(b.length-1)); };
const finite = (x,def=0)=>Number.isFinite(x)?x:def;

function buildPreprocessor(train){
  const ageMed = Number.isFinite(median(train.map(r=>r.Age))) ? median(train.map(r=>r.Age)) : 30;
  const embMode = mode(train.map(r=>r.Embarked)) ?? 'S';
  const sexCats=['female','male'], pclassCats=[1,2,3], embCats=['C','Q','S','UNKNOWN'];

  const ageVals=train.map(r=>finite((r.Age!=null&&!Number.isNaN(+r.Age))?+r.Age:ageMed,ageMed));
  const fareVals=train.map(r=>finite((r.Fare!=null&&!Number.isNaN(+r.Fare))?+r.Fare:0,0));
  const muA=mean(ageVals), sdA=sd(ageVals), muF=mean(fareVals), sdF=sd(fareVals);

  const useFamily = el('featFamily').checked;
  const useAlone  = el('featAlone').checked;

  const base = r => {
    const age=(r.Age!=null&&!Number.isNaN(+r.Age))?+r.Age:ageMed;
    const emb=(r.Embarked==null||r.Embarked==='')?'UNKNOWN':r.Embarked;
    const fare=(r.Fare!=null&&!Number.isNaN(+r.Fare))?+r.Fare:0;
    const fam=(+r.SibSp||0)+(+r.Parch||0)+1, alone=(fam===1)?1:0;
    const ageZ=sdA? (age-muA)/sdA : 0, fareZ=sdF? (fare-muF)/sdF : 0;
    let f=[ageZ,fareZ,...oneHot(r.Sex,sexCats),...oneHot(+r.Pclass,pclassCats),...oneHot(emb,embCats)];
    if(useFamily) f.push(fam);
    if(useAlone)  f.push(alone);
    return f.map(x=>finite(+x,0));
  };

  const FEAT = base(train[0]||{}).length;

  return {
    ageMed, embMode, sexCats, pclassCats, embCats, muA, sdA, muF, sdF,
    useFamily, useAlone, featLen:FEAT,
    mapRow: r => {
      const v = base(r);
      if(v.length!==FEAT){ if(v.length<FEAT) v.push(...Array(FEAT-v.length).fill(0)); else v.length=FEAT; }
      return v;
    }
  };
}

function tensorize(rows, pre){
  const X=[], Y=[];
  for(const r of rows){
    const f = pre.mapRow(r);
    if(f.every(Number.isFinite)){ X.push(f); if('Survived' in r) Y.push(+r.Survived); }
  }
  if(!X.length) throw new Error('No valid rows after preprocessing.');
  const xs = tf.tensor2d(X,[X.length, pre.featLen],'float32');
  let ys = null; if(Y.length) ys = tf.tensor2d(Y,[Y.length,1],'float32');
  return { xs, ys, nFeat: pre.featLen };
}

function stratifiedSplit(rows, r=0.2){
  const z=rows.filter(x=>+x.Survived===0), o=rows.filter(x=>+x.Survived===1);
  const split=g=>{const a=g.slice(); tf.util.shuffle(a); const n=Math.max(1,Math.floor(a.length*r)); return {val:a.slice(0,n), tr:a.slice(n)};};
  const a=split(z), b=split(o);
  const train=a.tr.concat(b.tr), val=a.val.concat(b.val);
  tf.util.shuffle(train); tf.util.shuffle(val);
  return { train, val };
}

/* ------------------------------- Model -------------------------------- */
function buildModel(inputDim){
  const m=tf.sequential();
  m.add(tf.layers.dense({units:16,activation:'relu',inputShape:[inputDim]}));
  m.add(tf.layers.dense({units:1,activation:'sigmoid'}));
  m.compile({optimizer:'adam',loss:'binaryCrossentropy',metrics:['accuracy']});
  return m;
}
const modelSummaryText = m => { const lines=[]; m.summary(undefined, undefined, s=>lines.push(s)); return lines.join('\n'); };

/* ---------------------------- Metrics/plots --------------------------- */
function rocPoints(yTrue, yProb, steps=200){
  const T=[]; for(let i=0;i<=steps;i++) T.push(i/steps);
  const pts=T.map(th=>{
    let TP=0,FP=0,TN=0,FN=0;
    for(let i=0;i<yTrue.length;i++){
      const y=yTrue[i], p=yProb[i]>=th?1:0;
      if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++;
    }
    const TPR=TP/(TP+FN||1), FPR=FP/(FP+TN||1);
    return {x:FPR,y:TPR,th};
  });
  const s=pts.slice().sort((a,b)=>a.x-b.x);
  let auc=0; for(let i=1;i<s.length;i++){ const a=s[i-1], b=s[i]; auc += (b.x-a.x)*(a.y+b.y)/2; }
  return { points:s, auc };
}
function drawROC(canvas, pts){
  const ctx=canvas.getContext('2d'); const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#0f1628'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#233350'; ctx.lineWidth=1;
  for(let i=0;i<=5;i++){const x=i/5; ctx.beginPath(); ctx.moveTo(40+x*(W-60), H-30); ctx.lineTo(40+x*(W-60), 20); ctx.stroke();}
  for(let i=0;i<=5;i++){const y=i/5; ctx.beginPath(); ctx.moveTo(40, 20+y*(H-50)); ctx.lineTo(W-20, 20+y*(H-50)); ctx.stroke();}
  ctx.strokeStyle='#8aa3ff'; ctx.lineWidth=2; ctx.beginPath();
  pts.forEach((p,i)=>{ const x=40+p.x*(W-60), y=H-30-p.y*(H-50); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
  ctx.stroke();
}

/* ---------------- Early stop with best-weight restore + yields -------- */
let stopFlag=false;
function earlyStopWithRestore(patience=5, monitor='val_loss'){
  let best=Infinity, wait=0, snapshot=null;
  return new tf.CustomCallback({
    onBatchEnd: async () => { await new Promise(r=>setTimeout(r,0)); },
    onEpochEnd: async (_epoch, logs) => {
      await tf.nextFrame();
      const cur=logs?.[monitor];
      if(cur!=null){
        if(cur<best-1e-12){
          best=cur; wait=0;
          if(snapshot) snapshot.forEach(t=>t.dispose());
          snapshot = state.model.getWeights().map(w=>w.clone());
        } else if(++wait>=patience){
          if(snapshot){ state.model.setWeights(snapshot); snapshot=null; }
          state.model.stopTraining = true;
        }
      }
      if(stopFlag) state.model.stopTraining = true;
    }
  });
}

/* ------------------------------ Handlers ------------------------------ */
// FINAL: load + normalize + REPAIR rows
async function onLoadFiles(){
  try{
    const fT = el('trainFile')?.files?.[0];
    const fX = el('testFile')?.files?.[0];
    if(!fT){ alert('Please choose train.csv'); return; }

    // Force Kaggle defaults (comma + double-quote)
    const trainRows = await parseWithPapa(fT, ',', '"');
    const testRows  = fX ? await parseWithPapa(fX, ',', '"') : [];

    // Normalize, then repair any mis-parsed rows (split Name / shifts)
    state.rawTrain = repairTitanicRows(trainRows.map(normalizeRow));
    state.rawTest  = repairTitanicRows(testRows.map(normalizeRow));

    // Update UI
    el('kTrain').textContent = state.rawTrain.length;
    el('kTest').textContent  = state.rawTest.length || '—';
    el('kMiss').textContent  = roughMissingPct(state.rawTrain) + '%';
    previewTable(state.rawTrain);
  }catch(e){
    console.error(e);
    alert('Failed to load CSV: ' + (e?.message || e));
  }
}

function onPreprocess(){
  try{
    if(!state.rawTrain.length){ alert('Load train.csv first'); return; }
    state.pre = buildPreprocessor(state.rawTrain);
    const {train, val} = stratifiedSplit(state.rawTrain, 0.2);
    const tT = tensorize(train, state.pre);
    const tV = tensorize(val, state.pre);
    state.xsTrain=tT.xs; state.ysTrain=tT.ys; state.xsVal=tV.xs; state.ysVal=tV.ys;

    el('preInfo').textContent = [
      `Features: ${tT.nFeat}`,
      `Train: ${state.xsTrain.shape} | Val: ${state.xsVal.shape}`,
      `Impute Age median=${state.pre.ageMed} | Embarked mode=${state.pre.embMode}`,
      `One-hot: Sex, Pclass, Embarked | Engineered: FamilySize=${state.pre.useFamily}, IsAlone=${state.pre.useAlone}`
    ].join('\n');
  }catch(e){
    console.error(e);
    alert('Preprocessing failed: ' + (e?.message || e));
  }
}

function onBuild(){
  try{
    if(!state.xsTrain){ alert('Run Preprocessing first'); return; }
    state.model = buildModel(state.xsTrain.shape[1]);
    el('modelSummary').textContent = 'Model built. Click "Show Summary" to view layers.';
  }catch(e){
    console.error(e);
    alert('Build failed: ' + (e?.message || e));
  }
}

function onSummary(){
  try{
    if(!state.model){ alert('Build the model first'); return; }
    el('modelSummary').textContent = modelSummaryText(state.model);
  }catch(e){
    console.error(e);
    alert('Summary failed: ' + (e?.message || e));
  }
}

async function onTrain(){
  try{
    if(!state.model){ alert('Build the model first'); return; }
    stopFlag=false;
    el('trainLog').textContent='';

    const cb = earlyStopWithRestore(5,'val_loss');
    await state.model.fit(state.xsTrain, state.ysTrain, {
      epochs: 40, batchSize: 16,
      validationData: [state.xsVal, state.ysVal],
      callbacks: [{
        onEpochEnd: async (ep, logs)=>{
          el('trainLog').textContent +=
            `epoch ${ep+1}: loss=${logs.loss.toFixed(4)} val_loss=${logs.val_loss.toFixed(4)} acc=${(logs.acc??logs.accuracy??0).toFixed(4)}\n`;
          await cb.onEpochEnd?.(ep, logs);
        },
        onBatchEnd: async (b, logs)=>{ await cb.onBatchEnd?.(b, logs); }
      }]
    });

    const valPred = tf.tidy(()=> state.model.predict(state.xsVal).dataSync());
    state.valProbs = Float32Array.from(valPred);

    const yTrue = Array.from(state.ysVal.dataSync()).map(v=>+v);
    const {points, auc} = rocPoints(yTrue, state.valProbs, 200);
    drawROC(el('rocCanvas'), points);
    el('aucText').textContent = `AUC = ${auc.toFixed(4)}`;
    updateThreshold(state.thresh);
  }catch(e){
    console.error(e);
    alert('Training failed: ' + (e?.message || e));
  }
}
function onStop(){ stopFlag=true; alert('Early stop requested (will stop after this epoch).'); }

function confusionStats(yTrue, yProb, th){
  let TP=0,FP=0,TN=0,FN=0;
  for(let i=0;i<yTrue.length;i++){
    const y=yTrue[i], p=yProb[i]>=th?1:0;
    if(y===1&&p===1)TP++; else if(y===0&&p===1)FP++; else if(y===0&&p===0)TN++; else FN++;
  }
  const prec=TP/(TP+FP||1), rec=TP/(TP+FN||1), f1=(2*prec*rec)/((prec+rec)||1);
  return {TP,FP,TN,FN,prec,rec,f1};
}
function updateThreshold(th){
  el('thVal').textContent=(+th).toFixed(2);
  if(state.valProbs==null) return;
  const yTrue=Array.from(state.ysVal.dataSync()).map(v=>+v);
  const st=confusionStats(yTrue,state.valProbs,+th);
  el('cmTP').textContent=st.TP; el('cmFN').textContent=st.FN;
  el('cmFP').textContent=st.FP; el('cmTN').textContent=st.TN;
  el('prf').textContent=`Precision: ${(st.prec*100).toFixed(2)}%\nRecall: ${(st.rec*100).toFixed(2)}%\nF1: ${st.f1.toFixed(4)}`;
  state.thresh=+th;
}

async function onPredict(){
  try{
    if(!state.model){ alert('Train the model first'); return; }
    if(!state.rawTest.length){ alert('Load test.csv'); return; }

    const probs = tf.tidy(()=>{
      const X = state.rawTest.map(state.pre.mapRow);
      const xs = tf.tensor2d(X,[X.length, state.pre.featLen],'float32');
      const out = state.model.predict(xs).dataSync();
      xs.dispose(); return out;
    });
    state.testProbs = Float32Array.from(probs);
    el('predInfo').textContent = `Predicted ${state.rawTest.length} rows. You can now download CSVs.`;
  }catch(e){
    console.error(e);
    alert('Prediction failed: ' + (e?.message || e));
  }
}

function downloadCSV(name, rows){
  if(!rows.length) return;
  const cols = Object.keys(rows[0]);
  const esc=v=>{ if(v==null) return ''; const s=String(v); return /[",\n]/.test(s)? '"'+s.replace(/"/g,'""')+'"' : s; };
  const csv = [cols.join(',')].concat(rows.map(r=>cols.map(c=>esc(r[c])).join(','))).join('\n');
  const blob=new Blob([csv],{type:'text/csv;charset=utf-8;'});
  const url=URL.createObjectURL(blob); const a=document.createElement('a');
  a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url);
}
function onDownloadSubmission(){
  try{
    if(state.testProbs==null){ alert('Run Predict first'); return; }
    const out = state.rawTest.map((r,i)=>({PassengerId:r[SCHEMA.id], Survived:(state.testProbs[i]>=state.thresh?1:0)}));
    downloadCSV('submission.csv', out);
  }catch(e){ console.error(e); alert('Download failed: '+(e?.message||e)); }
}
function onDownloadProbs(){
  try{
    if(state.testProbs==null){ alert('Run Predict first'); return; }
    const out = state.rawTest.map((r,i)=>({PassengerId:r[SCHEMA.id], ProbSurvived:state.testProbs[i]}));
    downloadCSV('probabilities.csv', out);
  }catch(e){ console.error(e); alert('Download failed: '+(e?.message||e)); }
}
async function onSaveModel(){
  try{
    if(!state.model){ alert('Train the model first'); return; }
    await state.model.save('downloads://titanic-tfjs');
  }catch(e){ console.error(e); alert('Save failed: '+(e?.message||e)); }
}

/* ----------------------------- Wire up UI ----------------------------- */
window.addEventListener('DOMContentLoaded', ()=>{
  el('btnLoad').addEventListener('click', onLoadFiles);
  el('btnPre').addEventListener('click', onPreprocess);
  el('btnBuild').addEventListener('click', onBuild);
  el('btnSummary').addEventListener('click', onSummary);
  el('btnTrain').addEventListener('click', onTrain);
  el('btnStop').addEventListener('click', onStop);
  el('thSlider').addEventListener('input', e=>updateThreshold(+e.target.value));
  el('btnPredict').addEventListener('click', onPredict);
  el('btnSub').addEventListener('click', onDownloadSubmission);
  el('btnProb').addEventListener('click', onDownloadProbs);
  el('btnSaveModel').addEventListener('click', onSaveModel);
});
