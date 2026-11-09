'use strict';
// The Python service owns data preparation, fitting, evaluation, and persistence.
state.history=[];
state.waitMessage='Initializing your simulation.';
const reportCache=new Map();
async function api(path,options={}){
  let response;
  try{response=await fetch(path,{...options,headers:{'Content-Type':'application/json',...options.headers}})}
  catch{throw new Error('The local app connection was lost. Keep the Casimir app window open and try again.')}
  let body;try{body=await response.json()}catch{throw new Error('The local app returned an unreadable response. Please restart it.')}
  if(!response.ok)throw new Error(body.error||'The request could not be completed.');
  return body;
}
async function pollSimulation(jobId){
  for(;;){
    const job=await api('/api/jobs/'+encodeURIComponent(jobId));
    if(job.state==='complete'){storage.set('pending-live-job',null);return job.result}
    if(job.state==='failed'){storage.set('pending-live-job',null);throw new Error(job.message)}
    state.waitMessage=job.message;
    if(state.route==='simulation'&&$('#wait-message'))$('#wait-message').textContent=job.message;
    await new Promise(resolve=>setTimeout(resolve,750));
  }
}
simulationAdapter.initialize=async inputs=>{
  const job=await api('/api/initialize',{method:'POST',body:JSON.stringify({inputs})});
  storage.set('pending-live-job',{jobId:job.job_id,inputs});
  return pollSimulation(job.job_id);
};
const format=value=>Number(value).toFixed(3);
const originalSimulation=simulation;
simulation=()=>originalSimulation()
  .replace('DESIGN PREVIEW','SYNTHETIC SIMULATION')
  .replace('Explore cell capacity in a controlled example.','Explore capacity within the documented simulation domain.')
  .replace('Prototype review: the waiting state and result are examples. Entered values do not affect the displayed result yet.','Synthetic simulation only. Predictions describe hypothetical cells, not measured commercial batteries.');
resultMarkup=function(){
  if(state.running)return `<div class="result-top"><h2 id="result-title" tabindex="-1">Your result</h2><span class="badge quiet-badge">IN PROGRESS</span></div><div class="waiting-state"><div class="waiting-ring" aria-hidden="true"><span>${icon('bolt',29)}</span></div><h3>Initializing your simulation</h3><p id="wait-message">${safe(state.waitMessage)}</p><div class="waiting-dots" aria-hidden="true"><i></i><i></i><i></i></div><small>Your result will appear automatically. First initialization can take longer.</small></div>`;
  if(!state.result)return `<div class="result-top"><h2 id="result-title" tabindex="-1">Your result</h2><span class="result-status"><span class="status-dot"></span> Ready when you are</span></div><div class="ready-state"><div class="ready-symbol">${icon('bolt',33)}</div><h3>A little setup.<br>A clearer picture.</h3><p>Initialize your simulation to see a capacity estimate and a brief explanation.</p><span class="ready-footnote">The app handles the preparation for you.</span></div>`;
  const r=state.result;
  return `<div class="result-top"><h2 id="result-title" tabindex="-1">Your result</h2><span class="result-status">${icon('check',14)} Simulation complete</span></div><div class="completed-result"><p class="capacity-label">Predicted initial discharge capacity</p><div class="capacity">${format(r.capacity)}<small>mAh</small></div><span class="badge quiet-badge">SIMULATION-BASED ESTIMATE</span><p class="result-explanation">The model estimates <strong>${format(r.capacity)} mAh</strong> of initial charge for your cell setup. The noise-free simulation reference is <strong>${format(r.reference)} mAh</strong>.</p><button class="primary full" id="view-details">View more results ${icon('arrow',15)}</button><button class="secondary full save-result" id="save-scenario">${icon('save',15)} Save result</button><p class="result-note">Based on synthetic manufacturing data.</p></div>`;
};
function chartPoints(points,residual=false){
  const width=640,height=300,left=64,right=20,top=20,bottom=55;
  const all=points.flatMap(p=>[p.reference_capacity,p.predicted_capacity]);
  const lo=Math.floor((Math.min(...all)-.05)*10)/10,hi=Math.ceil((Math.max(...all)+.05)*10)/10;
  const maxResidual=Math.max(.02,...points.map(p=>Math.abs(p.residual)))*1.15;
  const minY=residual?-maxResidual:lo,maxY=residual?maxResidual:hi;
  const x=v=>left+(v-lo)/(hi-lo)*(width-left-right);
  const y=v=>height-bottom-(v-minY)/(maxY-minY)*(height-top-bottom);
  let svg=`<svg class="evaluation-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${residual?'Reserved-test residuals in mAh against predicted capacity, with a zero-error line.':'Reserved-test reference capacity against predicted capacity in mAh. The diagonal marks perfect agreement.'}">`;
  for(let i=0;i<=4;i++){
    const xv=lo+(hi-lo)*i/4,yv=minY+(maxY-minY)*i/4;
    svg+=`<path d="M${x(xv)} ${top}V${height-bottom}M${left} ${y(yv)}H${width-right}" class="plot-grid"/><text x="${x(xv)}" y="${height-bottom+20}" text-anchor="middle">${xv.toFixed(2)}</text><text x="${left-9}" y="${y(yv)+3}" text-anchor="end">${yv.toFixed(2)}</text>`;
  }
  svg+=residual?`<path d="M${left} ${y(0)}H${width-right}" class="plot-reference"/>`:`<path d="M${x(lo)} ${y(lo)}L${x(hi)} ${y(hi)}" class="plot-reference"/>`;
  svg+=points.map(p=>`<circle cx="${x(residual?p.predicted_capacity:p.reference_capacity)}" cy="${y(residual?p.residual:p.predicted_capacity)}" r="2.5" class="plot-point"/>`).join('');
  return svg+`<text x="${(width+left-right)/2}" y="${height-8}" text-anchor="middle">${residual?'Predicted':'Reference'} capacity · mAh</text><text transform="translate(15 ${height/2}) rotate(-90)" text-anchor="middle">${residual?'Reference − prediction':'Predicted capacity'} · mAh</text></svg>`;
}
function comparisonChart(rows){
  const max=Math.max(...rows.map(r=>r.cv_mae_mean))*1.16;
  return `<svg class="evaluation-chart comparison-chart" viewBox="0 0 640 265" role="img" aria-label="Mean absolute error for five approaches on grouped development folds. Lower is better.">${rows.map((r,i)=>`<text x="5" y="${32+i*43}" class="algorithm-label">${safe(r.algorithm==='Support Vector Regression'?'Support Vector':r.algorithm)}</text><rect x="160" y="${18+i*43}" width="${r.cv_mae_mean/max*390}" height="21" rx="3" class="comparison-bar"/><text x="${170+r.cv_mae_mean/max*390}" y="${32+i*43}">${format(r.cv_mae_mean)}</text>`).join('')}<text x="160" y="253">Grouped development MAE · mAh (lower is better)</text></svg>`;
}
async function getReport(id){if(!reportCache.has(id))reportCache.set(id,await api('/api/reports/'+encodeURIComponent(id)));return reportCache.get(id)}
details=async function(result){
  const modal=$('#details-dialog');
  modal.innerHTML=`<div class="dialog-top"><span class="eyebrow">DETAILED RESULT</span><button class="icon-button" data-close="details-dialog" aria-label="Close detailed results">×</button></div><h2 id="details-title">Your result, explained.</h2><p class="dialog-intro">Loading your result details…</p>`;
  if(!modal.open)modal.showModal();
  try{
    const report=await getReport(result.model_run_id);
    if(!modal.open)return;
    const m=report.test_metrics;
    modal.classList.add('wide-details');
    modal.innerHTML=`<div class="dialog-top"><span class="eyebrow">DETAILED RESULT</span><button class="icon-button" data-close="details-dialog" aria-label="Close detailed results">×</button></div><h2 id="details-title">Your result, explained.</h2><p class="dialog-intro">Your predicted capacity is <strong>${format(result.capacity)} mAh</strong>. The analytical reference for your inputs is <strong>${format(result.reference)} mAh</strong>; the difference is <strong>${format(Math.abs(result.capacity-result.reference))} mAh</strong>.</p>
    <div class="notice"><strong>What this means</strong><p>Capacity is charge delivered during initial discharge. The reference is a noise-free equation value; model evaluation below uses noisy, reserved synthetic records. Neither measure establishes commercial battery performance or backup duration.</p></div>
    <h3>How the model performed</h3><p class="microcopy">${report.split_manifest.final_test_records} records from ${report.split_manifest.final_test_batches.length} reserved batches. These records were excluded from model selection.</p><div class="evaluation-metrics">${[['Average error',format(m.mae)+' mAh','MAE · smaller is better'],['Larger-error measure',format(m.rmse)+' mAh','RMSE · smaller is better'],['Explained variation',m.r2===null?'Undefined':m.r2.toFixed(4),'R² · not an accuracy percentage'],['Percentage error',m.mape===null?'Undefined':m.mape.toFixed(2)+'%','MAPE · smaller is better']].map(([t,v,n])=>`<article><span>${t}</span><strong>${v}</strong><small>${n}</small></article>`).join('')}</div>
    <section class="evaluation-section"><h3>Reference vs predicted capacity</h3><p>Points closer to the diagonal indicate closer agreement on the reserved test records.</p>${chartPoints(report.test_points)}</section>
    <section class="evaluation-section"><h3>Where prediction errors occur</h3><p>Residuals near zero mean smaller errors. Patterns can reveal a systematic miss.</p>${chartPoints(report.test_points,true)}</section>
    <section class="evaluation-section"><h3>Model comparison</h3><p>Development cross-validation only. ${safe(report.selected_algorithm)} was selected before the final test.</p>${comparisonChart(report.development_comparison)}</section>
    <details class="technical-report"><summary>Scenario and evaluation details</summary><div class="detail-values">${fields.map(f=>`<div><span>${f.name}</span><strong>${Number(result.inputs[f.key]).toFixed(f.decimals)} ${f.unit}</strong></div>`).join('')}<div><span>Prediction model</span><strong>${safe(result.model)}</strong></div><div><span>Simulation version</span><strong>${safe(result.simulation_version)}</strong></div><div><span>Model version</span><strong>${safe(result.model_run_id)}</strong></div><div><span>Recorded</span><strong>${safe(new Date(result.createdAt).toLocaleString())}</strong></div></div><h3>Results by reserved batch</h3><div class="table-wrap"><table><thead><tr><th>Batch</th><th>MAE · mAh</th><th>RMSE · mAh</th></tr></thead><tbody>${report.batch_metrics.map(b=>`<tr><td>${safe(b.batch)}</td><td>${format(b.mae)}</td><td>${format(b.rmse)}</td></tr>`).join('')}</tbody></table></div><p class="microcopy">The full download includes grouped fold variation, development feature importance, noise sensitivity, partition membership, and artifact checksums.</p></details>
    <button class="secondary full" id="export-result">${icon('download',16)} Download full result and evaluation</button>`;
    $('#export-result').onclick=()=>download('casimir-simulation-result.json',{prediction:result,evaluation:report});
  }catch(error){modal.innerHTML=`<div class="dialog-top"><span class="eyebrow">RESULT DETAILS</span><button class="icon-button" data-close="details-dialog" aria-label="Close detailed results">×</button></div><h2 id="details-title">Details unavailable</h2><p class="dialog-intro">${safe(error.message)}</p>`}
};
history=function(){return `${heading('Your saved results.','Revisit the scenarios you’ve kept for later.').replace('DESIGN PREVIEW','SYNTHETIC SIMULATION')}<section class="panel"><div class="table-heading"><div><h2>Saved scenarios</h2><p>Simulation results stored on this computer.</p></div>${state.history.length?'<button class="secondary small" id="export-history">Download all</button>':''}</div>${state.history.length?`<div class="table-wrap"><table><thead><tr><th scope="col">Scenario</th><th scope="col">Saved</th><th scope="col">Capacity</th><th scope="col">Result</th></tr></thead><tbody>${state.history.map((r,i)=>`<tr><td>Scenario ${String(i+1).padStart(2,'0')}<small class="table-example">Synthetic simulation</small></td><td>${safe(new Date(r.createdAt).toLocaleString())}</td><td class="numeric">${format(r.capacity)} mAh</td><td><button class="text-button" data-history="${i}">View result ↗</button></td></tr>`).join('')}</tbody></table></div>`:`<div class="empty">${icon('history',35)}<h3>Your saved results will live here.</h3><p>Initialize a simulation, then select Save result to keep it.</p><a class="primary" href="#simulation">Start a simulation ${icon('arrow',15)}</a></div>`}</section><p class="prototype-note">Synthetic simulation · Results are retained in the local app database.</p>`};
const prototypeBind=bind;
bind=function(){
  prototypeBind();
  if(state.route==='history'){
    if($('#export-history'))$('#export-history').onclick=()=>download('casimir-saved-results.json',{label:'Synthetic simulation results',units:'mAh',results:state.history});
    return;
  }
  if($('#save-scenario'))$('#save-scenario').onclick=async()=>{
    const button=$('#save-scenario');button.disabled=true;
    try{await api('/api/save',{method:'POST',body:JSON.stringify({prediction_id:state.result.prediction_id})});state.history=(await api('/api/history')).results;toast('Result saved. Find it in Saved results.')}
    catch(error){toast(error.message)}finally{if(document.contains(button))button.disabled=false}
  };
};
initialize=async function(event){
  event.preventDefault();if(state.running)return;
  state.error='';
  const invalid=fields.find(f=>state.inputs[f.key]===''||!Number.isFinite(Number(state.inputs[f.key]))||state.inputs[f.key]<f.min||state.inputs[f.key]>f.max||!$(`#input-${f.key}`).validity.valid);
  if(invalid){state.customOpen=true;state.error=`Enter ${invalid.name.toLowerCase()} between ${invalid.min} and ${invalid.max}.`;render();$(`#input-${invalid.key}`).focus();return}
  state.running=true;state.result=null;state.waitMessage='Initializing your simulation.';render();
  try{state.result=await simulationAdapter.initialize({...state.inputs})}
  catch(error){state.error=error.message}
  finally{state.running=false;render();if(state.route==='simulation')$('#result-title').focus({preventScroll:true})}
};
download=function(name,value){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Your download is ready.')};
async function loadHistory(){try{state.history=(await api('/api/history')).results;if(state.route==='history')render()}catch(error){toast(error.message)}}
async function resume(){const pending=storage.get('pending-live-job',null);if(!pending)return;state.inputs=pending.inputs;state.running=true;render();try{state.result=await pollSimulation(pending.jobId)}catch(error){storage.set('pending-live-job',null);state.error=error.message}finally{state.running=false;render()}}
render();loadHistory();resume();
