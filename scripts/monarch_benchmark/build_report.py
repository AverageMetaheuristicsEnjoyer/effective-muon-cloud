#!/usr/bin/env python3
"""Build a self-contained visual report from the large-model benchmark sweep."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.monarch_benchmark.common import (
    COMMON_CONTROL_FIELDS,
    MODEL_SPECS,
    VARIANTS,
    result_is_complete,
)


def validate_payload(payload: dict) -> list[dict]:
    results = payload.get("results", [])
    expected = {(model["name"], variant["name"]) for model in MODEL_SPECS for variant in VARIANTS}
    if len(results) != len(expected):
        raise ValueError(f"expected {len(expected)} results, got {len(results)}")
    incomplete = [index for index, result in enumerate(results) if not result_is_complete(result)]
    if incomplete:
        raise ValueError(f"incomplete benchmark results at indices {incomplete}")
    keys = [
        (result.get("model", {}).get("name"), result.get("variant", {}).get("name"))
        for result in results
    ]
    actual = set(keys)
    if len(actual) != len(keys):
        raise ValueError("benchmark matrix contains duplicate model/variant entries")
    if actual != expected:
        raise ValueError(f"benchmark matrix mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    uuids = {result["gpu"]["uuid"] for result in results}
    if len(uuids) != 1:
        raise ValueError(f"benchmark used multiple GPUs: {sorted(uuids)}")
    controls = {
        json.dumps(
            {field: result["benchmark"][field] for field in COMMON_CONTROL_FIELDS},
            sort_keys=True,
        )
        for result in results
    }
    if len(controls) != 1:
        raise ValueError(f"benchmark controls differ across trials: {controls}")
    contaminated = [
        (result["model"]["name"], result["variant"]["name"])
        for result in results
        if result["gpu"].get("foreign_processes_before")
        or result["gpu"].get("foreign_processes_after")
        or result["gpu"].get("contamination_monitor", {}).get("foreign_processes_seen")
        or result["gpu"].get("contamination_monitor", {}).get("error")
    ]
    if contaminated:
        raise ValueError(f"contaminated benchmark results: {contaminated}")
    return sorted(
        results,
        key=lambda result: (
            result["model"]["dense_equivalent_parameters"],
            [variant["name"] for variant in VARIANTS].index(result["variant"]["name"]),
        ),
    )


def comparison(results: list[dict]) -> dict:
    by_key = {(result["model"]["name"], result["variant"]["name"]): result for result in results}
    rows = []
    for model in MODEL_SPECS:
        monarch = by_key[(model["name"], "monarch_muon")]
        adamw = by_key[(model["name"], "dense_adamw")]
        muon = by_key[(model["name"], "dense_muon")]
        monarch_ms = monarch["summary"]["host_total_ms"]["median"]
        rows.append(
            {
                "model": model["name"],
                "label": model["label"],
                "dense_equivalent_parameters": model["dense_params_expected"],
                "monarch_parameters": model["monarch_params_expected"],
                "speedup_vs_adamw": adamw["summary"]["host_total_ms"]["median"] / monarch_ms,
                "speedup_vs_muon": muon["summary"]["host_total_ms"]["median"] / monarch_ms,
                "parameter_fraction": model["monarch_params_expected"] / model["dense_params_expected"],
            }
        )
    return {"rows": rows, "largest": rows[-1]}


def render(payload: dict) -> str:
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Large-model Monarch-Muon benchmark</title><style>
:root{color-scheme:dark;--bg:#080d18;--panel:#111a2b;--grid:#293650;--text:#edf3ff;--muted:#9eabc5;--m:#6ee7c7;--a:#ffad66;--u:#8fa8ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#173253 0,var(--bg) 42%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui}
main{width:min(1220px,94vw);margin:auto;padding:44px 0 72px}h1{font-size:clamp(2.2rem,5vw,4.2rem);letter-spacing:-.05em;margin:.15em 0}h2{margin:36px 0 14px}.eyebrow{color:var(--m);font-weight:800;letter-spacing:.14em;text-transform:uppercase;font-size:.75rem}p{color:var(--muted);line-height:1.65}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,.chart,.note{background:#111a2bea;border:1px solid #283651;border-radius:16px;box-shadow:0 16px 48px #0004}.card{padding:18px}.card small{color:var(--muted)}.value{font-size:1.55rem;font-weight:780;margin-top:8px}.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart{padding:16px}.chart h3{margin:0 0 10px}canvas{width:100%;height:290px}.legend{display:flex;gap:18px;color:var(--muted);font-size:.82rem;margin:10px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}table{width:100%;border-collapse:collapse;background:var(--panel);border-radius:14px;overflow:hidden}th,td{padding:11px 13px;border-bottom:1px solid var(--grid);text-align:right}th:first-child,td:first-child{text-align:left}.note{padding:18px 22px}code{color:#c7d8ff}footer{margin-top:30px;color:var(--muted);font-size:.8rem}@media(max-width:850px){.cards{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}}
</style></head><body><main><div class="eyebrow">End-to-end training-step benchmark</div><h1>Monarch-Muon<br>200M → 6.9B scaling</h1><p id="subtitle"></p><div class="cards" id="cards"></div>
<h2>Scaling behavior</h2><div class="legend"><span><i class="dot" style="background:var(--m)"></i>Monarch-Muon</span><span><i class="dot" style="background:var(--a)"></i>Dense AdamW</span><span><i class="dot" style="background:var(--u)"></i>Dense Muon</span></div><div class="charts"><section class="chart"><h3>Median optimizer-step latency</h3><canvas id="latency"></canvas></section><section class="chart"><h3>Token throughput</h3><canvas id="throughput"></canvas></section><section class="chart"><h3>Monarch speedup progression</h3><canvas id="speedup"></canvas></section><section class="chart"><h3>Peak allocated memory</h3><canvas id="memory"></canvas></section></div>
<h2>Largest model phase breakdown</h2><div class="chart"><canvas id="phases"></canvas></div><h2>Results</h2><div id="table"></div><h2>Conclusions</h2><div class="note" id="conclusions"></div><h2>Method and fairness controls</h2><div class="note" id="method"></div><footer id="footer"></footer>
<script>const DATA=__DATA__;const colors={monarch_muon:'#6ee7c7',dense_adamw:'#ffad66',dense_muon:'#8fa8ff'};const variants=['monarch_muon','dense_adamw','dense_muon'];const labels={monarch_muon:'Monarch-Muon',dense_adamw:'Dense AdamW',dense_muon:'Dense Muon'};const fmt=(x,d=2)=>Number(x).toFixed(d);const byKey=new Map(DATA.results.map(r=>[r.model.name+'|'+r.variant.name,r]));const largest=DATA.comparison.largest;const largestRuns=variants.map(v=>byKey.get(largest.model+'|'+v));const controls=DATA.results[0].benchmark;document.getElementById('subtitle').textContent='Same physical '+DATA.results[0].gpu.name+', '+fmt(controls.tokens_per_step/1000,3)+'K tokens per optimizer step, BF16 storage, and '+controls.measured_steps+' measured steps after '+controls.warmup_steps+' warmups. Generated '+DATA.generated_at+'.';document.getElementById('cards').innerHTML=[['6.9B vs AdamW',fmt(largest.speedup_vs_adamw)+'×'],['6.9B vs dense Muon',fmt(largest.speedup_vs_muon)+'×'],['6.9B Monarch params',fmt(largest.monarch_parameters/1e9,2)+'B'],['Measured GPU',DATA.results[0].gpu.name]].map(x=>'<div class="card"><small>'+x[0]+'</small><div class="value">'+x[1]+'</div></div>').join('');
function lineChart(id,series,ylabel){const c=document.getElementById(id),q=c.getContext('2d'),r=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*r;c.height=h*r;q.scale(r,r);const pts=series.flatMap(s=>s.data),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=0,ymax=Math.max(...ys)*1.08,p={l:58,r:16,t:14,b:40};const sx=x=>p.l+(Math.log10(x)-Math.log10(xmin))/(Math.log10(xmax)-Math.log10(xmin))*(w-p.l-p.r),sy=y=>h-p.b-(y-ymin)/(ymax-ymin||1)*(h-p.t-p.b);q.font='11px system-ui';q.strokeStyle='#293650';q.fillStyle='#9eabc5';for(let i=0;i<5;i++){let y=ymax*i/4,py=sy(y);q.beginPath();q.moveTo(p.l,py);q.lineTo(w-p.r,py);q.stroke();q.fillText(fmt(y,ymax<10?1:0),5,py+4)}DATA.comparison.rows.forEach(row=>{q.fillText(row.label,sx(row.dense_equivalent_parameters)-16,h-13)});q.fillText(ylabel,7,13);series.forEach(s=>{q.strokeStyle=s.color;q.fillStyle=s.color;q.lineWidth=2.2;q.beginPath();s.data.forEach((pnt,i)=>{i?q.lineTo(sx(pnt[0]),sy(pnt[1])):q.moveTo(sx(pnt[0]),sy(pnt[1]))});q.stroke();s.data.forEach(pnt=>{q.beginPath();q.arc(sx(pnt[0]),sy(pnt[1]),3,0,Math.PI*2);q.fill()})})}
function series(metric,scale=1){return variants.map(v=>({color:colors[v],data:DATA.comparison.rows.map(row=>[row.dense_equivalent_parameters,byKey.get(row.model+'|'+v).summary[metric].median*scale])}))}function draw(){lineChart('latency',series('host_total_ms'),'milliseconds');lineChart('throughput',series('tokens_per_second',1/1000),'K tokens/s');lineChart('speedup',[{color:colors.dense_adamw,data:DATA.comparison.rows.map(r=>[r.dense_equivalent_parameters,r.speedup_vs_adamw])},{color:colors.dense_muon,data:DATA.comparison.rows.map(r=>[r.dense_equivalent_parameters,r.speedup_vs_muon])}],'speedup ×');lineChart('memory',variants.map(v=>({color:colors[v],data:DATA.comparison.rows.map(row=>[row.dense_equivalent_parameters,byKey.get(row.model+'|'+v).memory.peak_allocated_bytes/1e9])})),'GB');barPhases()}function barPhases(){const c=document.getElementById('phases'),q=c.getContext('2d'),r=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*r;c.height=h*r;q.scale(r,r);const keys=['forward_ms','backward_ms','optimizer_ms'],cs=['#4ecdc4','#8fa8ff','#ffad66'],max=Math.max(...largestRuns.map(x=>x.summary.host_total_ms.median)),p={l:130,r:20,t:20,b:34};largestRuns.forEach((run,i)=>{let x=p.l,y=35+i*72;keys.forEach((key,j)=>{let value=run.summary[key].median,width=value/max*(w-p.l-p.r);q.fillStyle=cs[j];q.fillRect(x,y,width,32);x+=width});q.fillStyle='#edf3ff';q.font='12px system-ui';q.fillText(labels[run.variant.name],8,y+21);q.fillText(fmt(run.summary.host_total_ms.median,1)+' ms',x+7,y+21)});q.fillStyle='#9eabc5';q.fillText('Forward',p.l,h-10);q.fillText('Backward',p.l+72,h-10);q.fillText('Optimizer',p.l+155,h-10)}draw();addEventListener('resize',draw);
const rows=DATA.comparison.rows.flatMap(row=>variants.map(v=>{const r=byKey.get(row.model+'|'+v);return '<tr><td>'+row.label+'</td><td>'+labels[v]+'</td><td>'+fmt(r.model.actual_parameters/1e9,3)+'B</td><td>'+fmt(r.summary.host_total_ms.median,2)+'</td><td>'+fmt(r.summary.host_total_ms.p10,2)+'–'+fmt(r.summary.host_total_ms.p90,2)+'</td><td>'+fmt(r.summary.tokens_per_second.median/1000,2)+'K</td><td>'+fmt(r.summary.optimizer_ms.median,2)+'</td><td>'+fmt(r.memory.peak_allocated_bytes/1e9,2)+'</td></tr>'})).join('');document.getElementById('table').innerHTML='<table><thead><tr><th>Dense-equivalent</th><th>Setup</th><th>Actual params</th><th>Median ms</th><th>P10–P90 ms</th><th>Tokens/s</th><th>Optimizer ms</th><th>Peak GB</th></tr></thead><tbody>'+rows+'</tbody></table>';
const relation=s=>s>=1?fmt(s)+'× faster':fmt(1/s)+'× slower';document.getElementById('conclusions').innerHTML='<p>At 6.89B dense-equivalent parameters, Monarch-Muon was '+relation(largest.speedup_vs_adamw)+' than dense AdamW and '+relation(largest.speedup_vs_muon)+' than dense Muon by synchronized median step latency.</p><p>The Monarch model used '+fmt(100*largest.parameter_fraction,1)+'% of the dense parameter count. The curves therefore measure the practical speed effect of the structured parameterization plus its optimizer, not equal-parameter model quality.</p>';
document.getElementById('method').innerHTML='<p>Each point includes '+controls.accumulation_steps+' forward/backward microsteps and one optimizer update. Inputs are preallocated, so data loading is excluded. CUDA events are recorded on a dedicated stream without phase-level synchronizations; a device-wide synchronization closes every measured step and the host latency is the primary result. Lazy optimizer allocation and Dense Muon kernel compilation occur during warmup. Gradient clipping and LR scheduling are excluded.</p><p>All points used one physical GPU UUID, with three idle checks before selection, continuous '+controls.contamination_poll_seconds+' s process polling during each trial, and no foreign compute process before or after measurement. Models ran eager, with BF16 parameters, gradients, and nonscalar momentum/moment state; scalar counters may remain FP32. Dense AdamW uses PyTorch fused AdamW; both Muon variants use five BF16 Newton–Schulz iterations. This single-GPU storage mode is required to fit the 6.89B case and is not numerically equivalent to FP32-state pretraining.</p>';document.getElementById('footer').textContent='Raw result source: '+DATA.source;</script></main></body></html>""".replace("__DATA__", encoded)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    raw = json.loads(args.input.read_text())
    results = validate_payload(raw)
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": str(args.input.resolve()),
        "results": results,
        "comparison": comparison(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(payload))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
