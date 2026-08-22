#!/usr/bin/env python3
"""Build a self-contained visual report from the memory-efficient optimizer sweep."""
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
    result_is_recorded,
)

BASELINE_VARIANT = "dense_adamw"


def result_key(result: dict) -> tuple[str, str, int]:
    """Identity of a sweep point, which an out-of-memory payload carries in a
    different shape from a measured one."""
    if result.get("status") == "oom":
        return (
            result["model_size"],
            result["variant"],
            result["requested_controls"]["microbatch"],
        )
    return (
        result["model"]["name"],
        result["variant"]["name"],
        result["benchmark"]["microbatch"],
    )


def validate_payload(payload: dict) -> dict:
    results = payload.get("results", [])
    if not results:
        raise ValueError("sweep produced no results")
    unrecorded = [index for index, result in enumerate(results) if not result_is_recorded(result)]
    if unrecorded:
        raise ValueError(f"unfinished benchmark results at indices {unrecorded}")

    keys = [result_key(result) for result in results]
    if len(set(keys)) != len(keys):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"benchmark matrix contains duplicate entries: {duplicates}")

    known_models = {model["name"] for model in MODEL_SPECS}
    known_variants = {variant["name"] for variant in VARIANTS}
    unknown = [key for key in keys if key[0] not in known_models or key[1] not in known_variants]
    if unknown:
        raise ValueError(f"benchmark matrix contains unknown entries: {unknown}")

    complete = [result for result in results if result_is_complete(result)]
    if not complete:
        raise ValueError("every benchmark point ran out of memory")

    uuids = sorted({result["gpu"]["uuid"] for result in complete})
    if all(result["benchmark"].get("exclusive_gpu") for result in complete):
        # An exclusively scheduled card cannot be pinned across jobs, so the
        # guarantee downgrades from one card to one card model, and the report
        # says how many were used.
        names = {result["gpu"]["name"] for result in complete}
        if len(names) != 1:
            raise ValueError(f"benchmark used different GPU models: {sorted(names)}")
    elif len(uuids) != 1:
        raise ValueError(f"benchmark used multiple GPUs: {uuids}")
    controls = {
        json.dumps(
            {field: result["benchmark"][field] for field in COMMON_CONTROL_FIELDS},
            sort_keys=True,
        )
        for result in complete
    }
    if len(controls) != 1:
        raise ValueError(f"benchmark controls differ across trials: {controls}")
    contaminated = [
        result_key(result)
        for result in complete
        if result["gpu"].get("foreign_processes_before")
        or result["gpu"].get("foreign_processes_after")
        or result["gpu"].get("contamination_monitor", {}).get("foreign_processes_seen")
        or result["gpu"].get("contamination_monitor", {}).get("error")
    ]
    if contaminated:
        raise ValueError(f"contaminated benchmark results: {contaminated}")

    model_order = [model["name"] for model in MODEL_SPECS]
    variant_order = [variant["name"] for variant in VARIANTS]
    ordered = sorted(
        results,
        key=lambda result: (
            model_order.index(result_key(result)[0]),
            result_key(result)[2],
            variant_order.index(result_key(result)[1]),
        ),
    )
    return {"results": ordered, "complete": complete, "gpu_uuids": uuids}


def cell(result: dict, baseline: dict | None, update_proj_gap: int) -> dict:
    if result.get("status") == "oom":
        return {"status": "oom"}
    summary = result["summary"]
    memory = result["memory"]
    median_ms = summary["host_total_ms"]["median"]
    resample = result.get("resample_summary")
    resample_ms = resample["host_total_ms"]["median"] if resample else None
    entry = {
        "status": "complete",
        "median_ms": median_ms,
        "p10_ms": summary["host_total_ms"]["p10"],
        "p90_ms": summary["host_total_ms"]["p90"],
        "optimizer_ms": summary["optimizer_ms"]["median"],
        "tokens_per_second": summary["tokens_per_second"]["median"],
        "peak_allocated_bytes": memory["peak_allocated_bytes"],
        "optimizer_state_bytes": memory["optimizer_state_bytes"],
        "optimizer_moment_bytes": memory["optimizer_moment_bytes"],
        "optimizer_projector_bytes": memory["optimizer_projector_bytes"],
        "resample_ms": resample_ms,
        "resample_peak_allocated_bytes": (
            result["resample_memory"]["peak_allocated_bytes"] if result.get("resample_memory") else None
        ),
        "resample_extra_ms": max(0.0, resample_ms - median_ms) if resample_ms is not None else None,
        # Filled in by amortize(), which carries the rebuild cost across the
        # batch-size axis it does not depend on.
        "amortized_ms": median_ms,
    }
    if baseline is not None and baseline.get("status") == "complete":
        entry["speedup_vs_baseline"] = baseline["median_ms"] / median_ms
        entry["peak_memory_ratio"] = entry["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]
        entry["state_memory_ratio"] = (
            entry["optimizer_state_bytes"] / baseline["optimizer_state_bytes"]
            if baseline["optimizer_state_bytes"]
            else None
        )
    return entry


def amortize(grid: list[dict], update_proj_gap: int) -> None:
    """A rebuild reads the gradient's shape, not the batch, so it is measured at
    one batch size and charged to every batch size at its true frequency."""
    for model in grid:
        measured = {}
        for row in model["rows"]:
            for name, entry in row["cells"].items():
                if entry.get("resample_extra_ms") is not None and name not in measured:
                    measured[name] = (entry["resample_extra_ms"], row["microbatch"])
        for row in model["rows"]:
            for name, entry in row["cells"].items():
                if entry["status"] != "complete" or name not in measured:
                    continue
                extra, source = measured[name]
                entry["amortized_ms"] = entry["median_ms"] + extra / update_proj_gap
                entry["resample_extra_ms"] = extra
                entry["resample_measured_at_microbatch"] = source


def comparison(validated: dict) -> dict:
    results = validated["results"]
    update_proj_gap = validated["complete"][0]["benchmark"]["update_proj_gap"]
    by_key = {result_key(result): result for result in results}
    present_models = [model for model in MODEL_SPECS if any(key[0] == model["name"] for key in by_key)]
    present_variants = [
        variant for variant in VARIANTS if any(key[1] == variant["name"] for key in by_key)
    ]
    microbatches = sorted({key[2] for key in by_key})

    grid = []
    for model in present_models:
        rows = []
        for microbatch in microbatches:
            baseline_result = by_key.get((model["name"], BASELINE_VARIANT, microbatch))
            baseline = (
                cell(baseline_result, None, update_proj_gap) if baseline_result is not None else None
            )
            cells = {}
            for variant in present_variants:
                result = by_key.get((model["name"], variant["name"], microbatch))
                if result is None:
                    continue
                cells[variant["name"]] = cell(result, baseline, update_proj_gap)
            if cells:
                rows.append({"microbatch": microbatch, "cells": cells})
        if rows:
            grid.append(
                {
                    "name": model["name"],
                    "label": model["label"],
                    "dense_equivalent_parameters": model["dense_params_expected"],
                    "rows": rows,
                }
            )

    amortize(grid, update_proj_gap)

    # Largest microbatch each variant still fits, per model: the practical
    # headline of a memory-efficiency benchmark.
    capacity = []
    for model in grid:
        fits = {}
        for row in model["rows"]:
            for name, entry in row["cells"].items():
                if entry["status"] == "complete":
                    fits[name] = max(fits.get(name, 0), row["microbatch"])
        capacity.append({"name": model["name"], "label": model["label"], "fits": fits})

    return {
        "models": grid,
        "variants": [
            variant for variant in present_variants
        ],
        "microbatches": microbatches,
        "capacity": capacity,
        "update_proj_gap": update_proj_gap,
        "baseline": BASELINE_VARIANT,
    }


def render(payload: dict) -> str:
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", encoded)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Memory-efficient optimizer benchmark</title><style>
:root{color-scheme:dark;--bg:#080d18;--panel:#111a2b;--grid:#293650;--text:#edf3ff;--muted:#9eabc5}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#173253 0,var(--bg) 42%);color:var(--text);font-family:Inter,ui-sans-serif,system-ui}
main{width:min(1280px,94vw);margin:auto;padding:44px 0 72px}h1{font-size:clamp(2rem,4.6vw,3.6rem);letter-spacing:-.05em;margin:.15em 0}h2{margin:36px 0 14px}h3{margin:0 0 10px}
.eyebrow{color:#6ee7c7;font-weight:800;letter-spacing:.14em;text-transform:uppercase;font-size:.75rem}p{color:var(--muted);line-height:1.65}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,.chart,.note{background:#111a2bea;border:1px solid #283651;border-radius:16px;box-shadow:0 16px 48px #0004}
.card{padding:18px}.card small{color:var(--muted)}.value{font-size:1.5rem;font-weight:780;margin-top:8px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart{padding:16px}canvas{width:100%;height:300px}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:.82rem;margin:10px 0}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
.tabs{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}.tab{background:#16223a;border:1px solid #283651;color:var(--muted);border-radius:999px;padding:7px 15px;cursor:pointer;font:inherit;font-size:.85rem}
.tab[aria-selected=true]{background:#6ee7c7;border-color:#6ee7c7;color:#062018;font-weight:700}
.scroll{overflow-x:auto;border-radius:14px}table{width:100%;border-collapse:collapse;background:var(--panel);font-size:.87rem}
th,td{padding:9px 11px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
tbody tr.base td{background:#16223a}.oom{color:#ff8f8f;font-weight:700}.note{padding:18px 22px}footer{margin-top:30px;color:var(--muted);font-size:.8rem}
@media(max-width:900px){.cards{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}}
</style></head><body><main>
<div class="eyebrow">End-to-end training-step benchmark</div><h1>Memory-efficient optimizers<br>time and memory at scale</h1>
<p id="subtitle"></p><div class="cards" id="cards"></div>
<h2>Per model</h2><div class="tabs" id="tabs"></div><div class="legend" id="legend"></div>
<div class="charts">
<section class="chart"><h3>Peak allocated memory vs batch size</h3><canvas id="memory"></canvas></section>
<section class="chart"><h3>Median step latency vs batch size</h3><canvas id="latency"></canvas></section>
<section class="chart"><h3>Optimizer state at batch size 1</h3><canvas id="state"></canvas></section>
<section class="chart"><h3>Projection rebuild vs steady step</h3><canvas id="resample"></canvas></section>
</div>
<h2>Results</h2><div class="scroll" id="table"></div>
<h2>Batch size that still fits</h2><div class="scroll" id="capacity"></div>
<h2>Conclusions</h2><div class="note" id="conclusions"></div>
<h2>Method and fairness controls</h2><div class="note" id="method"></div><footer id="footer"></footer>
<script>
const DATA=__DATA__;const C=DATA.comparison;
const COLORS={monarch_muon:'#6ee7c7',dense_adamw:'#ffad66',dense_muon:'#8fa8ff',galore:'#f78fb3',frugal:'#a0e57c',apollo:'#5ad2f4',apollo_mini:'#c9a0ff',fira:'#ffd166'};
const LABELS=Object.fromEntries(C.variants.map(v=>[v.name,v.label]));
const fmt=(x,d=2)=>x===null||x===undefined?'--':Number(x).toFixed(d);
const gb=b=>b===null||b===undefined?'--':(b/1e9).toFixed(2);
const controls=DATA.controls;let active=C.models[0].name;
document.getElementById('subtitle').textContent='One '+DATA.gpu_name+', '+(controls.tokens_per_step/1024)+'K tokens per optimizer step held fixed across batch sizes, sequence length '+controls.sequence_length+', BF16 storage, '+controls.measured_steps+' measured steps after '+controls.warmup_steps+' warmups. Generated '+DATA.generated_at+'.';
document.getElementById('cards').innerHTML=DATA.headline.map(x=>'<div class="card"><small>'+x[0]+'</small><div class="value">'+x[1]+'</div></div>').join('');
document.getElementById('legend').innerHTML=C.variants.map(v=>'<span><i class="dot" style="background:'+COLORS[v.name]+'"></i>'+v.label+'</span>').join('');
document.getElementById('tabs').innerHTML=C.models.map(m=>'<button class="tab" role="tab" aria-selected="'+(m.name===active)+'" data-model="'+m.name+'">'+m.label+'</button>').join('');
document.getElementById('tabs').addEventListener('click',e=>{const b=e.target.closest('.tab');if(!b)return;active=b.dataset.model;
  document.querySelectorAll('.tab').forEach(t=>t.setAttribute('aria-selected',t.dataset.model===active));draw();});
const model=()=>C.models.find(m=>m.name===active);
function chart(id,series,ylabel,xs,xfmt){const c=document.getElementById(id),q=c.getContext('2d'),r=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;
  c.width=w*r;c.height=h*r;q.setTransform(r,0,0,r,0,0);q.clearRect(0,0,w,h);
  const ys=series.flatMap(s=>s.data.map(p=>p[1]));if(!ys.length){q.fillStyle='#9eabc5';q.font='12px system-ui';q.fillText('no measured points',12,24);return;}
  const ymax=Math.max(...ys)*1.1||1,p={l:60,r:16,t:14,b:38};
  const lx=Math.log2(xs[0]),hx=Math.log2(xs[xs.length-1]);
  const sx=x=>xs.length<2?p.l+(w-p.l-p.r)/2:p.l+(Math.log2(x)-lx)/(hx-lx)*(w-p.l-p.r);
  const sy=y=>h-p.b-y/ymax*(h-p.t-p.b);
  q.font='11px system-ui';q.strokeStyle='#293650';q.fillStyle='#9eabc5';
  for(let i=0;i<5;i++){const y=ymax*i/4,py=sy(y);q.beginPath();q.moveTo(p.l,py);q.lineTo(w-p.r,py);q.stroke();q.fillText(fmt(y,ymax<10?1:0),6,py+4);}
  xs.forEach(x=>q.fillText(xfmt(x),sx(x)-8,h-14));q.fillText(ylabel,7,12);
  series.forEach(s=>{q.strokeStyle=s.color;q.fillStyle=s.color;q.lineWidth=2.2;q.beginPath();
    s.data.forEach((pt,i)=>{const X=sx(pt[0]),Y=sy(pt[1]);i?q.lineTo(X,Y):q.moveTo(X,Y);});q.stroke();
    s.data.forEach(pt=>{q.beginPath();q.arc(sx(pt[0]),sy(pt[1]),3.2,0,Math.PI*2);q.fill();});});}
function bars(id,items,ylabel){const c=document.getElementById(id),q=c.getContext('2d'),r=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;
  c.width=w*r;c.height=h*r;q.setTransform(r,0,0,r,0,0);q.clearRect(0,0,w,h);
  if(!items.length){q.fillStyle='#9eabc5';q.font='12px system-ui';q.fillText('no measured points',12,24);return;}
  const max=Math.max(...items.map(i=>i.value))||1,p={l:110,r:60,t:10,b:20},bh=Math.min(30,(h-p.t-p.b)/items.length-6);
  q.font='12px system-ui';items.forEach((it,i)=>{const y=p.t+i*((h-p.t-p.b)/items.length),wd=it.value/max*(w-p.l-p.r);
    q.fillStyle=it.color;q.fillRect(p.l,y,wd,bh);q.fillStyle='#edf3ff';q.fillText(it.label,6,y+bh*0.72);
    q.fillStyle='#9eabc5';q.fillText(it.text,p.l+wd+7,y+bh*0.72);});q.fillStyle='#9eabc5';q.fillText(ylabel,6,h-6);}
function series(field,scale){return C.variants.map(v=>({color:COLORS[v.name],data:model().rows
  .filter(r=>r.cells[v.name]&&r.cells[v.name].status==='complete')
  .map(r=>[r.microbatch,r.cells[v.name][field]*scale])})).filter(s=>s.data.length);}
function draw(){const xs=C.microbatches;
  chart('memory',series('peak_allocated_bytes',1e-9),'GB',xs,x=>'bs '+x);
  chart('latency',series('median_ms',1),'ms',xs,x=>'bs '+x);
  const first=model().rows[0];
  bars('state',C.variants.filter(v=>first&&first.cells[v.name]&&first.cells[v.name].status==='complete')
    .map(v=>({label:v.label,color:COLORS[v.name],value:first.cells[v.name].optimizer_state_bytes,
      text:gb(first.cells[v.name].optimizer_state_bytes)+' GB'})),'optimizer state, GB at batch size '+(first?first.microbatch:'-'));
  bars('resample',C.variants.filter(v=>first&&first.cells[v.name]&&first.cells[v.name].resample_ms!==null&&first.cells[v.name].status==='complete')
    .map(v=>({label:v.label,color:COLORS[v.name],value:first.cells[v.name].resample_ms,
      text:fmt(first.cells[v.name].resample_ms,0)+' ms vs '+fmt(first.cells[v.name].median_ms,0)+' ms'})),'rebuild step, ms');
  table();}
function table(){const m=model();let html='<table><thead><tr><th>Batch</th><th>Setup</th><th>Median ms</th><th>P10-P90</th><th>Optimizer ms</th><th>Tokens/s</th><th>Peak GB</th><th>State GB</th><th>Proj GB</th><th>Rebuild +ms</th><th>Amortized ms</th><th>vs AdamW</th></tr></thead><tbody>';
  m.rows.forEach(row=>{C.variants.forEach(v=>{const e=row.cells[v.name];if(!e)return;
    const base=v.name===C.baseline?' class="base"':'';
    if(e.status==='oom'){html+='<tr'+base+'><td>'+row.microbatch+'</td><td>'+v.label+'</td><td class="oom" colspan="10">out of memory</td></tr>';return;}
    html+='<tr'+base+'><td>'+row.microbatch+'</td><td>'+v.label+'</td><td>'+fmt(e.median_ms)+'</td><td>'+fmt(e.p10_ms)+'-'+fmt(e.p90_ms)+'</td><td>'+fmt(e.optimizer_ms)+'</td><td>'+fmt(e.tokens_per_second/1000,1)+'K</td><td>'+gb(e.peak_allocated_bytes)+'</td><td>'+gb(e.optimizer_state_bytes)+'</td><td>'+gb(e.optimizer_projector_bytes)+'</td><td>'+fmt(e.resample_extra_ms,0)+'</td><td>'+fmt(e.amortized_ms)+'</td><td>'+(e.speedup_vs_baseline?fmt(e.speedup_vs_baseline)+'x':'--')+'</td></tr>';});});
  document.getElementById('table').innerHTML=html+'</tbody></table>';}
document.getElementById('capacity').innerHTML='<table><thead><tr><th>Model</th>'+C.variants.map(v=>'<th>'+v.label+'</th>').join('')+'</tr></thead><tbody>'+
  C.capacity.map(c=>'<tr><td>'+c.label+'</td>'+C.variants.map(v=>'<td>'+(c.fits[v.name]?c.fits[v.name]:'<span class="oom">--</span>')+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
document.getElementById('conclusions').innerHTML=DATA.conclusions.map(t=>'<p>'+t+'</p>').join('');
document.getElementById('method').innerHTML=DATA.method.map(t=>'<p>'+t+'</p>').join('');
document.getElementById('footer').textContent='Raw result source: '+DATA.source;
draw();addEventListener('resize',draw);
</script></main></body></html>"""


def headline(comparison_data: dict, controls: dict) -> list[list[str]]:
    largest = comparison_data["models"][-1]
    first_row = largest["rows"][0]
    baseline = first_row["cells"].get(BASELINE_VARIANT, {})
    savers = [
        (name, entry)
        for name, entry in first_row["cells"].items()
        if entry.get("status") == "complete" and entry.get("state_memory_ratio") is not None
    ]
    best = min(savers, key=lambda item: item[1]["state_memory_ratio"], default=None)
    labels = {variant["name"]: variant["label"] for variant in comparison_data["variants"]}
    cards = [
        [f"Largest model measured", largest["label"]],
        [
            "Baseline optimizer state",
            f"{baseline.get('optimizer_state_bytes', 0) / 1e9:.1f} GB" if baseline else "--",
        ],
    ]
    if best is not None:
        cards.append(
            [f"Smallest state ({labels[best[0]]})", f"{100 * best[1]['state_memory_ratio']:.0f}% of AdamW"]
        )
    cards.append(["Tokens per optimizer step", f"{controls['tokens_per_step'] // 1024}K"])
    return cards


def conclusions(comparison_data: dict) -> list[str]:
    labels = {variant["name"]: variant["label"] for variant in comparison_data["variants"]}
    largest = comparison_data["models"][-1]
    first_row = largest["rows"][0]
    lines = []
    entries = [
        (labels[name], entry)
        for name, entry in first_row["cells"].items()
        if entry.get("status") == "complete" and name != BASELINE_VARIANT
    ]
    if entries:
        memory = ", ".join(
            f"{label} {100 * entry['state_memory_ratio']:.0f}%"
            for label, entry in entries
            if entry.get("state_memory_ratio") is not None
        )
        speed = ", ".join(
            f"{label} {entry['speedup_vs_baseline']:.2f}x"
            for label, entry in entries
            if entry.get("speedup_vs_baseline") is not None
        )
        lines.append(
            f"At {largest['label']} and batch size {first_row['microbatch']}, optimizer state relative "
            f"to dense AdamW: {memory}. Step latency relative to dense AdamW: {speed}."
        )
    capacity = comparison_data["capacity"][-1]
    fits = ", ".join(
        f"{labels[name]} {value}" for name, value in sorted(capacity["fits"].items(), key=lambda kv: -kv[1])
    )
    lines.append(f"Largest microbatch that fits at {capacity['label']}: {fits}.")
    lines.append(
        "The reference optimizers are fused or foreach kernels while the memory-efficient ones are "
        "per-parameter Python loops, so a latency gap here is an implementation gap as much as an "
        "algorithmic one. Optimizer state and peak memory are the numbers that carry across "
        "implementations."
    )
    return lines


def method(comparison_data: dict, controls: dict, exclusive: bool, uuids: list[str]) -> list[str]:
    cards = (
        "one exclusively scheduled GPU"
        if len(uuids) == 1
        else f"{len(uuids)} exclusively scheduled GPUs of the same model"
    )
    exclusivity = (
        f"Points ran on {cards}, so no foreign-process polling was possible or necessary. "
        "A scheduled card cannot be pinned across jobs, so where more than one was used the "
        "control is same card model rather than same physical card."
        if exclusive
        else f"All points used one physical GPU UUID, with three idle checks before selection, "
        f"continuous {controls['contamination_poll_seconds']} s process polling during each trial, "
        f"and no foreign compute process before or after measurement."
    )
    return [
        f"Every point runs {controls['tokens_per_step'] // 1024}K tokens per optimizer step at "
        f"sequence length {controls['sequence_length']}; the microbatch is traded against gradient "
        f"accumulation so the token budget per step is identical across the batch-size axis. Inputs "
        f"are preallocated, so data loading is excluded. CUDA events are recorded on a dedicated "
        f"stream without phase-level synchronizations; a device-wide synchronization closes every "
        f"measured step and the host latency is the primary result.",
        f"Projections are built during warmup and rebuilt every {comparison_data['update_proj_gap']} "
        f"steps, which is longer than the measured window, so the median is a clean steady-state "
        f"step. The rebuild is measured separately in its own window with the gap forced to 1, with "
        f"its own peak-memory reset, because the FP32 factorization workspace never appears in the "
        f"steady state. A rebuild factorizes the gradient, whose shape does not depend on the "
        f"batch, so it is measured once per model and optimizer and charged to every batch size; "
        f"the amortized column adds it back at its true frequency.",
        f"Models run eager with BF16 parameters, gradients, and nonscalar momentum/moment state; "
        f"scalar counters may remain FP32. Projection matrices are BF16 and are counted in the "
        f"optimizer-state figure, which is a fix over the earlier report where projector objects "
        f"were silently worth zero bytes. Memory-efficient variants project only the parameters the "
        f"model marks is_proj_params, so embeddings and the LM head keep full AdamW state. "
        f"{exclusivity}",
    ]


def load_results(paths: list[Path]) -> dict:
    """Accepts a sweep results.json or an output directory of per-point files,
    so parallel jobs writing into their own directories still make one report."""
    results = []
    for path in paths:
        if path.is_dir():
            runs = path / "runs"
            for entry in sorted((runs if runs.is_dir() else path).glob("*.json")):
                results.append(json.loads(entry.read_text()))
        else:
            results.extend(json.loads(path.read_text()).get("results", []))
    return {"results": results}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, nargs="+",
                        help="results.json files or sweep output directories")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    validated = validate_payload(load_results(args.input))
    comparison_data = comparison(validated)
    reference = validated["complete"][0]
    controls = {field: reference["benchmark"][field] for field in COMMON_CONTROL_FIELDS}
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": ", ".join(str(path.resolve()) for path in args.input),
        "gpu_uuids": validated["gpu_uuids"],
        "gpu_name": reference["gpu"]["name"],
        "controls": controls,
        "comparison": comparison_data,
        "headline": headline(comparison_data, controls),
        "conclusions": conclusions(comparison_data),
        "method": method(
            comparison_data, controls, bool(controls.get("exclusive_gpu")), validated["gpu_uuids"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(payload))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
