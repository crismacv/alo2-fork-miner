"""R8 comparison: leader vs ours (or any --ours-ref / --label variant).

Results accumulate across runs in /tmp/dashboard/runs.jsonl. Assets (ref +
renders + 8-view sweep) are saved to /tmp/dashboard/assets/<stem>__<runid>/.
The static dashboard at /tmp/dashboard/r9_compare.html is regenerated from
the full jsonl every run, so each new invocation just adds 3 more rows.

Usage:
  python compare_r8.py --n 3 \\
      --leader-ref 5dc2dab --ours-ref HEAD --label "R9-prompts" \\
      --judge-url http://localhost:8003/v1 --judge-model zai-org/GLM-4.6V-Flash
"""
from __future__ import annotations
import argparse
import asyncio
import base64
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline_service"))
sys.path.insert(0, str(ROOT.parent / "qwen_pipeline"))

ENV = ROOT.parent.parent / ".env"
for line in ENV.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("compare_r8")

import render as render_mod  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

OR_BASE = "https://openrouter.ai/api/v1"
OR_KEY = os.environ["OPENROUTER_API_KEY"]

COMP_REPO = Path("/tmp/404-active-competition")
ROUND_N = 8
DASH_DIR = Path("/tmp/dashboard")
RUNS_LOG = DASH_DIR / "runs.jsonl"
ASSET_DIR = DASH_DIR / "assets"


# ---------- R8 prompts ----------

def refresh_repo() -> None:
    if COMP_REPO.exists():
        subprocess.run(["git", "-C", str(COMP_REPO), "pull", "--quiet"], check=False)
    else:
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/404-Repo/404-active-competition.git",
            str(COMP_REPO),
        ], check=True)


def load_r8_prompts() -> list[tuple[str, str]]:
    refresh_repo()
    p = COMP_REPO / "rounds" / str(ROUND_N) / "prompts.txt"
    if not p.exists():
        raise FileNotFoundError(f"R{ROUND_N} prompts not in repo: {p}")
    pairs = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("http"):
            url = line
            stem = line.rsplit("/", 1)[1].rsplit(".", 1)[0]
        else:
            url = f"https://sn12domain.org/{line}"
            stem = line.rsplit(".", 1)[0].rsplit("/", 1)[-1]
        pairs.append((stem, url))
    return pairs


def download_ref(stem: str, url: str) -> bytes:
    cache = Path("/tmp/r8_refs") / f"{stem}.png"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    cache.write_bytes(data)
    return data


# ---------- Per-version subprocess runner ----------

GEN_RUNNER_SRC = '''"""Per-stem generator. Invoked as subprocess from the worktree's working dir."""
from __future__ import annotations
import argparse, asyncio, json, logging, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline_service"))
sys.path.insert(0, str(ROOT.parent / "qwen_pipeline"))

ENV = ROOT.parent.parent / ".env"
for line in ENV.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("gen_one")

from llm.session_store import SessionStore
from modules.scene_coder.agent import SceneCoderAgent
from modules.critic.agent import CriticAgent
from openai import AsyncOpenAI
import render as render_mod

OR_BASE = "https://openrouter.ai/api/v1"
OR_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "qwen/qwen3.5-397b-a17b"

ENSEMBLE = 6
MAX_ITER = 2
STOP_SCORE = 0.85


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out-js", required=True)
    p.add_argument("--out-meta", required=True)
    args = p.parse_args()

    img = Path(args.ref).read_bytes()
    client = AsyncOpenAI(base_url=OR_BASE, api_key=OR_KEY, timeout=240)
    store = SessionStore()
    coder = SceneCoderAgent(client=client, model=MODEL, session_store=store,
                            temperature=0.0, seed=42, max_tokens=8192,
                            backend="vllm", total_stages=6)
    critic = CriticAgent(client=client, model=MODEL, max_tokens=4096, seed=42,
                         reasoning_effort="medium", ensemble_size=1,
                         backend="vllm", total_stages=6)

    t0 = time.time()
    cand_ids = [f"{args.stem}-k{k}" for k in range(ENSEMBLE)]
    async def _gen(k):
        try:
            return await coder.code(task_id=cand_ids[k], image_bytes=img,
                                    image_mime="image/png", candidate_id=k,
                                    seed_override=42 + k * 7919,
                                    temperature_override=0.0 if k == 0 else 0.3)
        except Exception as e:
            log.warning(f"  k{k} fail: {type(e).__name__}: {str(e)[:80]}")
            return None
    cands = await asyncio.gather(*[_gen(k) for k in range(ENSEMBLE)])
    pairs = [(cand_ids[k], c) for k, c in enumerate(cands) if c]
    if not pairs:
        Path(args.out_meta).write_text(json.dumps({"status": "coder_fail"}))
        sys.exit(2)

    renders = await asyncio.gather(*[render_mod.render_front(c) for _, c in pairs])
    survivors = [(t, c, r) for (t, c), r in zip(pairs, renders) if r is not None]
    if not survivors:
        best_js = pairs[0][1]
        best_score = None
    else:
        async def _crit(t, c, r):
            try:
                return await critic.critique(task_id=f"{t}-crit", image_bytes=img,
                                              image_mime="image/png", render_png=r,
                                              artifact_context={"kind":"coder_v1","osd":None,"js_code":c})
            except Exception as e:
                log.warning(f"  critic fail: {type(e).__name__}")
                return None
        reports = await asyncio.gather(*[_crit(*s) for s in survivors])
        scored = [(t, c, r, rep) for (t, c, r), rep in zip(survivors, reports) if rep]
        if not scored:
            best_js, best_score = survivors[0][1], None
        else:
            scored.sort(key=lambda x: -x[3].overall_score)
            best_tid, best_js, best_render, best_report = scored[0]
            best_score = best_report.overall_score
            best_k = int(best_tid.rsplit("-k", 1)[1])
            if best_k > 0:
                store.rename_actor(best_tid, f"coder#k{best_k}", "coder")
            for it in range(MAX_ITER):
                if best_report.stop or best_report.overall_score >= STOP_SCORE:
                    break
                try:
                    patched = await coder.code_critic_repair(
                        task_id=best_tid, image_bytes=img, image_mime="image/png",
                        render_png=best_render,
                        overall_score=best_report.overall_score,
                        issues=best_report.issues,
                        matching_aspects=best_report.matching_aspects, osd=None)
                except Exception as e:
                    log.warning(f"  patcher fail: {type(e).__name__}")
                    break
                new_r = await render_mod.render_front(patched)
                if new_r is None:
                    continue
                new_rep = await _crit(best_tid, patched, new_r)
                if new_rep is None:
                    continue
                if new_rep.overall_score > best_report.overall_score:
                    best_js, best_render, best_report = patched, new_r, new_rep
                    best_score = new_rep.overall_score

    Path(args.out_js).write_text(best_js)
    Path(args.out_meta).write_text(json.dumps({
        "status": "ok", "best_score": best_score, "dt": time.time() - t0,
        "n_ensemble": len(pairs), "n_survivors": len(survivors),
    }))
    print("OK")


if __name__ == "__main__":
    asyncio.run(main())
'''


def ensure_worktree(commit_ref: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", commit_ref)[:16]
    wt = ROOT.parent / f"alo2_fork_{safe}"
    if not wt.exists():
        subprocess.run([
            "git", "-C", str(ROOT), "worktree", "add", "-f",
            str(wt), commit_ref,
        ], check=True)
        log.info(f"created worktree {wt}")
    (wt / "_gen_one.py").write_text(GEN_RUNNER_SRC)
    return wt


async def run_one_subprocess(work_dir: Path, stem: str, ref_path: Path) -> tuple[str | None, dict]:
    out_js = Path(f"/tmp/gen_{work_dir.name}_{stem}.js")
    out_meta = Path(f"/tmp/gen_{work_dir.name}_{stem}.json")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(work_dir / "_gen_one.py"),
        "--stem", stem, "--ref", str(ref_path),
        "--out-js", str(out_js), "--out-meta", str(out_meta),
        cwd=str(work_dir),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    meta = json.loads(out_meta.read_text()) if out_meta.exists() else {"status": "no_meta"}
    js = out_js.read_text() if out_js.exists() else None
    if proc.returncode != 0:
        log.warning(f"  {work_dir.name}/{stem[:10]} rc={proc.returncode} err={stderr.decode()[-300:]}")
    return js, meta


# ---------- Judge (3-vote via local GLM-4.6V-Flash) ----------

S1_SYSTEM = (
    "You are a specialized 3D model evaluation system.\n"
    "Analyze visual quality and prompt adherence with expert precision.\n"
    "Always respond with valid JSON only."
)
S1_USER = (
    "You see two 3D models rendered from slightly left of the front.\n"
    "The reference image shows the target object.\n\n"
    "Which model is a more faithful 3D reproduction of the reference?\n\n"
    "Penalty 0-10:\n"
    "0 = Perfect match to reference\n"
    "3 = Minor issues\n"
    "5 = Moderate issues\n"
    "7 = Major issues\n"
    "10 = Completely wrong object\n\n"
    'Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}'
)
_re_p = re.compile(r'"penalty_1"\s*:\s*(\d+).*?"penalty_2"\s*:\s*(\d+)', re.DOTALL)
_re_p2 = re.compile(r'penalty_?1[^0-9]+(\d+).*?penalty_?2[^0-9]+(\d+)', re.DOTALL | re.IGNORECASE)


def _b64url(b: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(b).decode()}"


async def judge_one(client: AsyncOpenAI, model: str, ref: bytes, a: bytes, b: bytes,
                    seed: int, swap: bool) -> tuple[int, int] | None:
    A, B = (b, a) if swap else (a, b)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": S1_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": "Reference image (target object):"},
                    {"type": "image_url", "image_url": {"url": _b64url(ref)}},
                    {"type": "text", "text": "3D model 1:"},
                    {"type": "image_url", "image_url": {"url": _b64url(A)}},
                    {"type": "text", "text": "3D model 2:"},
                    {"type": "image_url", "image_url": {"url": _b64url(B)}},
                    {"type": "text", "text": S1_USER},
                ]},
            ],
            max_tokens=2048, temperature=0.0, seed=seed,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        m = _re_p.search(text) or _re_p2.search(text)
        if not m:
            log.warning(f"  judge: no parse from: {text[:200]}")
            return None
        p1, p2 = int(m.group(1)), int(m.group(2))
        p1, p2 = max(0, min(10, p1)), max(0, min(10, p2))
        return (p2, p1) if swap else (p1, p2)
    except Exception as e:
        log.warning(f"  judge err: {type(e).__name__}: {str(e)[:80]}")
        return None


async def judge_3vote(client: AsyncOpenAI, model: str, ref: bytes, leader: bytes, ours: bytes,
                      base_seed: int = 42) -> dict:
    votes = await asyncio.gather(*[
        judge_one(client, model, ref, leader, ours, base_seed + i * 13, bool(i & 1))
        for i in range(3)
    ])
    valid = [v for v in votes if v is not None]
    if not valid:
        return {"leader": None, "ours": None, "verdict": "JUDGE_FAIL", "votes": votes}
    pa = sum(v[0] for v in valid) / len(valid)
    pb = sum(v[1] for v in valid) / len(valid)
    if abs(pa - pb) < 0.5:
        verdict = "DRAW"
    elif pb < pa:
        verdict = "OURS_WIN"
    else:
        verdict = "LEADER_WIN"
    return {"leader": round(pa, 2), "ours": round(pb, 2), "verdict": verdict,
            "votes": votes, "n_valid": len(valid)}


# ---------- Asset persistence ----------

def save_asset(b: bytes | None, path: Path) -> str | None:
    if not b:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b)
    return str(path.relative_to(DASH_DIR))


def append_run_row(row: dict) -> None:
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------- Dashboard HTML (reads all runs.jsonl) ----------

DASH_HTML_PATH = DASH_DIR / "r9_compare.html"


def write_dashboard():
    rows = []
    if RUNS_LOG.exists():
        for line in RUNS_LOG.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)

    # Group by (leader_ref, ours_ref, label) for cumulative tally
    groups: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("leader_ref", "?"), r.get("ours_ref", "?"), r.get("label", ""))
        g = groups.setdefault(key, {"w": 0, "d": 0, "l": 0, "f": 0, "rows": []})
        v = r.get("judge", {}).get("verdict", "JUDGE_FAIL")
        if v == "OURS_WIN":
            g["w"] += 1
        elif v == "DRAW":
            g["d"] += 1
        elif v == "LEADER_WIN":
            g["l"] += 1
        else:
            g["f"] += 1
        g["rows"].append(r)

    overall_w = sum(g["w"] for g in groups.values())
    overall_d = sum(g["d"] for g in groups.values())
    overall_l = sum(g["l"] for g in groups.values())
    overall_f = sum(g["f"] for g in groups.values())
    overall_total = overall_w + overall_d + overall_l

    parts = [f"""<!doctype html>
<meta charset=utf-8>
<title>Duel R{ROUND_N} · cumulative</title>
<style>
:root {{
  --bg:#0d0d10; --panel:#17171b; --panel2:#1d1d22; --border:#2a2a32;
  --fg:#e4e4ea; --muted:#8a8a92; --accent:#5cd3a5; --danger:#ff7676; --warn:#f5c25b;
}}
* {{ box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       margin:0; background:var(--bg); color:var(--fg); }}
header {{ padding:14px 24px; background:var(--panel); border-bottom:1px solid var(--border); }}
h1 {{ margin:0 0 6px; font-size:16px; font-weight:600; letter-spacing:.02em; }}
.tally-row {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; font-size:13px; }}
.tally {{ display:inline-flex; gap:10px; padding:6px 14px; border-radius:6px;
          background:var(--panel2); border:1px solid var(--border);
          font-variant-numeric:tabular-nums; }}
.tally.big {{ font-size:15px; padding:8px 18px; }}
.tally .w {{ color:var(--accent); font-weight:600; }}
.tally .l {{ color:var(--danger); font-weight:600; }}
.tally .d {{ color:var(--warn); font-weight:600; }}
.tally .label {{ color:var(--muted); font-size:11px; padding-right:6px;
                 border-right:1px solid var(--border); margin-right:4px; }}
.meta {{ color:var(--muted); font-size:12px; }}

main {{ padding:18px 24px; display:flex; flex-direction:column; gap:18px; }}
.duel {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
        padding:16px; }}
.duel-head {{ display:flex; justify-content:space-between; align-items:center;
              margin-bottom:12px; gap:12px; flex-wrap:wrap; }}
.duel-head .stem {{ color:var(--muted); font-size:11px; font-family:monospace; }}
.duel-head .ts {{ color:var(--muted); font-size:11px; }}
.badges {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.badge {{ padding:3px 10px; border-radius:4px; font-size:11px; font-weight:600;
         text-transform:uppercase; letter-spacing:.04em; }}
.badge.win {{ background:rgba(92,211,165,.15); color:var(--accent); border:1px solid rgba(92,211,165,.4); }}
.badge.loss {{ background:rgba(255,118,118,.12); color:var(--danger); border:1px solid rgba(255,118,118,.35); }}
.badge.draw {{ background:rgba(245,194,91,.12); color:var(--warn); border:1px solid rgba(245,194,91,.35); }}
.badge.fail {{ background:rgba(120,120,128,.18); color:var(--muted); border:1px solid var(--border); }}

.three-col {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-bottom:12px; }}
.col {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px;
       overflow:hidden; }}
.col .label2 {{ padding:8px 10px; font-size:11px; color:var(--muted);
              border-bottom:1px solid var(--border); display:flex;
              justify-content:space-between; gap:6px;
              text-transform:uppercase; letter-spacing:.05em; }}
.col .label2 .num {{ font-family:monospace; color:var(--fg); font-weight:600; }}
.col img.main {{ width:100%; aspect-ratio:1/1; object-fit:cover; display:block; background:#222; }}

.strip {{ display:grid; grid-template-columns:repeat(8, 1fr); gap:4px;
         background:var(--panel2); padding:6px; border:1px solid var(--border);
         border-radius:6px; margin-bottom:6px; }}
.strip-label {{ font-size:10px; color:var(--muted); padding:6px 0 4px 2px;
                text-transform:uppercase; letter-spacing:.05em;
                display:flex; gap:8px; align-items:center; }}
.strip img {{ width:100%; aspect-ratio:1/1; object-fit:cover; background:#222;
             border-radius:3px; display:block; }}
.strip .missing {{ width:100%; aspect-ratio:1/1; background:#1a1a1f;
                  border:1px dashed var(--border); border-radius:3px;
                  display:flex; align-items:center; justify-content:center;
                  color:var(--muted); font-size:10px; }}

.votes {{ display:flex; gap:8px; font-size:11px; color:var(--muted); flex-wrap:wrap; }}
.votes span {{ background:var(--panel2); padding:3px 8px; border-radius:3px;
              border:1px solid var(--border); font-family:monospace; }}
.config-tag {{ background:var(--panel2); padding:3px 8px; border-radius:3px;
              border:1px solid var(--border); font-size:10px; color:var(--muted);
              font-family:monospace; }}
</style>
<header>
  <h1>Duel · R{ROUND_N} · cumulative across all runs</h1>
  <div class=tally-row>
    <div class="tally big">
      <span class=label>OVERALL ({overall_total})</span>
      <span class=w>{overall_w}W</span><span class=d>{overall_d}D</span><span class=l>{overall_l}L</span>
      <span style="color:var(--muted)">· fail {overall_f}</span>
      <span style="color:var(--muted)">· net {overall_w - overall_l:+d}</span>
    </div>
"""]

    for (lref, oref, label), g in sorted(groups.items()):
        total = g["w"] + g["d"] + g["l"]
        parts.append(f"""    <div class=tally>
      <span class=label>{label or '—'}</span>
      <span class=meta>{lref[:7]} vs {oref[:7]}</span>
      <span class=w>{g['w']}W</span><span class=d>{g['d']}D</span><span class=l>{g['l']}L</span>
      <span style="color:var(--muted)">·n={total} ·net {g['w'] - g['l']:+d}</span>
    </div>
""")

    parts.append("""  </div>
</header>
<main>
""")

    verdict_badges = {
        "OURS_WIN": ('<span class="badge loss">LEADER LOSES</span>',
                     '<span class="badge win">OURS WINS</span>'),
        "LEADER_WIN": ('<span class="badge win">LEADER WINS</span>',
                       '<span class="badge loss">OURS LOSES</span>'),
        "DRAW": ('<span class="badge draw">DRAW</span>',
                 '<span class="badge draw">DRAW</span>'),
        "JUDGE_FAIL": ('<span class="badge fail">FAIL</span>',
                       '<span class="badge fail">FAIL</span>'),
    }

    def strip_html(rel_paths: list[str | None] | None) -> str:
        if not rel_paths:
            return '<div class="strip">' + ''.join(
                '<div class="missing">—</div>' for _ in range(8)) + '</div>'
        tiles = []
        for rp in rel_paths[:8]:
            if rp:
                tiles.append(f'<img src="{rp}">')
            else:
                tiles.append('<div class="missing">×</div>')
        return '<div class="strip">' + ''.join(tiles) + '</div>'

    for r in rows:
        v = r.get("judge", {}).get("verdict", "JUDGE_FAIL")
        leader_badge, ours_badge = verdict_badges.get(v, ("", ""))

        votes_html = ""
        for i, vote in enumerate(r.get("judge", {}).get("votes", [])):
            if vote is None:
                votes_html += f'<span style="color:var(--danger)">v{i+1}: fail</span>'
            else:
                votes_html += f'<span>v{i+1}: L={vote[0]} O={vote[1]}</span>'

        lm = r.get("leader_meta", {})
        om = r.get("ours_meta", {})
        ls = f"{lm['best_score']:.2f}" if isinstance(lm.get('best_score'), (int, float)) else "—"
        os_ = f"{om['best_score']:.2f}" if isinstance(om.get('best_score'), (int, float)) else "—"

        ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r.get('ts', 0)))
        lref_short = r.get('leader_ref', '?')[:7]
        oref_short = r.get('ours_ref', '?')[:7]
        label = r.get('label', '')

        parts.append(f"""
<div class=duel>
  <div class=duel-head>
    <div>
      <div class=stem>{r['stem']}</div>
      <div class=ts>{ts_str}</div>
    </div>
    <div class=badges>
      <span class=config-tag>{label or 'unlabeled'} · {lref_short} → {oref_short}</span>
      <div class=votes>{votes_html}</div>
      <span class="badge {'win' if v == 'OURS_WIN' else 'loss' if v == 'LEADER_WIN' else 'draw' if v == 'DRAW' else 'fail'}">{v.replace('_',' ')}</span>
    </div>
  </div>
  <div class=three-col>
    <div class=col>
      <div class=label2><span>PROMPT</span><span class=num>reference</span></div>
      <img class=main src="{r.get('ref') or ''}">
    </div>
    <div class=col>
      <div class=label2><span>LEADER · {lref_short}</span><span class=num>critic {ls} · penalty {r.get('judge',{}).get('leader','—')}</span></div>
      <img class=main src="{r.get('leader_main') or ''}">
    </div>
    <div class=col>
      <div class=label2><span>OURS · {oref_short}{(' · '+label) if label else ''}</span><span class=num>critic {os_} · penalty {r.get('judge',{}).get('ours','—')}</span></div>
      <img class=main src="{r.get('ours_main') or ''}">
    </div>
  </div>
  <div class=strip-label>LEADER · 8-view sweep {leader_badge}</div>
  {strip_html(r.get('leader_views'))}
  <div class=strip-label>OURS · 8-view sweep {ours_badge}</div>
  {strip_html(r.get('ours_views'))}
</div>
""")

    parts.append("</main>")
    DASH_HTML_PATH.write_text("\n".join(parts))
    log.info(f"dashboard → {DASH_HTML_PATH}  ({len(rows)} rows total, {DASH_HTML_PATH.stat().st_size//1024} KB)")


# ---------- Main ----------

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--leader-ref", default="5dc2dab")
    p.add_argument("--ours-ref", default="HEAD")
    p.add_argument("--label", default="", help="free-form tag for this variant (e.g. 'R9-prompts')")
    p.add_argument("--judge-url", required=True)
    p.add_argument("--judge-model", default="zai-org/GLM-4.6V-Flash")
    p.add_argument("--judge-key", default="local")
    p.add_argument("--regenerate-only", action="store_true",
                   help="don't run any new comparisons; just rebuild the HTML from runs.jsonl")
    args = p.parse_args()

    DASH_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    if args.regenerate_only:
        write_dashboard()
        return

    seed = args.seed if args.seed != -1 else random.randint(0, 1 << 30)
    random.seed(seed)
    log.info(f"seed={seed} label={args.label!r}")

    pairs_all = load_r8_prompts()
    log.info(f"R{ROUND_N} prompts: {len(pairs_all)}")
    pairs = random.sample(pairs_all, args.n)
    log.info(f"picked: {[s[:12] for s, _ in pairs]}")

    leader_wt = ensure_worktree(args.leader_ref)
    ours_wt = ensure_worktree(args.ours_ref)

    # Resolve refs to actual commit hashes (so "HEAD" gets a stable label)
    def resolve_ref(ref: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", ref], text=True).strip()[:12]
        except Exception:
            return ref
    leader_full = resolve_ref(args.leader_ref)
    ours_full = resolve_ref(args.ours_ref)

    run_id = uuid.uuid4().hex[:8]

    for stem, url in pairs:
        log.info(f"=== {stem[:14]} ===")
        ref_bytes = download_ref(stem, url)
        ref_path = Path("/tmp/r8_refs") / f"{stem}.png"
        stem_dir = ASSET_DIR / f"{stem[:14]}__{run_id}"
        stem_dir.mkdir(parents=True, exist_ok=True)

        (leader_js, leader_meta), (ours_js, ours_meta) = await asyncio.gather(
            run_one_subprocess(leader_wt, stem, ref_path),
            run_one_subprocess(ours_wt, stem, ref_path),
        )

        leader_main_b, ours_main_b, leader_views_b, ours_views_b = await asyncio.gather(
            render_mod.render_front(leader_js) if leader_js else asyncio.sleep(0, result=None),
            render_mod.render_front(ours_js) if ours_js else asyncio.sleep(0, result=None),
            render_mod.render_multi_view(leader_js, n=8, img_size=256) if leader_js else asyncio.sleep(0, result=[]),
            render_mod.render_multi_view(ours_js, n=8, img_size=256) if ours_js else asyncio.sleep(0, result=[]),
        )

        if leader_main_b and ours_main_b:
            client_judge = AsyncOpenAI(base_url=args.judge_url, api_key=args.judge_key, timeout=180)
            judge = await judge_3vote(client_judge, args.judge_model, ref_bytes, leader_main_b, ours_main_b)
        else:
            judge = {"verdict": "JUDGE_FAIL", "leader": None, "ours": None, "votes": []}

        ref_rel = save_asset(ref_bytes, stem_dir / "ref.png")
        leader_main_rel = save_asset(leader_main_b, stem_dir / "leader_main.png")
        ours_main_rel = save_asset(ours_main_b, stem_dir / "ours_main.png")
        leader_views_rel = [save_asset(v, stem_dir / f"leader_v{i}.png") for i, v in enumerate(leader_views_b or [])]
        ours_views_rel = [save_asset(v, stem_dir / f"ours_v{i}.png") for i, v in enumerate(ours_views_b or [])]

        row = {
            "run_id": run_id, "ts": int(time.time()), "stem": stem,
            "leader_ref": leader_full, "ours_ref": ours_full, "label": args.label,
            "ref": ref_rel, "leader_main": leader_main_rel, "ours_main": ours_main_rel,
            "leader_views": leader_views_rel, "ours_views": ours_views_rel,
            "leader_meta": leader_meta, "ours_meta": ours_meta, "judge": judge,
        }
        append_run_row(row)
        log.info(f"  verdict: {judge['verdict']}  leader={judge.get('leader')}  ours={judge.get('ours')}")
        # Refresh dashboard after every stem so the user can watch progress.
        try:
            write_dashboard()
        except Exception as e:
            log.warning(f"  dashboard refresh failed: {type(e).__name__}: {e}")

    write_dashboard()


if __name__ == "__main__":
    asyncio.run(main())
