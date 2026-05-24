"""Walk one R8 stem through every pipeline step with full visibility.

Each step prints exactly what's happening and saves intermediate artifacts to
/tmp/dashboard/debug/<stem>/. Final HTML page at
/tmp/dashboard/debug/<stem>/index.html shows everything together.

Usage:
  python debug_one.py [--stem <hash>] [--seed N]
      [--leader-ref 5dc2dab] [--ours-ref HEAD] [--label "R9-prompts"]
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
log = logging.getLogger("debug_one")

import render as render_mod  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

# Reuse helpers from compare_r8
sys.path.insert(0, str(ROOT))
from compare_r8 import (  # noqa: E402
    load_r8_prompts, download_ref, ensure_worktree, run_one_subprocess,
    judge_one, judge_3vote, write_dashboard, _b64url, append_run_row,
    DASH_DIR, ASSET_DIR, ROUND_N,
)

DEBUG_DIR = DASH_DIR / "debug"


# ---------- Step: classify reference image ----------

CLASSIFY_SYSTEM = (
    "You categorize 3D reference images for a procedural-modeling benchmark.\n"
    "Answer with valid JSON only — no prose outside the JSON."
)

CLASSIFY_USER = """Look at the reference image and classify it.

Categories (pick ONE primary):
- vehicle     : car, truck, motorcycle, plane, drone, bicycle, boat, scooter
- multi_subject : two or more visually distinct objects (cup-on-saucer, phone-with-charger, figure-with-tool, set of items)
- furniture   : chair, table, sofa, bed, shelf
- pottery     : vase, bottle, vessel, ceramic
- creature    : animal, character, figurine
- machine     : device, appliance, tool, instrument
- single_other: single object that doesn't fit above

Also estimate visual complexity (low / medium / high) and whether the
subject is the kind of thing that often goes wrong in procedural Three.js
(e.g. cars, multi-subject, organic shapes).

Output JSON:
{
  "category": "<one of the labels above>",
  "subject": "<2-5 word description of what the object is>",
  "n_distinct_objects": <int>,
  "complexity": "low | medium | high",
  "challenge": "<one short sentence about likely procedural-modeling pitfalls>"
}
"""


INVENTORY_USER = """Produce a STRUCTURED inventory of this reference image
for procedural 3D modeling. Be exhaustive — include every visually distinct
element, including contents inside transparent or open containers (liquid,
floating pieces, food, items).

Return JSON only:
{
  "subjects": [
    {
      "name": "<short lowercase noun, underscores ok>",
      "count": <int — 1 if single, ~N if many of the same>,
      "color": "<dominant color>",
      "approx_shape": "<one word: cylinder, sphere, cube, cone, vase, leaf, almond, ring, irregular, ...>",
      "size_relative": "<small | medium | large | dominant>",
      "placement": "<one of: standalone | sits_on_base | held_by_X | inside_X | embedded_in_volume_of_X | clustered_on_top_of_X | distributed_throughout_X | distributed_on_surface_of_X | surrounding_X | lined_up_in_X>"
    },
    ...
  ],
  "scene_layout": "<one sentence describing overall spatial relationship>"
}

Critical rules:
- For drinks/yogurt/soup with floating berries or chunks: those berries are
  separate subjects with placement = "embedded_in_volume_of_<container>" or
  "distributed_throughout_<container>", NOT "clustered_on_top_of_<container>".
- For decorated cakes, do not collapse all toppings into one subject — count
  groups separately if the colors / shapes differ.
- For vehicles, include wheels (count=4), headlights (count=2), etc.
- Include the container as its own subject.
- PATTERN vs SUBJECT — surface patterns (rainbow stripes on a blanket,
  polka dots on fabric, logo painted on a bottle, plaid, gradient) are
  NOT separate subjects. They are flat color bands on the underlying
  surface. List only the underlying surface as a subject and describe
  the pattern in its `color` or `placement` field (e.g.
  "rainbow_striped_horizontal", "polka_dotted"). If the elements have
  no shadow or thickness — flush with the surface — they are patterns.
"""


async def inventory_glm(client: AsyncOpenAI, model: str, ref: bytes) -> dict:
    """Structured inventory pass. Returns dict with subjects list and layout
    string. Returns {} on parse error so downstream falls back gracefully."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You produce structured JSON only — no prose."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _b64url(ref)}},
                    {"type": "text", "text": INVENTORY_USER},
                ]},
            ],
            max_tokens=4096, temperature=0.0, seed=42,
        )
        text = resp.choices[0].message.content or ""
        raw = text
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        # Greedy JSON match — pick the outermost {} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {"raw": raw, "parse_error": "no_json"}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            txt = m.group(0).replace("'", '"')
            try:
                data = json.loads(txt)
            except Exception as e:
                return {"raw": raw, "parse_error": str(e)}
        data["raw"] = raw
        return data
    except Exception as e:
        return {"raw": "", "parse_error": f"{type(e).__name__}: {e}"}


async def classify(client: AsyncOpenAI, model: str, ref: bytes) -> dict:
    """Returns dict with category/subject/n_distinct/complexity/challenge + raw."""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": _b64url(ref)}},
                    {"type": "text", "text": CLASSIFY_USER},
                ]},
            ],
            max_tokens=2048, temperature=0.0, seed=42,
        )
        text = resp.choices[0].message.content or ""
        raw = text
        # Strip <think>...</think>
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        # Extract first JSON object
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            # Try harder for nested
            m = re.search(r"\{.*?\}", text, re.DOTALL)
        if not m:
            return {"category": "unknown", "raw": raw, "parse_error": "no_json"}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            # Try to repair common issues
            txt = m.group(0).replace("'", '"')
            try:
                data = json.loads(txt)
            except Exception:
                return {"category": "unknown", "raw": raw, "parse_error": str(e)}
        data["raw"] = raw
        return data
    except Exception as e:
        return {"category": "unknown", "raw": "", "parse_error": f"{type(e).__name__}: {e}"}


# ---------- Step: judge with full vote reasoning captured ----------

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


async def judge_one_verbose(client: AsyncOpenAI, model: str, ref: bytes, a: bytes, b: bytes,
                             seed: int, swap: bool) -> dict:
    """Like judge_one but returns full raw text + parsed result for inspection."""
    A, B = (b, a) if swap else (a, b)
    out = {"seed": seed, "swap": swap, "raw": None, "penalty_leader": None,
           "penalty_ours": None, "error": None}
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
        out["raw"] = text
        text_clean = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        m = _re_p.search(text_clean) or _re_p2.search(text_clean)
        if not m:
            out["error"] = "no_parse"
            return out
        p1, p2 = int(m.group(1)), int(m.group(2))
        p1, p2 = max(0, min(10, p1)), max(0, min(10, p2))
        # Map back to leader/ours
        if swap:
            out["penalty_leader"], out["penalty_ours"] = p2, p1
        else:
            out["penalty_leader"], out["penalty_ours"] = p1, p2
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


# ---------- Asset save ----------

def save_bytes(b: bytes | None, p: Path) -> None:
    if b is None:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)


# ---------- Debug HTML per stem ----------

def write_debug_page(stem_dir: Path, ctx: dict):
    """Render a self-contained debug page for one stem with every step's info."""
    cls = ctx["classification"]
    classification_card = f"""
<div class=card>
  <h2>STEP 1 · Reference classification</h2>
  <img src="ref.png" style="width:280px;border-radius:6px">
  <table>
    <tr><td class=k>category</td><td><b>{cls.get('category','?')}</b></td></tr>
    <tr><td class=k>subject</td><td>{cls.get('subject','—')}</td></tr>
    <tr><td class=k>n_distinct_objects</td><td>{cls.get('n_distinct_objects','—')}</td></tr>
    <tr><td class=k>complexity</td><td>{cls.get('complexity','—')}</td></tr>
    <tr><td class=k>challenge</td><td>{cls.get('challenge','—')}</td></tr>
  </table>
  <details><summary>raw classifier response</summary>
    <pre>{(cls.get('raw') or '')[:3000]}</pre>
  </details>
</div>
"""

    lm = ctx["leader_meta"]
    om = ctx["ours_meta"]
    leader_card = f"""
<div class=card>
  <h2>STEP 2A · Leader generation [{ctx['leader_ref'][:7]}]</h2>
  <table>
    <tr><td class=k>status</td><td>{lm.get('status','?')}</td></tr>
    <tr><td class=k>best critic score</td><td>{lm.get('best_score','—')}</td></tr>
    <tr><td class=k>ensemble survivors</td><td>{lm.get('n_survivors','?')} / {lm.get('n_ensemble','?')}</td></tr>
    <tr><td class=k>time</td><td>{lm.get('dt',0):.1f}s</td></tr>
  </table>
  <details><summary>generated .js source (first 4KB)</summary>
    <pre>{(ctx.get('leader_js') or '')[:4000]}</pre>
  </details>
</div>
"""
    ours_card = f"""
<div class=card>
  <h2>STEP 2B · Ours generation [{ctx['ours_ref'][:7]}] {('· label ' + ctx['label']) if ctx['label'] else ''}</h2>
  <table>
    <tr><td class=k>status</td><td>{om.get('status','?')}</td></tr>
    <tr><td class=k>best critic score</td><td>{om.get('best_score','—')}</td></tr>
    <tr><td class=k>ensemble survivors</td><td>{om.get('n_survivors','?')} / {om.get('n_ensemble','?')}</td></tr>
    <tr><td class=k>time</td><td>{om.get('dt',0):.1f}s</td></tr>
  </table>
  <details><summary>generated .js source (first 4KB)</summary>
    <pre>{(ctx.get('ours_js') or '')[:4000]}</pre>
  </details>
</div>
"""

    # Renders: main + 8 views in a strip
    leader_views_html = "".join(
        f'<img src="leader_v{i}.png">' if (stem_dir / f"leader_v{i}.png").exists()
        else '<div class="missing">×</div>'
        for i in range(8)
    )
    ours_views_html = "".join(
        f'<img src="ours_v{i}.png">' if (stem_dir / f"ours_v{i}.png").exists()
        else '<div class="missing">×</div>'
        for i in range(8)
    )
    render_card = f"""
<div class=card>
  <h2>STEP 3 · Renders</h2>
  <div class=three-col>
    <div><h3>PROMPT</h3><img class=main src="ref.png"></div>
    <div><h3>LEADER main</h3><img class=main src="leader_main.png"></div>
    <div><h3>OURS main</h3><img class=main src="ours_main.png"></div>
  </div>
  <h3>LEADER 8-view sweep</h3>
  <div class=strip>{leader_views_html}</div>
  <h3>OURS 8-view sweep</h3>
  <div class=strip>{ours_views_html}</div>
</div>
"""

    # Judge votes
    judge = ctx["judge"]
    votes_html = ""
    for i, v in enumerate(judge.get("votes_full", []), 1):
        if v.get("error"):
            votes_html += f"""
<div class=vote-card style="border-left:3px solid var(--danger)">
  <div class=vote-head>vote {i} · seed={v.get('seed')} swap={v.get('swap')} · <b style="color:var(--danger)">ERROR: {v['error']}</b></div>
  <details><summary>raw response</summary><pre>{(v.get('raw') or '')[:3000]}</pre></details>
</div>"""
        else:
            pl = v["penalty_leader"]
            po = v["penalty_ours"]
            verdict_color = "var(--accent)" if po < pl else "var(--danger)" if pl < po else "var(--warn)"
            votes_html += f"""
<div class=vote-card>
  <div class=vote-head>vote {i} · seed={v['seed']} swap={v['swap']} · <b style="color:{verdict_color}">L={pl} O={po}</b></div>
  <details><summary>raw response</summary><pre>{(v.get('raw') or '')[:3000]}</pre></details>
</div>"""

    verdict = judge.get("verdict", "JUDGE_FAIL")
    verdict_color = {"OURS_WIN": "var(--accent)", "LEADER_WIN": "var(--danger)",
                     "DRAW": "var(--warn)", "JUDGE_FAIL": "var(--muted)"}[verdict]
    judge_card = f"""
<div class=card>
  <h2>STEP 4 · 3-vote judge</h2>
  <div class=verdict-line>
    <span style="font-size:20px;color:{verdict_color};font-weight:600">{verdict.replace('_',' ')}</span>
    &nbsp;·&nbsp;
    <span class=muted>avg penalty:</span>
    leader = <b>{judge.get('leader','—')}</b>,
    ours = <b>{judge.get('ours','—')}</b>
  </div>
  {votes_html}
</div>
"""

    html = f"""<!doctype html>
<meta charset=utf-8>
<title>debug · {ctx['stem'][:14]}</title>
<style>
:root {{
  --bg:#0d0d10; --panel:#17171b; --panel2:#1d1d22; --border:#2a2a32;
  --fg:#e4e4ea; --muted:#8a8a92; --accent:#5cd3a5; --danger:#ff7676; --warn:#f5c25b;
}}
* {{ box-sizing:border-box }}
body {{ font-family:-apple-system,BlinkMacSystemFont,sans-serif; margin:0;
       background:var(--bg); color:var(--fg); }}
header {{ padding:14px 24px; background:var(--panel); border-bottom:1px solid var(--border); }}
h1 {{ margin:0; font-size:15px; font-weight:600; }}
header .meta {{ color:var(--muted); font-size:12px; margin-top:6px; font-family:monospace; }}
main {{ padding:18px 24px; display:flex; flex-direction:column; gap:18px; max-width:1400px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:18px; }}
.card h2 {{ margin:0 0 14px; font-size:13px; color:var(--muted); letter-spacing:.08em;
           text-transform:uppercase; font-weight:600; }}
.card h3 {{ margin:14px 0 8px; font-size:11px; color:var(--muted); letter-spacing:.05em;
           text-transform:uppercase; }}
table {{ border-collapse:collapse; width:100%; max-width:600px; }}
td {{ padding:5px 12px; border-bottom:1px solid var(--border); font-size:13px; }}
td.k {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em;
       width:200px; }}
pre {{ background:var(--panel2); padding:10px; border-radius:4px; font-size:11px;
      overflow:auto; max-height:300px; border:1px solid var(--border); margin:6px 0; }}
details {{ margin-top:8px; }}
details summary {{ cursor:pointer; color:var(--muted); font-size:12px; padding:4px 0; }}
.three-col {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }}
.three-col img.main {{ width:100%; aspect-ratio:1/1; object-fit:cover;
                       background:#222; border-radius:6px; }}
.strip {{ display:grid; grid-template-columns:repeat(8,1fr); gap:4px;
         background:var(--panel2); padding:6px; border-radius:6px;
         border:1px solid var(--border); }}
.strip img {{ width:100%; aspect-ratio:1/1; object-fit:cover; background:#222;
             border-radius:3px; }}
.strip .missing {{ width:100%; aspect-ratio:1/1; background:#1a1a1f; border:1px dashed var(--border);
                  display:flex; align-items:center; justify-content:center; color:var(--muted); }}
.vote-card {{ background:var(--panel2); border:1px solid var(--border); border-radius:6px;
             padding:10px 14px; margin-bottom:8px; border-left:3px solid var(--border); }}
.vote-head {{ font-size:13px; font-family:monospace; }}
.muted {{ color:var(--muted); }}
.verdict-line {{ padding:8px 0 14px; font-size:13px; }}
</style>
<header>
  <h1>debug · {ctx['stem']}</h1>
  <div class=meta>
    {ctx['label'] or '—'} · leader {ctx['leader_ref'][:7]} → ours {ctx['ours_ref'][:7]} ·
    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ctx['ts']))}
  </div>
</header>
<main>
  {classification_card}
  {leader_card}
  {ours_card}
  {render_card}
  {judge_card}
</main>
"""
    (stem_dir / "index.html").write_text(html)


# ---------- Main: walk one stem ----------

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem", default=None)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--leader-ref", default="5dc2dab")
    p.add_argument("--ours-ref", default="HEAD")
    p.add_argument("--label", default="R9-prompts")
    p.add_argument("--judge-url", required=True)
    p.add_argument("--judge-model", default="zai-org/GLM-4.6V-Flash")
    p.add_argument("--judge-key", default="local")
    args = p.parse_args()

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # ─── Pick stem ────────────────────────────────────────────────────
    pairs_all = load_r8_prompts()
    log.info(f"R{ROUND_N} prompts: {len(pairs_all)}")
    if args.stem:
        match = [p for p in pairs_all if p[0].startswith(args.stem)]
        if not match:
            log.error(f"no R8 stem matches prefix {args.stem!r}")
            sys.exit(1)
        stem, url = match[0]
    else:
        seed = args.seed if args.seed != -1 else random.randint(0, 1 << 30)
        random.seed(seed)
        stem, url = random.choice(pairs_all)
        log.info(f"random seed={seed} → picked {stem[:14]}")

    log.info(f"=== {stem} ===")
    log.info(f"  url: {url}")

    stem_dir = DEBUG_DIR / stem[:14]
    stem_dir.mkdir(parents=True, exist_ok=True)
    ctx: dict = {"stem": stem, "label": args.label, "ts": int(time.time()),
                 "leader_ref": args.leader_ref, "ours_ref": args.ours_ref}

    # ─── STEP 1: ref + classify ──────────────────────────────────────
    log.info("STEP 1 · download ref + classify via GLM")
    ref_bytes = download_ref(stem, url)
    save_bytes(ref_bytes, stem_dir / "ref.png")
    log.info(f"  ref: {len(ref_bytes)} bytes → {stem_dir / 'ref.png'}")

    client_judge = AsyncOpenAI(base_url=args.judge_url, api_key=args.judge_key, timeout=180)
    classification = await classify(client_judge, args.judge_model, ref_bytes)
    log.info(f"  category: {classification.get('category','?')}")
    log.info(f"  subject : {classification.get('subject','—')}")
    log.info(f"  n_dist  : {classification.get('n_distinct_objects','—')}")
    log.info(f"  complex : {classification.get('complexity','—')}")
    log.info(f"  challenge: {classification.get('challenge','—')}")
    if classification.get("parse_error"):
        log.warning(f"  parse_error: {classification['parse_error']}")
        log.warning(f"  raw[:400]: {(classification.get('raw') or '')[:400]}")
    ctx["classification"] = classification

    # ─── STEP 2: generate leader + ours ──────────────────────────────
    log.info("STEP 2 · generate leader + ours in parallel (ensemble=6, max_iter=2)")
    leader_wt = ensure_worktree(args.leader_ref)
    ours_wt = ensure_worktree(args.ours_ref)
    ref_path = Path("/tmp/r8_refs") / f"{stem}.png"

    # Derive categories from classification → coder gets a pruned system
    # prompt with only the relevant handbooks + few-shot.
    cats = []
    cat = classification.get("category")
    if cat:
        cats.append(cat)
    # Also treat multi-subject as a category if the classifier saw >1
    # distinct object — keeps the contained-Y guidance live.
    try:
        if int(classification.get("n_distinct_objects", 1)) > 1:
            cats.append("multi_subject")
    except Exception:
        pass

    # Structured GLM inventory pass — extracts subjects + spatial placement.
    # Passed to ours subprocess (NOT leader) so the leader baseline runs
    # unchanged and the win comes from real lift, not from inventory help.
    log.info("  GLM inventory pass...")
    inventory = await inventory_glm(client_judge, args.judge_model, ref_bytes)
    if inventory and "subjects" in inventory:
        log.info(f"  inventory: {len(inventory['subjects'])} subjects · "
                 f"layout='{(inventory.get('scene_layout') or '')[:80]}'")
        for s in inventory["subjects"][:6]:
            log.info(f"    · {s.get('name')} × {s.get('count')} "
                     f"({s.get('placement')})")
    else:
        log.warning(f"  inventory parse failed: {inventory.get('parse_error')}")
    inv_json = None
    if inventory and "subjects" in inventory:
        inv_clean = {k: v for k, v in inventory.items() if k != "raw"}
        inv_json = json.dumps(inv_clean, indent=2, ensure_ascii=False)
    ctx["inventory"] = {k: v for k, v in inventory.items() if k != "raw"}

    t0 = time.time()
    # Pass judge URL down: ours uses GLM bracket selection (matches the
    # validator); leader baseline runner gracefully falls back to critic
    # if its worktree predates the bracket helper.
    (leader_js, leader_meta), (ours_js, ours_meta) = await asyncio.gather(
        run_one_subprocess(leader_wt, stem, ref_path, categories=cats,
                            judge_url=args.judge_url, judge_model=args.judge_model,
                            judge_key=args.judge_key),
        run_one_subprocess(ours_wt, stem, ref_path, categories=cats,
                            judge_url=args.judge_url, judge_model=args.judge_model,
                            judge_key=args.judge_key,
                            inventory_json=inv_json),
    )
    log.info(f"  generation took {time.time()-t0:.1f}s")
    log.info(f"  leader: status={leader_meta.get('status')} "
             f"best_score={leader_meta.get('best_score')} "
             f"survivors={leader_meta.get('n_survivors')}/{leader_meta.get('n_ensemble')}")
    log.info(f"  ours  : status={ours_meta.get('status')} "
             f"best_score={ours_meta.get('best_score')} "
             f"survivors={ours_meta.get('n_survivors')}/{ours_meta.get('n_ensemble')}")
    ctx["leader_meta"] = leader_meta
    ctx["ours_meta"] = ours_meta
    ctx["leader_js"] = leader_js
    ctx["ours_js"] = ours_js
    if leader_js:
        (stem_dir / "leader.js").write_text(leader_js)
    if ours_js:
        (stem_dir / "ours.js").write_text(ours_js)

    # ─── STEP 3: render ──────────────────────────────────────────────
    log.info("STEP 3 · render main + 8-view sweep for both")
    leader_main, ours_main, leader_views, ours_views = await asyncio.gather(
        render_mod.render_front(leader_js) if leader_js else asyncio.sleep(0, result=None),
        render_mod.render_front(ours_js) if ours_js else asyncio.sleep(0, result=None),
        render_mod.render_multi_view(leader_js, n=8, img_size=256) if leader_js else asyncio.sleep(0, result=[]),
        render_mod.render_multi_view(ours_js, n=8, img_size=256) if ours_js else asyncio.sleep(0, result=[]),
    )
    log.info(f"  leader_main: {'OK' if leader_main else 'FAIL'} "
             f"({len(leader_main) if leader_main else 0} bytes)")
    log.info(f"  ours_main  : {'OK' if ours_main else 'FAIL'} "
             f"({len(ours_main) if ours_main else 0} bytes)")
    log.info(f"  leader 8-view: {sum(1 for v in (leader_views or []) if v)}/8")
    log.info(f"  ours   8-view: {sum(1 for v in (ours_views or []) if v)}/8")
    save_bytes(leader_main, stem_dir / "leader_main.png")
    save_bytes(ours_main, stem_dir / "ours_main.png")
    for i, v in enumerate(leader_views or []):
        save_bytes(v, stem_dir / f"leader_v{i}.png")
    for i, v in enumerate(ours_views or []):
        save_bytes(v, stem_dir / f"ours_v{i}.png")

    # ─── STEP 4: 3-vote judge with full text capture ─────────────────
    log.info("STEP 4 · 3-vote judge (with full vote text saved)")
    if not (leader_main and ours_main):
        log.warning("  skipping: missing renders")
        votes_full = []
        judge = {"verdict": "JUDGE_FAIL", "leader": None, "ours": None,
                 "votes": [], "votes_full": []}
    else:
        votes_full = await asyncio.gather(*[
            judge_one_verbose(client_judge, args.judge_model, ref_bytes,
                              leader_main, ours_main,
                              42 + i * 13, bool(i & 1))
            for i in range(3)
        ])
        for i, v in enumerate(votes_full, 1):
            if v["error"]:
                log.warning(f"  vote {i}: ERROR {v['error']}")
            else:
                log.info(f"  vote {i}: seed={v['seed']} swap={v['swap']} "
                         f"L={v['penalty_leader']} O={v['penalty_ours']}")
        valid = [v for v in votes_full if v["error"] is None]
        if not valid:
            judge = {"verdict": "JUDGE_FAIL", "leader": None, "ours": None,
                     "votes": [], "votes_full": votes_full}
        else:
            pa = sum(v["penalty_leader"] for v in valid) / len(valid)
            pb = sum(v["penalty_ours"] for v in valid) / len(valid)
            if abs(pa - pb) < 0.5:
                v_str = "DRAW"
            elif pb < pa:
                v_str = "OURS_WIN"
            else:
                v_str = "LEADER_WIN"
            judge = {
                "verdict": v_str,
                "leader": round(pa, 2),
                "ours": round(pb, 2),
                "votes": [(v["penalty_leader"], v["penalty_ours"]) if v["error"] is None
                          else None for v in votes_full],
                "votes_full": votes_full,
                "n_valid": len(valid),
            }
    log.info(f"  AGGREGATE: {judge['verdict']}  L={judge.get('leader','—')} O={judge.get('ours','—')}")
    ctx["judge"] = judge

    # ─── Persist row + dashboard ─────────────────────────────────────
    log.info("STEP 5 · persist row + regenerate cumulative dashboard")
    run_id = uuid.uuid4().hex[:8]
    asset_dir = ASSET_DIR / f"{stem[:14]}__{run_id}"
    asset_dir.mkdir(parents=True, exist_ok=True)

    def _save_rel(b: bytes | None, name: str) -> str | None:
        if not b:
            return None
        p = asset_dir / name
        p.write_bytes(b)
        return str(p.relative_to(DASH_DIR))

    # Also save the .js sources so the dashboard 3D viewer can import them.
    if leader_js:
        (asset_dir / "leader.js").write_text(leader_js)
    if ours_js:
        (asset_dir / "ours.js").write_text(ours_js)

    row = {
        "run_id": run_id, "ts": ctx["ts"], "stem": stem,
        "leader_ref": args.leader_ref, "ours_ref": args.ours_ref, "label": args.label,
        "ref": _save_rel(ref_bytes, "ref.png"),
        "leader_main": _save_rel(leader_main, "leader_main.png"),
        "ours_main": _save_rel(ours_main, "ours_main.png"),
        "leader_views": [_save_rel(v, f"leader_v{i}.png")
                         for i, v in enumerate(leader_views or [])],
        "ours_views": [_save_rel(v, f"ours_v{i}.png")
                       for i, v in enumerate(ours_views or [])],
        "leader_meta": leader_meta, "ours_meta": ours_meta,
        "judge": {k: v for k, v in judge.items() if k != "votes_full"},
        "classification": {k: v for k, v in classification.items() if k != "raw"},
        "debug_url": f"debug/{stem[:14]}/index.html",
    }
    append_run_row(row)
    write_dashboard()

    # Debug page
    write_debug_page(stem_dir, ctx)
    log.info(f"  debug page → {stem_dir / 'index.html'}")
    log.info(f"  open: http://localhost:8081/debug/{stem[:14]}/index.html")

    log.info("DONE.")


if __name__ == "__main__":
    asyncio.run(main())
