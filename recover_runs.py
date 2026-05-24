"""Best-effort recovery: rebuild dashboard rows from surviving /tmp/gen_*.js
files. Uses the same renderer + judge as compare_r8.py.

Pairs are detected by stem: any stem that has BOTH a /tmp/gen_alo2_fork_5dc2dab_<stem>.js
AND a /tmp/gen_alo2_fork_HEAD_<stem>.js gets recovered with label 'recovered'.

Run with the judge tunnel up (localhost:8003).
"""
from __future__ import annotations
import argparse
import asyncio
import glob
import json
import logging
import os
import re
import sys
import time
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
log = logging.getLogger("recover")

import render as render_mod
from openai import AsyncOpenAI

from compare_r8 import (
    DASH_DIR, ASSET_DIR, RUNS_LOG, judge_3vote, save_asset,
    append_run_row, write_dashboard, download_ref, load_r8_prompts,
)


async def classify_simple(client, model, ref_bytes):
    """Lightweight classifier — best-effort, returns dict with 'subject' or empty."""
    import base64
    b64 = "data:image/png;base64," + base64.b64encode(ref_bytes).decode()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": b64}},
                {"type": "text", "text":
                 'Output JSON only: {"subject":"<2-5 words>","category":"<one of vehicle, multi_subject, furniture, pottery, creature, machine, single_other>"}'},
            ]}],
            max_tokens=512, temperature=0.0, seed=42,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        log.warning(f"classify fail: {e}")
    return {}


async def recover_one(stem: str, leader_js: str, ours_js: str, url: str,
                       judge_client, judge_model: str):
    log.info(f"=== {stem[:14]} ===")
    try:
        ref_bytes = download_ref(stem, url)
    except Exception as e:
        log.warning(f"  ref download fail: {e}")
        return None
    cls = await classify_simple(judge_client, judge_model, ref_bytes)
    log.info(f"  subject: {cls.get('subject','?')}")

    leader_main, ours_main, leader_views, ours_views = await asyncio.gather(
        render_mod.render_front(leader_js),
        render_mod.render_front(ours_js),
        render_mod.render_multi_view(leader_js, n=8, img_size=256),
        render_mod.render_multi_view(ours_js, n=8, img_size=256),
    )
    log.info(f"  rendered: leader={'OK' if leader_main else 'FAIL'} "
             f"ours={'OK' if ours_main else 'FAIL'}")

    if leader_main and ours_main:
        judge = await judge_3vote(judge_client, judge_model, ref_bytes,
                                   leader_main, ours_main)
    else:
        judge = {"verdict": "JUDGE_FAIL", "leader": None, "ours": None, "votes": []}

    run_id = uuid.uuid4().hex[:8]
    stem_dir = ASSET_DIR / f"{stem[:14]}__{run_id}"
    stem_dir.mkdir(parents=True, exist_ok=True)

    def _save(b, name):
        if not b:
            return None
        (stem_dir / name).write_bytes(b)
        return str((stem_dir / name).relative_to(DASH_DIR))

    (stem_dir / "leader.js").write_text(leader_js)
    (stem_dir / "ours.js").write_text(ours_js)
    row = {
        "run_id": run_id, "ts": int(time.time()), "stem": stem,
        "leader_ref": "5dc2dab", "ours_ref": "HEAD", "label": "recovered",
        "ref": _save(ref_bytes, "ref.png"),
        "leader_main": _save(leader_main, "leader_main.png"),
        "ours_main": _save(ours_main, "ours_main.png"),
        "leader_views": [_save(v, f"leader_v{i}.png")
                         for i, v in enumerate(leader_views or [])],
        "ours_views": [_save(v, f"ours_v{i}.png")
                       for i, v in enumerate(ours_views or [])],
        "leader_meta": {"status": "recovered"},
        "ours_meta": {"status": "recovered"},
        "judge": judge,
        "classification": cls,
    }
    append_run_row(row)
    log.info(f"  verdict: {judge['verdict']} L={judge.get('leader')} O={judge.get('ours')}")
    return row


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge-url", default="http://localhost:8003/v1")
    p.add_argument("--judge-model", default="zai-org/GLM-4.6V-Flash")
    p.add_argument("--judge-key", default="local")
    p.add_argument("--leader-prefix", default="5dc2dab")
    p.add_argument("--ours-prefix", default="HEAD")
    args = p.parse_args()

    # Pair surviving .js files by stem.
    leader_files = glob.glob(f"/tmp/gen_alo2_fork_{args.leader_prefix}_*.js")
    ours_files = glob.glob(f"/tmp/gen_alo2_fork_{args.ours_prefix}_*.js")
    leader_by_stem = {}
    for f in leader_files:
        m = re.match(rf"/tmp/gen_alo2_fork_{re.escape(args.leader_prefix)}_(.+)\.js", f)
        if m:
            leader_by_stem[m.group(1)] = f
    ours_by_stem = {}
    for f in ours_files:
        m = re.match(rf"/tmp/gen_alo2_fork_{re.escape(args.ours_prefix)}_(.+)\.js", f)
        if m:
            ours_by_stem[m.group(1)] = f
    paired = sorted(set(leader_by_stem) & set(ours_by_stem))
    log.info(f"paired stems: {len(paired)}")

    # Look up URL for each stem from R8 prompts.
    r8 = dict(load_r8_prompts())  # stem → url

    judge_client = AsyncOpenAI(base_url=args.judge_url, api_key=args.judge_key, timeout=180)

    for i, stem in enumerate(paired, 1):
        log.info(f"({i}/{len(paired)}) recovering")
        url = r8.get(stem)
        if not url:
            log.warning(f"  no R8 url for {stem[:14]}, skipping")
            continue
        leader_js = Path(leader_by_stem[stem]).read_text()
        ours_js = Path(ours_by_stem[stem]).read_text()
        await recover_one(stem, leader_js, ours_js, url, judge_client, args.judge_model)
        write_dashboard()


if __name__ == "__main__":
    asyncio.run(main())
