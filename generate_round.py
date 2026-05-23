"""Per-round generation and CDN upload.

Watches the 404-active-competition repo for `rounds/<N>/prompts.txt`, runs the
full pipeline on each prompt, and uploads every produced `.js` to the R2
bucket. Designed to run once the round opens; safe to re-run (idempotent on
both generation cache and upload).

Layout in R2:
  banana/rounds/<N>/<stem>.js   (the public miner output)

Public URL: https://pub-<pubhash>.r2.dev/rounds/<N>/<stem>.js
(set R2_PUBLIC_URL to override).
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import time
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
log = logging.getLogger("gen_round")

import boto3

# Reuse the full pipeline runner
from test_full_pipeline import run_one, MODEL, OR_BASE, OR_KEY


COMP_REPO = Path("/tmp/404-active-competition")
BUCKET = os.environ.get("R2_BUCKET", "banana")
LOCAL_OUT = Path(os.environ.get("ROUND_OUT_DIR", "/tmp/round_out"))


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_WRITE_URL"],
        aws_access_key_id=os.environ["R2_WRITE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_WRITE_SECRET_KEY"],
        region_name="auto",
    )


def _refresh_repo() -> None:
    if COMP_REPO.exists():
        os.system(f"cd {COMP_REPO} && git pull --quiet")
    else:
        os.system(f"git clone --depth 1 https://github.com/404-Repo/404-active-competition.git {COMP_REPO}")


def _stems_for_round(n: int) -> list[str]:
    p = COMP_REPO / "rounds" / str(n) / "prompts.txt"
    if not p.exists():
        return []
    stems = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # URL like https://sn12domain.org/<stem>.png
        if "/" in line:
            stem = line.rsplit("/", 1)[1].rsplit(".", 1)[0]
        else:
            stem = line.rsplit(".", 1)[0]
        stems.append(stem)
    return stems


def _download_prompt_image(stem: str, dest: Path) -> bytes | None:
    """Download the actual reference PNG for a stem from the public CDN."""
    if dest.exists():
        return dest.read_bytes()
    import urllib.request
    url = f"https://sn12domain.org/{stem}.png"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        dest.write_bytes(data)
        return data
    except Exception as e:
        log.warning(f"download {stem[:12]}: {e}")
        return None


async def _generate_and_upload(round_n: int, stem: str, s3) -> dict:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    ref_local = LOCAL_OUT / f"{stem}.png"
    js_local = LOCAL_OUT / f"{stem}.js"
    key = f"rounds/{round_n}/{stem}.js"

    # Skip if already uploaded
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        log.info(f"  {stem[:12]} already in R2 — skipping")
        return {"stem": stem, "status": "skipped", "key": key}
    except Exception:
        pass

    img = _download_prompt_image(stem, ref_local)
    if img is None:
        return {"stem": stem, "status": "ref_missing"}

    # Patch run_one to point REFS at our local output dir
    import test_full_pipeline as tfp
    tfp.REFS = LOCAL_OUT  # generic ref lookup uses REFS / f"{stem}.png"

    # Run the full pipeline (returns dict). The function looks up a leader_js too,
    # which we don't have for R8 — pass a stem that doesn't exist in leader_r5/r7.
    # Trick: temporarily monkeypatch the leader lookup to return our_js as leader
    # → no comparison, just our generation.
    # Actually simpler: call coder directly via a minimal helper.
    from llm.session_store import SessionStore
    from modules.scene_coder.agent import SceneCoderAgent
    from modules.critic.agent import CriticAgent
    from modules.qwen_edit import edit as qwen_edit
    from openai import AsyncOpenAI
    import render as render_mod

    client = AsyncOpenAI(base_url=OR_BASE, api_key=OR_KEY, timeout=240)
    store = SessionStore()
    coder = SceneCoderAgent(client=client, model=MODEL, session_store=store,
                            temperature=0.0, seed=42, max_tokens=8192,
                            backend="openrouter", total_stages=6)
    critic = CriticAgent(client=client, model=MODEL, max_tokens=4096, seed=42,
                         reasoning_effort="medium", ensemble_size=1,
                         backend="openrouter", total_stages=6)

    os.environ["QWEN_EDIT_ENABLED"] = "1"
    t0 = time.time()
    pre_img = await qwen_edit(img)
    log.info(f"  {stem[:12]} preprocess {time.time()-t0:.1f}s")

    # Ensemble + critic-pick + patcher loop (mirrors test_full_pipeline.run_one)
    ENSEMBLE = 6
    cand_task_ids = [f"{stem}-k{k}" for k in range(ENSEMBLE)]
    async def _gen(k):
        try:
            return await coder.code(task_id=cand_task_ids[k], image_bytes=pre_img,
                                    image_mime="image/png", candidate_id=k,
                                    seed_override=42 + k * 7919,
                                    temperature_override=0.0 if k == 0 else 0.3)
        except Exception as e:
            log.warning(f"  k{k} coder fail: {type(e).__name__}: {str(e)[:80]}")
            return None
    cands = await asyncio.gather(*[_gen(k) for k in range(ENSEMBLE)])
    pairs = [(cand_task_ids[k], c) for k, c in enumerate(cands) if c]
    if not pairs:
        return {"stem": stem, "status": "coder_fail"}

    renders = await asyncio.gather(*[render_mod.render_front(c) for _, c in pairs])
    survivors = [(tid, c, r) for (tid, c), r in zip(pairs, renders) if r is not None]
    if not survivors:
        # fallback: upload first candidate's JS anyway (even if render failed locally —
        # validator's renderer may still accept it)
        best_js = pairs[0][1]
    else:
        async def _crit(tid, c, r):
            try:
                return await critic.critique(task_id=f"{tid}-crit", image_bytes=pre_img,
                                              image_mime="image/png", render_png=r,
                                              artifact_context={"kind": "coder_v1",
                                                                "osd": None, "js_code": c})
            except Exception as e:
                log.warning(f"  critic fail: {type(e).__name__}: {str(e)[:80]}")
                return None
        reports = await asyncio.gather(*[_crit(*s) for s in survivors])
        scored = [(tid, c, r, rep) for (tid, c, r), rep in zip(survivors, reports) if rep]
        if not scored:
            best_js = survivors[0][1]
        else:
            scored.sort(key=lambda x: -x[3].overall_score)
            best_task_id, best_js, best_render, best_report = scored[0]
            log.info(f"  {stem[:12]} best_score={best_report.overall_score:.3f}")

            # Patcher loop (max 2 iters)
            best_k = int(best_task_id.rsplit("-k", 1)[1])
            if best_k > 0:
                store.rename_actor(best_task_id, f"coder#k{best_k}", "coder")
            for it in range(2):
                if best_report.stop or best_report.overall_score >= 0.85:
                    break
                try:
                    patched = await coder.code_critic_repair(
                        task_id=best_task_id, image_bytes=pre_img, image_mime="image/png",
                        render_png=best_render,
                        overall_score=best_report.overall_score,
                        issues=best_report.issues,
                        matching_aspects=best_report.matching_aspects,
                        osd=None,
                    )
                except Exception as e:
                    log.warning(f"  patcher iter {it}: {type(e).__name__}: {str(e)[:80]}")
                    break
                new_r = await render_mod.render_front(patched)
                if new_r is None:
                    continue
                new_rep = await _crit(best_task_id, patched, new_r)
                if new_rep is None:
                    continue
                if new_rep.overall_score > best_report.overall_score:
                    best_js, best_render, best_report = patched, new_r, new_rep

    # Save + upload
    js_local.write_text(best_js)
    s3.put_object(Bucket=BUCKET, Key=key, Body=best_js.encode("utf-8"),
                  ContentType="application/javascript")
    log.info(f"  {stem[:12]} uploaded to R2 ({len(best_js)} bytes, {time.time()-t0:.0f}s)")
    return {"stem": stem, "status": "ok", "key": key, "bytes": len(best_js)}


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N stems (for testing)")
    p.add_argument("--watch", action="store_true",
                   help="Loop until prompts.txt exists then process")
    args = p.parse_args()

    s3 = _s3()
    while True:
        _refresh_repo()
        stems = _stems_for_round(args.round)
        if stems:
            log.info(f"R{args.round}: {len(stems)} prompts to process")
            break
        if not args.watch:
            log.warning(f"R{args.round} prompts.txt not present yet; exit")
            return
        log.info(f"R{args.round} prompts.txt not present; sleeping 60s...")
        await asyncio.sleep(60)

    if args.limit:
        stems = stems[:args.limit]
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    # Process in batches of 4 to avoid OpenRouter rate-limit
    BATCH = 4
    results = []
    for i in range(0, len(stems), BATCH):
        chunk = stems[i:i+BATCH]
        log.info(f"--- batch {i//BATCH + 1}: stems {i+1}..{min(i+BATCH, len(stems))} ---")
        batch_results = await asyncio.gather(*[
            _generate_and_upload(args.round, s, s3) for s in chunk
        ])
        results.extend(batch_results)
        Path("/tmp/round_results.json").write_text(json.dumps(results, indent=2))

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    fail = len(results) - ok - skipped
    log.info(f"=== R{args.round} SUMMARY === ok:{ok} skipped:{skipped} fail:{fail}")


if __name__ == "__main__":
    asyncio.run(main())
