"""Run alo2's FULL pipeline (ensemble + critic + patcher loop) via OpenRouter.

Matches the production behavior:
  1. Ensemble: 6 coder calls with varied seed/temperature in parallel.
  2. Render each candidate.
  3. Critic each rendered candidate, get overall_score.
  4. Pick highest overall_score.
  5. Critic-patcher loop, max_iter=2: if stop=False, ask patcher to fix issues.
  6. Final JS goes to duel against leader (with 3-vote averaged judge).

Cost ~10x the simplified test (~$1.50 per prompt instead of $0.15) but reflects
the real production setup.
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
log = logging.getLogger("full_pipeline")

# alo2 imports
from llm.session_store import SessionStore
from modules.scene_coder.agent import SceneCoderAgent
from modules.critic.agent import CriticAgent
from modules.qwen_edit import edit as qwen_edit
from openai import AsyncOpenAI

# Local render/judge
import render as render_mod
import judge

OR_BASE = "https://openrouter.ai/api/v1"
OR_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "qwen/qwen3.5-397b-a17b"

REFS = ROOT.parent / "competition_db/refs"
LEADER_R7 = ROOT.parent / "competition_db/leader_r7"
LEADER_R5 = ROOT.parent / "competition_db/leader_r5"

ENSEMBLE_SIZE = 6
ENSEMBLE_BASE_TEMP = 0.3   # temperature for non-zero candidates
MAX_CRITIC_ITERS = 2
STOP_SCORE = 0.85   # alo2's score_threshold (higher is better; >=0.85 means stop)


def _build_agents(client) -> tuple[SceneCoderAgent, CriticAgent, SessionStore]:
    store = SessionStore()
    coder = SceneCoderAgent(client=client, model=MODEL, session_store=store,
                            temperature=0.0, seed=42, max_tokens=8192,
                            backend="openrouter", total_stages=6)
    critic = CriticAgent(client=client, model=MODEL,
                         max_tokens=4096, seed=42,
                         reasoning_effort="medium", ensemble_size=1,
                         backend="openrouter", total_stages=6)
    return coder, critic, store


async def _gen_candidate(coder: SceneCoderAgent, *, task_id: str,
                         image_bytes: bytes, k: int) -> str | None:
    try:
        return await coder.code(
            task_id=task_id, image_bytes=image_bytes, image_mime="image/png",
            candidate_id=k,
            seed_override=42 + k * 7919,
            temperature_override=0.0 if k == 0 else ENSEMBLE_BASE_TEMP,
        )
    except Exception as e:
        log.warning(f"  coder k={k} failed: {type(e).__name__}: {str(e)[:120]}")
        return None


async def _critique(critic: CriticAgent, *, task_id: str, image_bytes: bytes,
                    render: bytes, js: str):
    try:
        return await critic.critique(
            task_id=task_id, image_bytes=image_bytes, image_mime="image/png",
            render_png=render,
            artifact_context={"kind": "coder_v1", "osd": None, "js_code": js},
        )
    except Exception as e:
        log.warning(f"  critic failed: {type(e).__name__}: {str(e)[:120]}")
        return None


async def run_one(stem: str, source_dir: Path) -> dict:
    ref_path = REFS / f"{stem}.png"
    leader_path = source_dir / f"{stem}.js"
    if not ref_path.exists() or not leader_path.exists():
        return {"stem": stem, "result": "MISSING"}

    ref = ref_path.read_bytes()
    leader_js = leader_path.read_text()
    t0 = time.time()

    # 0. preprocess via Qwen-Image-Edit
    os.environ["QWEN_EDIT_ENABLED"] = "1"
    img = await qwen_edit(ref)
    log.info(f"[{stem[:12]}] preprocess done in {time.time()-t0:.1f}s")

    client = AsyncOpenAI(base_url=OR_BASE, api_key=OR_KEY, timeout=240)
    coder, critic, session_store = _build_agents(client)

    # 1. Ensemble: 6 candidates in parallel. Each gets its own task_id so the
    # session_store can later route patcher calls back to the winning session.
    t1 = time.time()
    cand_task_ids = [f"{stem}-k{k}" for k in range(ENSEMBLE_SIZE)]
    cands_js_full = await asyncio.gather(*[
        _gen_candidate(coder, task_id=cand_task_ids[k], image_bytes=img, k=k)
        for k in range(ENSEMBLE_SIZE)
    ])
    pairs = [(cand_task_ids[k], c) for k, c in enumerate(cands_js_full) if c]
    log.info(f"[{stem[:12]}] ensemble: {len(pairs)}/{ENSEMBLE_SIZE} non-null ({time.time()-t1:.1f}s)")
    if not pairs:
        return {"stem": stem, "result": "CODER_FAIL", "dt": time.time() - t0}

    # 2. Render each
    renders = await asyncio.gather(*[render_mod.render_front(c) for _, c in pairs])
    survivors = [(tid, c, r) for (tid, c), r in zip(pairs, renders) if r is not None]
    log.info(f"[{stem[:12]}] render: {len(survivors)}/{len(pairs)} survived")
    if not survivors:
        return {"stem": stem, "result": "RENDER_FAIL", "dt": time.time() - t0}

    # 3. Critic each, pick best by overall_score
    reports = await asyncio.gather(*[
        _critique(critic, task_id=f"{tid}-crit", image_bytes=img, render=r, js=c)
        for (tid, c, r) in survivors
    ])
    scored = [(tid, c, r, rep) for (tid, c, r), rep in zip(survivors, reports)
              if rep is not None]
    if not scored:
        best_task_id, best_js, best_render, best_k = survivors[0][0], survivors[0][1], survivors[0][2], 0
        best_report = None
    else:
        scored.sort(key=lambda x: -x[3].overall_score)
        best_task_id, best_js, best_render, best_report = scored[0]
        # Recover the candidate index (k) from the task_id suffix to find its
        # session in the store (actor is "coder#kN" for k>0, "coder" for k=0).
        try:
            best_k = int(best_task_id.rsplit("-k", 1)[1])
        except Exception:
            best_k = 0
        winning_actor = f"coder#k{best_k}" if best_k > 0 else "coder"
        # Move the winning candidate's session to the default "coder" key so
        # code_critic_repair can find it.
        if best_k > 0:
            session_store.rename_actor(best_task_id, winning_actor, "coder")
        log.info(f"[{stem[:12]}] critic best: {best_report.overall_score:.3f} stop={best_report.stop} (task={best_task_id} k={best_k})")

    # 4. Patcher loop
    iters = 0
    while iters < MAX_CRITIC_ITERS and best_report is not None and not best_report.stop \
            and best_report.overall_score < STOP_SCORE:
        try:
            patched_js = await coder.code_critic_repair(
                task_id=best_task_id,    # reuse the winning candidate's session
                image_bytes=img, image_mime="image/png",
                render_png=best_render,
                overall_score=best_report.overall_score,
                issues=best_report.issues,
                matching_aspects=best_report.matching_aspects,
                osd=None,
            )
        except Exception as e:
            log.warning(f"  patcher iter {iters}: {type(e).__name__}: {str(e)[:120]}")
            break
        new_render = await render_mod.render_front(patched_js)
        if new_render is None:
            log.info(f"[{stem[:12]}] iter {iters} render failed, keep prev")
            iters += 1
            continue
        new_report = await _critique(critic, task_id=f"{stem}-iter{iters}",
                                     image_bytes=img, render=new_render, js=patched_js)
        if new_report is None:
            iters += 1
            continue
        log.info(f"[{stem[:12]}] iter {iters}: {new_report.overall_score:.3f} stop={new_report.stop}")
        if new_report.overall_score > best_report.overall_score:
            best_js, best_render, best_report = patched_js, new_render, new_report
        iters += 1

    # 5. Duel against leader (3-vote averaged judge from our judge.py)
    leader_render = await render_mod.render_front(leader_js)
    if best_render is None or leader_render is None:
        return {"stem": stem, "result": "RENDER_FAIL", "dt": time.time() - t0}
    pa, pb = await judge.duel(ref, best_render, leader_render, seed=42, votes=3)
    verdict = "WIN" if pa < pb else ("LOSS" if pa > pb else "DRAW")
    log.info(f"[{stem[:12]}] ours={pa} leader={pb} → {verdict} ({time.time()-t0:.0f}s)")
    return {"stem": stem, "result": verdict, "ours": pa, "leader": pb,
            "dt": time.time() - t0,
            "best_score": (best_report.overall_score if best_report else None)}


async def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--source", choices=["r7", "r5"], default="r7")
    p.add_argument("--out", default="/tmp/full_pipeline_results.json")
    args = p.parse_args()

    pool = LEADER_R7 if args.source == "r7" else LEADER_R5
    stems = sorted(p.stem for p in pool.glob("*.js"))[:args.n]
    log.info(f"running FULL pipeline on {len(stems)} {args.source} prompts")

    out_path = Path(args.out)
    cached = {r["stem"]: r for r in json.loads(out_path.read_text())} if out_path.exists() else {}
    results = []
    for i, stem in enumerate(stems, 1):
        if stem in cached and cached[stem].get("result") not in (None, "CODER_FAIL"):
            log.info(f"[{i}/{len(stems)}] {stem[:12]} cached → {cached[stem]['result']}")
            results.append(cached[stem]); continue
        log.info(f"[{i}/{len(stems)}] {stem[:12]} starting")
        r = await run_one(stem, pool)
        results.append(r)
        out_path.write_text(json.dumps(results, indent=2))

    w = sum(1 for r in results if r.get("result") == "WIN")
    d = sum(1 for r in results if r.get("result") == "DRAW")
    l_ = sum(1 for r in results if r.get("result") == "LOSS")
    f_ = sum(1 for r in results if r.get("result") in ("MISSING", "CODER_FAIL", "RENDER_FAIL"))
    log.info(f"=== FULL PIPELINE SUMMARY === W:{w} D:{d} L:{l_} F:{f_} (total {len(results)})")


if __name__ == "__main__":
    asyncio.run(main())
