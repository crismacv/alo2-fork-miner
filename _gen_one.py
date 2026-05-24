"""Single-stem generation runner. Used by compare_r8.py.

Reads stem, ref image path, output path from CLI. Prints "OK" on success.
Assumes pipeline_service is on sys.path (set by parent).
"""
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
    # Ensemble
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
            # Patcher loop
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
