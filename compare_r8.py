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
from modules.scene_coder import prompts as _prompts
from modules.critic.agent import CriticAgent
from openai import AsyncOpenAI
import render as render_mod
import base64, re

OR_BASE = "https://openrouter.ai/api/v1"
OR_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "qwen/qwen3.5-397b-a17b"

ENSEMBLE = 6
MAX_ITER = 2
STOP_SCORE = 0.85


# ---- GLM judge bracket (mirrors validator selection method) ----
_GLM_SYS = ("You are a specialized 3D model evaluation system.\\n"
            "Analyze visual quality and prompt adherence with expert precision.\\n"
            "Always respond with valid JSON only.")
_GLM_USR = ("You see two 3D models rendered from slightly left of the front.\\n"
            "The reference image shows the target object.\\n\\n"
            "Which model is a more faithful 3D reproduction of the reference?\\n\\n"
            "Penalty 0-10:\\n"
            "0 = Perfect match\\n3 = Minor issues\\n5 = Moderate issues\\n"
            "7 = Major issues\\n10 = Completely wrong\\n\\n"
            'Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}')
_GLM_P  = re.compile(r'"penalty_1"\\s*:\\s*(\\d+).*?"penalty_2"\\s*:\\s*(\\d+)', re.DOTALL)
_GLM_P2 = re.compile(r'penalty_?1[^0-9]+(\\d+).*?penalty_?2[^0-9]+(\\d+)', re.DOTALL | re.IGNORECASE)


def _b64url(b: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(b).decode()}"


async def _glm_duel_one(client, model, ref, a, b, seed, swap):
    A, B = (b, a) if swap else (a, b)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _GLM_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": "Reference image (target object):"},
                    {"type": "image_url", "image_url": {"url": _b64url(ref)}},
                    {"type": "text", "text": "3D model 1:"},
                    {"type": "image_url", "image_url": {"url": _b64url(A)}},
                    {"type": "text", "text": "3D model 2:"},
                    {"type": "image_url", "image_url": {"url": _b64url(B)}},
                    {"type": "text", "text": _GLM_USR},
                ]},
            ],
            max_tokens=2048, temperature=0.0, seed=seed,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>\\s*", "", text, flags=re.DOTALL).strip()
        m = _GLM_P.search(text) or _GLM_P2.search(text)
        if not m:
            return None
        p1, p2 = int(m.group(1)), int(m.group(2))
        return (p2, p1) if swap else (p1, p2)
    except Exception:
        return None


async def glm_duel_3vote(client, model, ref, a, b, base_seed=42):
    votes = await asyncio.gather(*[
        _glm_duel_one(client, model, ref, a, b, base_seed + i * 13, bool(i & 1))
        for i in range(3)
    ])
    valid = [v for v in votes if v is not None]
    if not valid:
        return None  # judge failure → caller falls back
    pa = sum(v[0] for v in valid) / len(valid)
    pb = sum(v[1] for v in valid) / len(valid)
    return pa, pb  # (penalty_a, penalty_b)


async def glm_bracket_pick(client, model, ref, renders, log=None):
    """Single-elimination tournament. `renders` is list of PNG bytes; returns
    winning index. Each match is a 3-vote duel. Tied → keep first."""
    indices = list(range(len(renders)))
    round_idx = 0
    while len(indices) > 1:
        round_idx += 1
        pairs = []
        byes = []
        for i in range(0, len(indices), 2):
            if i + 1 < len(indices):
                pairs.append((indices[i], indices[i + 1]))
            else:
                byes.append(indices[i])

        async def _match(a, b):
            r = await glm_duel_3vote(client, model, ref, renders[a], renders[b],
                                      base_seed=42 + round_idx * 1000 + a * 31 + b)
            if r is None:
                return a  # judge fail → keep first
            pa, pb = r
            return a if pa <= pb else b

        winners = await asyncio.gather(*[_match(a, b) for a, b in pairs])
        if log:
            log.info(f"  bracket R{round_idx}: pairs={pairs} byes={byes} → "
                     f"winners={winners}")
        indices = list(winners) + byes
    return indices[0]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--out-js", required=True)
    p.add_argument("--out-meta", required=True)
    p.add_argument("--categories", default="",
                   help="comma-separated category tags from classification; if set, "
                   "the system prompt is pruned to the matching handbooks/few-shot.")
    p.add_argument("--judge-url", default="",
                   help="OpenAI-compatible base URL for GLM-4.6V-Flash. If set, "
                   "candidate selection uses a 3-vote GLM bracket instead of "
                   "Qwen critic scores (matches the validator's selection method).")
    p.add_argument("--judge-model", default="zai-org/GLM-4.6V-Flash")
    p.add_argument("--judge-key", default="local")
    p.add_argument("--inventory-file", default="",
                   help="path to JSON file produced by GLM pre-inventory; if set, "
                   "is prepended to the coder user message so the model starts "
                   "from a structured subject list with spatial distribution hints.")
    args = p.parse_args()

    img = Path(args.ref).read_bytes()
    client = AsyncOpenAI(base_url=OR_BASE, api_key=OR_KEY, timeout=240)
    store = SessionStore()
    # Optional system-prompt pruning by category. Monkey-patch the module's
    # CODER_SYSTEM_PROMPT before constructing the agent so the agent picks
    # up the leaner prompt.
    if args.categories.strip():
        cats = [c.strip() for c in args.categories.split(",") if c.strip()]
        # Older worktrees (e.g. the leader-baseline checkout) may not have
        # build_system_prompt — fall back silently and use the full prompt.
        if hasattr(_prompts, "build_system_prompt"):
            pruned = _prompts.build_system_prompt(cats)
            _prompts.CODER_SYSTEM_PROMPT = pruned
            from modules.scene_coder import agent as _agent_mod
            _agent_mod.CODER_SYSTEM_PROMPT = pruned
            log.info(f"  prompt pruned for categories {cats}: {len(pruned)} chars")
        else:
            log.info(f"  worktree lacks build_system_prompt; using full prompt")

    # Prepend GLM-derived structured inventory to the user message templates
    # so the model starts from accurate spatial distribution hints.
    if args.inventory_file and Path(args.inventory_file).exists():
        inv_raw = Path(args.inventory_file).read_text()
        inventory_block = (
            "Pre-computed inventory (from a vision pass on the same reference, "
            "use this as the authoritative subject list and placement plan — do "
            "NOT silently drop any subject or change `placement` unless the "
            "reference clearly contradicts):\\n"
            "```json\\n" + inv_raw + "\\n```\\n\\n"
        )
        from modules.scene_coder import agent as _agent_mod
        for attr in ("CODER_USER_TEMPLATE_FRESH", "CODER_USER_TEMPLATE_OSD"):
            old = getattr(_prompts, attr, None)
            if old:
                new = inventory_block + old
                setattr(_prompts, attr, new)
                setattr(_agent_mod, attr, new)
        log.info(f"  inventory prepended ({len(inv_raw)} chars)")
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
    selection_method = "critic"
    if not survivors:
        best_js = pairs[0][1]
        best_score = None
        best_render = None
        best_report = None
        best_tid = pairs[0][0]
    else:
        async def _crit(t, c, r):
            try:
                return await critic.critique(task_id=f"{t}-crit", image_bytes=img,
                                              image_mime="image/png", render_png=r,
                                              artifact_context={"kind":"coder_v1","osd":None,"js_code":c})
            except Exception as e:
                log.warning(f"  critic fail: {type(e).__name__}")
                return None

        # SELECTION: prefer GLM bracket (matches validator) when judge URL
        # is supplied; fall back to Qwen critic sort otherwise.
        if args.judge_url.strip() and len(survivors) >= 2:
            from openai import AsyncOpenAI as _AOC
            judge_cli = _AOC(base_url=args.judge_url, api_key=args.judge_key, timeout=180)
            try:
                ref_bytes = img
                winner_idx = await glm_bracket_pick(
                    judge_cli, args.judge_model, ref_bytes,
                    [s[2] for s in survivors], log=log,
                )
                best_tid, best_js, best_render = survivors[winner_idx]
                # One critic call on the winner just to drive patcher repair
                # (we still want detailed issues/matching_aspects feedback).
                best_report = await _crit(best_tid, best_js, best_render)
                best_score = best_report.overall_score if best_report else None
                selection_method = "glm_bracket"
            except Exception as e:
                log.warning(f"  glm bracket failed: {e}; falling back to critic")
                args_judge_url = ""
        if selection_method != "glm_bracket":
            reports = await asyncio.gather(*[_crit(*s) for s in survivors])
            scored = [(t, c, r, rep) for (t, c, r), rep in zip(survivors, reports) if rep]
            if not scored:
                best_tid, best_js, best_render = survivors[0]
                best_report, best_score = None, None
            else:
                scored.sort(key=lambda x: -x[3].overall_score)
                best_tid, best_js, best_render, best_report = scored[0]
                best_score = best_report.overall_score

        # Move winning candidate's session to default "coder" key so patcher
        # can find it (mirror of original logic).
        best_k = int(best_tid.rsplit("-k", 1)[1])
        if best_k > 0:
            store.rename_actor(best_tid, f"coder#k{best_k}", "coder")

        for it in range(MAX_ITER):
            if best_report is None:
                break
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
            # Adopt patched iff it ALSO wins a GLM duel against the current best
            # (when bracket is enabled). Otherwise fall back to critic-score
            # comparison.
            adopt = False
            if args.judge_url.strip():
                try:
                    from openai import AsyncOpenAI as _AOC2
                    judge_cli2 = _AOC2(base_url=args.judge_url, api_key=args.judge_key, timeout=180)
                    duel = await glm_duel_3vote(judge_cli2, args.judge_model, img,
                                                 best_render, new_r,
                                                 base_seed=42 + it * 7919)
                    if duel is not None:
                        pa, pb = duel  # a=current best, b=patched
                        adopt = (pb < pa - 0.25)  # require small clear edge
                except Exception:
                    adopt = (new_rep.overall_score > best_report.overall_score)
            else:
                adopt = (new_rep.overall_score > best_report.overall_score)
            if adopt:
                best_js, best_render, best_report = patched, new_r, new_rep
                best_score = new_rep.overall_score

    Path(args.out_js).write_text(best_js)
    Path(args.out_meta).write_text(json.dumps({
        "status": "ok", "best_score": best_score, "dt": time.time() - t0,
        "n_ensemble": len(pairs), "n_survivors": len(survivors),
        "selection_method": selection_method,
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


# R8 audit candidates ordered by direct H2H strength (winner_c is strongest).
WINNER_POOLS = {
    "winner_a": Path("/tmp/winner_a_r8"),  # 5HgGDgMf — first audit candidate (vs leader +0.12)
    "winner_b": Path("/tmp/winner_b_r8"),  # 5G4Z9uJLPN — beat winner_a by 0.03
    "winner_c": Path("/tmp/winner_c_r8"),  # 5GZSyVqH  — beat winner_b by 0.09 (strongest H2H)
    "winner":   Path("/tmp/winner_r8"),    # legacy alias → winner_a
}


def load_winner_js(stem: str, which: str = "winner_c") -> tuple[str | None, dict]:
    """Read a pre-downloaded winner submission as the 'leader' arm.
    No subprocess, no generation cost — just a file read.

    `which` selects which audit-candidate's submission to use:
      - 'winner_a' (5HgGDgMf...) : first audit, vs-leader margin 0.12
      - 'winner_b' (5G4Z9uJLPN...): second audit
      - 'winner_c' (5GZSyVqH...) : strongest by H2H — default for R10 onwards
      - 'winner'                  : alias for winner_a (legacy)
    """
    pool = WINNER_POOLS.get(which, WINNER_POOLS["winner_c"])
    p = pool / f"{stem}.js"
    if not p.exists():
        return None, {"status": "winner_missing", "dt": 0.0}
    return p.read_text(), {"status": which, "dt": 0.0, "best_score": None,
                            "n_ensemble": 1, "n_survivors": 1}


async def run_one_subprocess(work_dir: Path, stem: str, ref_path: Path,
                              categories: list[str] | None = None,
                              judge_url: str | None = None,
                              judge_model: str = "zai-org/GLM-4.6V-Flash",
                              judge_key: str = "local",
                              inventory_json: str | None = None) -> tuple[str | None, dict]:
    out_js = Path(f"/tmp/gen_{work_dir.name}_{stem}.js")
    out_meta = Path(f"/tmp/gen_{work_dir.name}_{stem}.json")
    cmd = [
        sys.executable, str(work_dir / "_gen_one.py"),
        "--stem", stem, "--ref", str(ref_path),
        "--out-js", str(out_js), "--out-meta", str(out_meta),
    ]
    if categories:
        cmd += ["--categories", ",".join(categories)]
    if judge_url:
        cmd += ["--judge-url", judge_url,
                "--judge-model", judge_model,
                "--judge-key", judge_key]
    if inventory_json:
        # Write to a temp file (CLI arg length and shell-escaping risk for big
        # JSON); subprocess reads from disk.
        inv_path = Path(f"/tmp/inv_{work_dir.name}_{stem}.json")
        inv_path.write_text(inventory_json)
        cmd += ["--inventory-file", str(inv_path)]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
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


DASHBOARD_CSS = """
:root {
  --bg:#0d0d10; --panel:#17171b; --panel2:#1d1d22; --border:#2a2a32;
  --fg:#e4e4ea; --muted:#8a8a92; --accent:#5cd3a5; --danger:#ff7676; --warn:#f5c25b;
}
* { box-sizing:border-box }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
       margin:0; background:var(--bg); color:var(--fg); font-size:13px; }
header { padding:12px 20px; background:var(--panel); border-bottom:1px solid var(--border);
         position:sticky; top:0; z-index:50; }
h1 { margin:0 0 6px; font-size:14px; font-weight:600; letter-spacing:.02em; }
.tally-row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; font-size:11px; }
.tally { display:inline-flex; gap:8px; padding:4px 10px; border-radius:5px;
         background:var(--panel2); border:1px solid var(--border);
         font-variant-numeric:tabular-nums; }
.tally.big { font-size:13px; padding:6px 14px; }
.tally .w { color:var(--accent); font-weight:600; }
.tally .l { color:var(--danger); font-weight:600; }
.tally .d { color:var(--warn); font-weight:600; }
.tally .label { color:var(--muted); font-size:10px; padding-right:5px;
                border-right:1px solid var(--border); margin-right:2px;
                text-transform:uppercase; letter-spacing:.05em; }
.tally .ref { color:var(--muted); font-family:monospace; font-size:10px; }
.meta { color:var(--muted); font-size:11px; }

main { padding:14px 18px; display:flex; flex-direction:column; gap:14px;
       max-width:1200px; margin:0 auto; width:100%; }

.duel { background:var(--panel); border:1px solid var(--border); border-radius:8px;
        overflow:hidden; }
.duel-top { display:grid; grid-template-columns:220px 220px 220px 1fr; gap:1px;
            background:var(--border); }
.tile { background:var(--panel2); padding:10px; display:flex; flex-direction:column; gap:6px; }
.tile-head { display:flex; justify-content:space-between; align-items:center; gap:8px;
             min-height:22px; }
.tile-head .label { font-size:10px; color:var(--muted); font-weight:500;
                    letter-spacing:.05em; text-transform:uppercase; }
.tile-head .num { color:var(--fg); font-family:monospace; font-weight:600; font-size:10px; }
.tile img.main { width:100%; aspect-ratio:1/1; object-fit:cover; display:block;
                 background:#222; border-radius:4px; cursor:zoom-in;
                 transition:filter .15s ease; }
.tile img.main:hover { filter:brightness(1.08); }
.tile .info { font-size:10px; color:var(--muted);
              display:flex; justify-content:space-between; gap:8px; }
.btn3d { background:var(--panel); color:var(--fg);
         border:1px solid var(--border); padding:3px 9px; font-size:10px;
         border-radius:4px; cursor:pointer; letter-spacing:.04em;
         text-transform:uppercase; font-weight:500; white-space:nowrap; }
.btn3d:hover { background:var(--accent); color:#000; border-color:var(--accent); }

.summary-tile { background:var(--panel2); padding:10px 12px;
                display:flex; flex-direction:column; gap:8px; }
.summary-tile .stem { color:var(--muted); font-size:9px; font-family:monospace;
                      word-break:break-all; line-height:1.3; }
.summary-tile .btn-del { margin-top:auto; align-self:flex-end;
                         background:transparent; color:var(--muted);
                         border:1px solid var(--border); padding:3px 8px;
                         font-size:10px; border-radius:4px; cursor:pointer;
                         letter-spacing:.04em; text-transform:uppercase; }
.summary-tile .btn-del:hover { background:var(--danger); color:#000;
                                border-color:var(--danger); }
.summary-tile .verdict { font-size:12px; font-weight:600; padding:4px 10px;
                         border-radius:4px; display:inline-block; align-self:flex-start;
                         text-transform:uppercase; letter-spacing:.04em; }
.summary-tile .verdict.win { background:rgba(92,211,165,.15); color:var(--accent);
                              border:1px solid rgba(92,211,165,.4); }
.summary-tile .verdict.loss { background:rgba(255,118,118,.12); color:var(--danger);
                              border:1px solid rgba(255,118,118,.35); }
.summary-tile .verdict.draw { background:rgba(245,194,91,.12); color:var(--warn);
                              border:1px solid rgba(245,194,91,.35); }
.summary-tile .verdict.fail { background:rgba(120,120,128,.18); color:var(--muted);
                              border:1px solid var(--border); }
.summary-tile .row2 { display:flex; flex-direction:column; gap:3px; font-size:10px;
                      color:var(--muted); }
.summary-tile .row2 span { background:var(--panel); padding:2px 6px; border-radius:3px;
                           border:1px solid var(--border); font-family:monospace; }
.summary-tile .config { font-size:9px; color:var(--muted); font-family:monospace;
                        letter-spacing:.02em; }

.strips { padding:8px; background:var(--panel); display:none; border-top:1px solid var(--border); }
.strips.open { display:block; }
.strip { display:grid; grid-template-columns:repeat(8,1fr); gap:3px;
         padding:5px; margin-top:5px; border:1px solid var(--border); border-radius:4px;
         background:var(--panel2); }
.strip:first-child { margin-top:0; }
.strip-label { font-size:9px; color:var(--muted); padding:5px 4px 3px;
               text-transform:uppercase; letter-spacing:.05em; }
.strip img { width:100%; aspect-ratio:1/1; object-fit:cover; background:#222;
             border-radius:2px; display:block; cursor:zoom-in; }
.strip img:hover { filter:brightness(1.1); }
.strip .missing { width:100%; aspect-ratio:1/1; background:#1a1a1f;
                  border:1px dashed var(--border); border-radius:2px;
                  display:flex; align-items:center; justify-content:center;
                  color:var(--muted); font-size:9px; }
.toggle-strips { width:100%; background:var(--panel); color:var(--muted);
                 border:none; border-top:1px solid var(--border); padding:5px;
                 font-size:10px; cursor:pointer; letter-spacing:.05em;
                 text-transform:uppercase; font-weight:500; }
.toggle-strips:hover { color:var(--fg); background:var(--panel2); }

/* modal — used for both 3D viewer and zoomed images */
#modal { position:fixed; inset:0; background:rgba(0,0,0,.9); display:none;
         z-index:100; padding:30px; align-items:center; justify-content:center; }
#modal.open { display:flex; flex-direction:column; }
#modal iframe { flex:1; width:100%; border:1px solid var(--border); border-radius:6px;
                background:#202024; min-height:0; }
#modal img.zoomed { max-width:100%; max-height:100%; border:1px solid var(--border);
                    border-radius:6px; background:#222; object-fit:contain; }
#modal .close { position:absolute; top:14px; right:14px; background:var(--panel);
                color:var(--fg); border:1px solid var(--border); padding:6px 12px;
                border-radius:4px; cursor:pointer; z-index:1; }
#modal .title { color:var(--muted); font-size:12px; margin-bottom:10px;
                font-family:monospace; align-self:flex-start; }
"""

DASHBOARD_JS = """
function _modalShow(title, mode) {
  const m = document.getElementById('modal');
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('modalFrame').style.display = (mode === '3d') ? '' : 'none';
  document.getElementById('modalImg').style.display = (mode === 'img') ? '' : 'none';
  m.classList.add('open');
}
function open3d(src, title) {
  document.getElementById('modalFrame').src = 'viewer.html?src=' + encodeURIComponent(src);
  _modalShow(title, '3d');
}
function openImg(src, title) {
  document.getElementById('modalImg').src = src;
  _modalShow(title, 'img');
}
function closeModal() {
  const m = document.getElementById('modal');
  document.getElementById('modalFrame').src = '';
  document.getElementById('modalImg').src = '';
  m.classList.remove('open');
}
function toggleStrips(id) {
  document.getElementById(id).classList.toggle('open');
}
async function deleteRow(runId, label) {
  if (!confirm(`delete this row (${label})?`)) return;
  const resp = await fetch('/delete/' + encodeURIComponent(runId), { method: 'POST' });
  if (resp.ok) {
    // soft-remove from DOM right away (snappier than full reload)
    const el = document.querySelector('[data-run-id="' + runId + '"]');
    if (el) el.remove();
  } else {
    alert('delete failed: ' + (await resp.text()));
  }
}
window.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
"""


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
<title>Duel R{ROUND_N}</title>
<style>{DASHBOARD_CSS}</style>
<header>
  <h1>Duel · R{ROUND_N} · cumulative</h1>
  <div class=tally-row>
    <div class="tally big">
      <span class=label>Overall (n={overall_total})</span>
      <span class=w>{overall_w}W</span><span class=d>{overall_d}D</span><span class=l>{overall_l}L</span>
      <span style="color:var(--muted)">· fail {overall_f}</span>
      <span style="color:var(--muted)">· net {overall_w - overall_l:+d}</span>
    </div>
"""]

    for (lref, oref, label), g in sorted(groups.items()):
        total = g["w"] + g["d"] + g["l"]
        parts.append(f"""    <div class=tally>
      <span class=label>{label or '—'}</span>
      <span class=ref>{lref[:7]} vs {oref[:7]}</span>
      <span class=w>{g['w']}W</span><span class=d>{g['d']}D</span><span class=l>{g['l']}L</span>
      <span style="color:var(--muted)">·n={total} ·net {g['w'] - g['l']:+d}</span>
    </div>
""")

    parts.append("""  </div>
</header>
<main>
""")

    def strip_html(rel_paths: list[str | None] | None) -> str:
        if not rel_paths:
            return ''.join('<div class="missing">—</div>' for _ in range(8))
        tiles = []
        for rp in rel_paths[:8]:
            if rp:
                tiles.append(f'<img src="{rp}">')
            else:
                tiles.append('<div class="missing">×</div>')
        return ''.join(tiles)

    for idx, r in enumerate(rows):
        v = r.get("judge", {}).get("verdict", "JUDGE_FAIL")
        verdict_cls = {'OURS_WIN': 'win', 'LEADER_WIN': 'loss',
                       'DRAW': 'draw', 'JUDGE_FAIL': 'fail'}[v]
        verdict_label = {'OURS_WIN': 'Ours wins', 'LEADER_WIN': 'Leader wins',
                         'DRAW': 'Draw', 'JUDGE_FAIL': 'Judge fail'}[v]

        votes_html = ''
        for i, vote in enumerate(r.get("judge", {}).get("votes", [])):
            if vote is None:
                votes_html += f'<span style="color:var(--danger)">v{i+1}: fail</span>'
            else:
                votes_html += f'<span>v{i+1}: L={vote[0]} O={vote[1]}</span>'

        lm = r.get("leader_meta", {})
        om = r.get("ours_meta", {})
        ls = f"{lm['best_score']:.2f}" if isinstance(lm.get('best_score'), (int, float)) else "—"
        os_ = f"{om['best_score']:.2f}" if isinstance(om.get('best_score'), (int, float)) else "—"

        ts_str = time.strftime('%H:%M:%S', time.localtime(r.get('ts', 0)))
        lref_short = r.get('leader_ref', '?')[:7]
        oref_short = r.get('ours_ref', '?')[:7]
        label = r.get('label', '')

        # 3D-viewer src paths: assets/<stem>__<run>/leader.js etc.
        # Only show 3D button when the .js file actually exists on disk
        # (recovered rows sometimes lacked the source).
        leader_main = r.get('leader_main') or ''
        ours_main = r.get('ours_main') or ''
        leader_js_src = ''
        ours_js_src = ''
        if leader_main:
            cand = leader_main.rsplit('/', 1)[0] + '/leader.js'
            if (DASH_DIR / cand).exists():
                leader_js_src = cand
        if ours_main:
            cand = ours_main.rsplit('/', 1)[0] + '/ours.js'
            if (DASH_DIR / cand).exists():
                ours_js_src = cand

        cls = r.get('classification') or {}
        cls_info = f"{cls.get('subject','')} · {cls.get('category','?')}" if cls.get('subject') else ''

        strips_id = f"strips-{idx}"
        stem_short = r['stem'][:10]
        ref_path = r.get('ref') or ''
        leader_btn = (f'<button class=btn3d onclick="open3d(&quot;{leader_js_src}&quot;,'
                      f'&quot;leader · {stem_short}&quot;)">3D</button>') if leader_js_src else ''
        ours_btn = (f'<button class=btn3d onclick="open3d(&quot;{ours_js_src}&quot;,'
                    f'&quot;ours · {stem_short}&quot;)">3D</button>') if ours_js_src else ''
        ref_btn = (f'<button class=btn3d onclick="openImg(&quot;{ref_path}&quot;,'
                   f'&quot;reference · {stem_short}&quot;)">Zoom</button>') if ref_path else ''
        leader_zoom_btn = (f'<button class=btn3d onclick="openImg(&quot;{leader_main}&quot;,'
                           f'&quot;leader render · {stem_short}&quot;)">Zoom</button>') if leader_main else ''
        ours_zoom_btn = (f'<button class=btn3d onclick="openImg(&quot;{ours_main}&quot;,'
                         f'&quot;ours render · {stem_short}&quot;)">Zoom</button>') if ours_main else ''
        leader_penalty = r.get('judge', {}).get('leader', '—')
        ours_penalty = r.get('judge', {}).get('ours', '—')

        # Tile head right column: each tile gets a small action row separate
        # from the title so buttons never overlap text.
        ref_onclick = f"onclick=\"openImg('{ref_path}','reference · {stem_short}')\"" if ref_path else ""
        leader_img_onclick = f"onclick=\"openImg('{leader_main}','leader render · {stem_short}')\"" if leader_main else ""
        ours_img_onclick = f"onclick=\"openImg('{ours_main}','ours render · {stem_short}')\"" if ours_main else ""

        run_id = r.get("run_id", "")
        del_btn = (f'<button class=btn-del onclick="deleteRow(&quot;{run_id}&quot;,'
                   f'&quot;{stem_short}&quot;)" title="delete this row">delete</button>')

        parts.append(f"""
<div class=duel data-run-id="{run_id}">
  <div class=duel-top>
    <div class=tile>
      <div class=tile-head>
        <span class=label>Reference</span>
        {ref_btn}
      </div>
      <img class=main src="{ref_path}" {ref_onclick}>
      <div class=info><span>{cls_info}</span><span>{ts_str}</span></div>
    </div>
    <div class=tile>
      <div class=tile-head>
        <span class=label>Leader · {lref_short}</span>
        {leader_btn}
      </div>
      <img class=main src="{leader_main}" {leader_img_onclick}>
      <div class=info><span>critic {ls}</span><span>penalty {leader_penalty}</span><span>{lm.get('dt', 0):.0f}s</span></div>
    </div>
    <div class=tile>
      <div class=tile-head>
        <span class=label>Ours · {oref_short}</span>
        {ours_btn}
      </div>
      <img class=main src="{ours_main}" {ours_img_onclick}>
      <div class=info><span>critic {os_}</span><span>penalty {ours_penalty}</span><span>{om.get('dt', 0):.0f}s</span></div>
    </div>
    <div class="tile summary-tile">
      <span class="verdict {verdict_cls}">{verdict_label}</span>
      <div class=config>{label or 'unlabeled'} · {lref_short} → {oref_short}</div>
      <div class=row2>{votes_html}</div>
      <div class=stem>{r['stem']}</div>
      {del_btn}
    </div>
  </div>
  <button class=toggle-strips onclick="toggleStrips('{strips_id}')">8-view sweep ▾</button>
  <div class=strips id="{strips_id}">
    <div class=strip-label>Leader</div>
    <div class=strip>{strip_html(r.get('leader_views'))}</div>
    <div class=strip-label>Ours</div>
    <div class=strip>{strip_html(r.get('ours_views'))}</div>
  </div>
</div>""")

    parts.append("""</main>
<div id="modal" onclick="if(event.target.id==='modal')closeModal()">
  <button class=close onclick="closeModal()">close ✕  (Esc)</button>
  <div class=title id="modalTitle"></div>
  <iframe id="modalFrame" style="display:none"></iframe>
  <img id="modalImg" class=zoomed style="display:none">
</div>
<script>""" + DASHBOARD_JS + "</script>")

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
        # Save .js sources so the dashboard's 3D viewer can import them.
        if leader_js:
            (stem_dir / "leader.js").write_text(leader_js)
        if ours_js:
            (stem_dir / "ours.js").write_text(ours_js)

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
