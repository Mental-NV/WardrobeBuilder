#!/usr/bin/env python3
import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from openai import OpenAI

FIDELITY_MODELS = {"gpt-image-1", "gpt-image-1.5", "chatgpt-image-latest"}

# Optional typed exceptions (script also works without them)
try:
    from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError
    RETRYABLE_EXCEPTIONS = (RateLimitError, APIError, APITimeoutError, APIConnectionError)
except Exception:
    RETRYABLE_EXCEPTIONS = (Exception,)

@dataclass
class Job:
    name: str
    user_message: str


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} [{level}] {msg}\n"
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def info(self, msg: str) -> None:
        self._write("INFO", msg)

    def warn(self, msg: str) -> None:
        self._write("WARN", msg)

    def error(self, msg: str) -> None:
        self._write("ERROR", msg)


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_path(base_dir: Path, maybe_path: str) -> Path:
    p = Path(maybe_path)
    return p if p.is_absolute() else (base_dir / p)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_prompt(system_prompt: str, user_message: str) -> str:
    return (
        "SYSTEM INSTRUCTIONS (follow strictly):\n"
        f"{system_prompt.strip()}\n\n"
        "USER REQUEST:\n"
        f"{user_message.strip()}\n"
    )


def sanitize_filename(name: str) -> str:
    """
    Convert job name into a safe filename stem:
    - keep letters/numbers/_/-
    - replace spaces with _
    - drop other characters
    """
    s = name.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "job"


def parse_jobs(cfg: Dict[str, Any]) -> List[Job]:
    jobs_raw = cfg.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("Config must contain a non-empty 'jobs' array.")

    jobs: List[Job] = []
    for i, j in enumerate(jobs_raw):
        if not isinstance(j, dict):
            raise ValueError(f"jobs[{i}] must be an object")

        name = str(j.get("name") or f"job_{i}").strip()
        if not name:
            raise ValueError(f"jobs[{i}] invalid 'name'")

        user_message = j.get("user_message")
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError(f"jobs[{i}] missing/invalid 'user_message'")

        jobs.append(Job(name=name, user_message=user_message.strip()))

    return jobs


class StartRateLimiter:
    """
    Two-layer limiter:
      1) Semaphore caps concurrent in-flight requests.
      2) Min-interval gates request start times (smooths bursts).
    """
    def __init__(self, max_concurrent: int, min_interval_ms: int):
        self._sem = threading.Semaphore(max(1, int(max_concurrent)))
        self._min_interval = max(0.0, float(min_interval_ms) / 1000.0)
        self._lock = threading.Lock()
        self._last_start = 0.0

    def __enter__(self):
        self._sem.acquire()
        if self._min_interval > 0:
            with self._lock:
                now = time.time()
                wait = (self._last_start + self._min_interval) - now
                if wait > 0:
                    time.sleep(wait)
                self._last_start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._sem.release()
        return False


_thread_local = threading.local()


def get_client() -> OpenAI:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI()
    return _thread_local.client


def is_retryable_exception(e: Exception) -> bool:
    if RETRYABLE_EXCEPTIONS != (Exception,) and isinstance(e, RETRYABLE_EXCEPTIONS):
        return True

    msg = (str(e) or "").lower()
    transient_markers = [
        "rate limit", "429", "timeout", "timed out", "temporarily", "connection",
        "reset", "overloaded", "server error", "502", "503", "504"
    ]
    return any(m in msg for m in transient_markers)


def backoff_sleep(attempt: int, base_delay_s: float, max_delay_s: float) -> None:
    delay = min(max_delay_s, base_delay_s * (2 ** attempt))
    jitter = random.uniform(0.0, 0.25 * delay)
    time.sleep(delay + jitter)


def image_edit_with_retries(
    *,
    system_prompt: str,
    user_message: str,
    image_path: Path,
    model: str,
    size: str,
    quality: str,
    output_format: str,
    input_fidelity: Optional[str],
    limiter: StartRateLimiter,
    max_retries: int,
    base_delay_s: float,
    max_delay_s: float,
) -> bytes:
    prompt = build_prompt(system_prompt, user_message)

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            client = get_client()
            with limiter:
                req: Dict[str, Any] = {
                    "model": model,
                    "prompt": prompt,
                    "image": [image_path.open("rb")],
                    "size": size,
                    "quality": quality,
                    "output_format": output_format,
                }
                model_norm = (model or "").strip()
                if model_norm in FIDELITY_MODELS and input_fidelity:
                    req["input_fidelity"] = input_fidelity

                result = client.images.edit(**req)

            b64 = result.data[0].b64_json
            return base64.b64decode(b64)

        except Exception as e:
            last_err = e
            if attempt >= max_retries or not is_retryable_exception(e):
                break
            backoff_sleep(attempt, base_delay_s=base_delay_s, max_delay_s=max_delay_s)

    raise RuntimeError(f"Image edit failed: {last_err}") from last_err


def run_timestamp_folder(base_out: Path, name: str) -> Path:
    name = sanitize_filename(name)
    ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = base_out / name / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def print_plan(
    jobs: List[Job],
    out_root: Path,
    model: str,
    size: str,
    quality: str,
    output_format: str,
    input_image_path: Path,
    logger: Logger,
) -> None:
    header = (
        f"PLAN\n"
        f"- Input photo: {input_image_path}\n"
        f"- Output root: {out_root}\n"
        f"- Model: {model}\n"
        f"- Size: {size}\n"
        f"- Quality: {quality}\n"
        f"- Output format: {output_format}\n"
        f"- Jobs: {len(jobs)}\n"
    )
    print(header)
    logger.info(header.strip())

    ext = output_format.lower().lstrip(".")
    for i, job in enumerate(jobs, start=1):
        filename = f"{sanitize_filename(job.name)}.{ext}"
        out_path = out_root / filename
        line = f"  {i:02d}. {job.name} -> {out_path}"
        print(line)
        logger.info(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch try-on edits (inline JSON, parallel + rate-limit safe).")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--dry-run", action="store_true", help="Print plan + create log/manifest; do not call API")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY") and not args.dry_run:
        raise EnvironmentError("OPENAI_API_KEY is not set (required unless --dry-run).")

    config_path = Path(args.config).resolve()
    base_dir = Path(__file__).resolve().parent
    cfg = load_config(config_path)

    system_prompt = cfg.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("Config must include 'system_prompt' as a non-empty string.")
    system_prompt = system_prompt.strip()

    input_image_path = cfg.get("input_image_path")
    if not isinstance(input_image_path, str) or not input_image_path.strip():
        raise ValueError("Config must include 'input_image_path' as a non-empty string.")
    image_path = resolve_path(base_dir, input_image_path.strip())
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model = str(cfg.get("model", "gpt-image-1"))
    size = str(cfg.get("size", "auto"))
    quality = str(cfg.get("quality", "high"))
    output_format = str(cfg.get("output_format", "png")).lower().lstrip(".")

    input_fidelity = cfg.get("input_fidelity")
    input_fidelity = str(input_fidelity) if input_fidelity is not None else None

    name = str(cfg.get("name") or config_path.stem).strip()
    output_dir = str(cfg.get("output_dir", "out"))
    out_base = resolve_path(base_dir, output_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    out_root = run_timestamp_folder(out_base, name)

    logger = Logger(out_root / "run.log")
    logger.info(f"Started run. dry_run={args.dry_run} config={config_path}")

    parallel_cfg = cfg.get("parallel", {}) if isinstance(cfg.get("parallel", {}), dict) else {}
    max_workers = int(parallel_cfg.get("max_workers", 4))
    max_concurrent = int(parallel_cfg.get("max_concurrent_requests", 2))
    min_interval_ms = int(parallel_cfg.get("min_interval_ms", 900))
    limiter = StartRateLimiter(max_concurrent=max_concurrent, min_interval_ms=min_interval_ms)

    retry_cfg = cfg.get("retries", {}) if isinstance(cfg.get("retries", {}), dict) else {}
    max_retries = int(retry_cfg.get("max_retries", 4))
    base_delay_s = float(retry_cfg.get("base_delay_s", 1.0))
    max_delay_s = float(retry_cfg.get("max_delay_s", 20.0))

    jobs = parse_jobs(cfg)

    print_plan(
        jobs=jobs,
        out_root=out_root,
        model=model,
        size=size,
        quality=quality,
        output_format=output_format,
        input_image_path=image_path,
        logger=logger,
    )

    manifest = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": bool(args.dry_run),
        "config_path": str(config_path),
        "input_image_path": str(image_path),
        "output_root": str(out_root),
        "settings": {
            "model": model,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "input_fidelity": input_fidelity,
            "parallel": {
                "max_workers": max_workers,
                "max_concurrent_requests": max_concurrent,
                "min_interval_ms": min_interval_ms,
            },
            "retries": {
                "max_retries": max_retries,
                "base_delay_s": base_delay_s,
                "max_delay_s": max_delay_s,
            },
        },
        "results": [],
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        logger.info("Dry run complete. No API calls were made.")
        print(f"\nDry run complete. Logs and manifest created in:\n  {out_root}\n")
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(job: Job) -> Tuple[str, str, str]:
        filename = f"{sanitize_filename(job.name)}.{output_format}"
        out_path = out_root / filename
        ensure_parent_dir(out_path)

        img_bytes = image_edit_with_retries(
            system_prompt=system_prompt,
            user_message=job.user_message,
            image_path=image_path,
            model=model,
            size=size,
            quality=quality,
            output_format=output_format,
            input_fidelity=input_fidelity,
            limiter=limiter,
            max_retries=max_retries,
            base_delay_s=base_delay_s,
            max_delay_s=max_delay_s,
        )
        out_path.write_bytes(img_bytes)
        return (job.name, filename, str(out_path))

    failures = 0
    logger.info(
        f"Executing {len(jobs)} jobs with max_workers={max_workers}, "
        f"max_concurrent={max_concurrent}, min_interval_ms={min_interval_ms}"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, job): job for job in jobs}
        done_count = 0

        for fut in as_completed(futures):
            job = futures[fut]
            done_count += 1
            try:
                name, filename, out_path = fut.result()
                msg = f"[{done_count}/{len(jobs)}] OK  {name} -> {out_path}"
                print(msg)
                logger.info(msg)
                manifest["results"].append({
                    "name": job.name,
                    "status": "ok",
                    "output_file": filename,
                    "output_path": out_path,
                })
            except Exception as e:
                failures += 1
                msg = f"[{done_count}/{len(jobs)}] ERR {job.name}: {e}"
                print(msg, file=sys.stderr)
                logger.error(msg)
                manifest["results"].append({
                    "name": job.name,
                    "status": "error",
                    "error": str(e),
                })

            (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Finished run. failures={failures} output_root={out_root}")

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
