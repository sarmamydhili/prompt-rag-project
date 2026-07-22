#!/usr/bin/env python3
"""
Submit a prepared generation batch JSONL to the xAI Batch API.

Uses SDK batch.add (chat objects) so unsupported models fail loudly.
Requires XAI_API_KEY (project .env) and xai-sdk.

Note: grok-3-latest is NOT batch-supported. Use --model grok-4 or grok-4.3.

Examples:
  .venv/bin/python pipeline/generation_pipeline/submit_generation_batch.py \\
    pipeline/generation_pipeline/generation_batches/generation_batch_20260722_190323.jsonl \\
    --name ap_cybersecurity_mcq --model grok-4

  .venv/bin/python pipeline/generation_pipeline/submit_generation_batch.py \\
    --status batch_...
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv(project_root / ".env")

# Models confirmed working with xAI Batch API (as of 2026-07).
KNOWN_BATCH_MODELS = ("grok-4", "grok-4.3")


def _client():
    if not os.getenv("XAI_API_KEY"):
        raise SystemExit("XAI_API_KEY is not set. Add it to .env or export it.")
    try:
        from xai_sdk import Client
    except ImportError as exc:
        raise SystemExit(
            "xai-sdk is not installed. Run: .venv/bin/pip install xai-sdk"
        ) from exc
    return Client()


def _sidecar_path(jsonl_path: Path) -> Path:
    return jsonl_path.with_name(jsonl_path.stem + "_xai_batch.json")


def _manifest_path(jsonl_path: Path) -> Path | None:
    candidate = jsonl_path.with_name(jsonl_path.stem + "_manifest.json")
    return candidate if candidate.exists() else None


def _load_jsonl(jsonl_path: Path) -> list:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {i}: {e}") from e
    if not rows:
        raise SystemExit(f"No requests found in {jsonl_path}")
    return rows


def _warn_if_model_risky(model: str) -> None:
    if model in KNOWN_BATCH_MODELS:
        return
    print(
        f"WARNING: '{model}' may not be supported by xAI Batch API. "
        f"Known-good examples: {', '.join(KNOWN_BATCH_MODELS)}. "
        f"grok-3-latest is rejected (empty batch / INVALID_ARGUMENT)."
    )


def submit_batch(
    jsonl_path: Path,
    batch_name: str,
    model_override: str | None = None,
) -> dict:
    if not jsonl_path.is_file():
        raise SystemExit(f"JSONL not found: {jsonl_path}")
    if jsonl_path.suffix != ".jsonl":
        raise SystemExit(f"Expected a .jsonl file, got: {jsonl_path}")

    from xai_sdk.chat import system, user

    rows = _load_jsonl(jsonl_path)
    models_in_file = {((r.get("body") or {}).get("model")) for r in rows}
    effective_model = model_override or next(iter(models_in_file), None)
    if model_override:
        print(f"Model override: {model_override} (file had {sorted(models_in_file)})")
    else:
        print(f"Models in file: {sorted(models_in_file)}")
    _warn_if_model_risky(effective_model or "")

    client = _client()
    print(f"Creating batch '{batch_name}' ...")
    batch = client.batch.create(batch_name)

    chats = []
    for row in rows:
        body = row.get("body") or {}
        custom_id = row.get("custom_id")
        if not custom_id:
            raise SystemExit("Each JSONL row needs a custom_id")
        model = model_override or body.get("model")
        if not model:
            raise SystemExit(f"No model for custom_id={custom_id}")
        temperature = body.get("temperature", 0.3)
        messages = body.get("messages") or []

        chat = client.chat.create(
            model=model,
            temperature=temperature,
            batch_request_id=custom_id,
        )
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "system":
                chat.append(system(content))
            elif role == "user":
                chat.append(user(content))
            else:
                raise SystemExit(
                    f"Unsupported role '{role}' in custom_id={custom_id}"
                )
        chats.append(chat)

    print(f"Adding {len(chats)} requests to {batch.batch_id} ...")
    client.batch.add(batch_id=batch.batch_id, batch_requests=chats)

    # Confirm counts landed
    refreshed = client.batch.get(batch_id=batch.batch_id)
    state = getattr(refreshed, "state", None)
    num_requests = getattr(state, "num_requests", None) if state else None
    print(f"batch state num_requests={num_requests}")

    record = {
        "file_id": None,
        "batch_id": batch.batch_id,
        "batch_name": batch_name,
        "source_jsonl": str(jsonl_path),
        "model": effective_model,
        "num_requests": num_requests,
        "submit_mode": "batch.add",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest = _manifest_path(jsonl_path)
    if manifest:
        record["manifest"] = str(manifest)

    sidecar = _sidecar_path(jsonl_path)
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"batch_id={batch.batch_id}")
    print(f"saved={sidecar}")
    if not num_requests:
        print(
            "WARNING: num_requests is still 0. Check Console; "
            "unsupported models often produce empty batches."
        )
    return record


def print_batch_status(batch_id: str) -> None:
    client = _client()
    batch = client.batch.get(batch_id=batch_id)
    state = getattr(batch, "state", None)
    print(f"batch_id: {batch.batch_id}")
    print(f"name: {getattr(batch, 'name', None)}")
    print(f"input_file_id: {getattr(batch, 'input_file_id', None) or '(none)'}")
    if state is None:
        print("state: (unavailable)")
        return
    for attr in (
        "num_requests",
        "num_pending",
        "num_success",
        "num_error",
        "num_cancelled",
    ):
        print(f"  {attr}: {getattr(state, attr, None)}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit a generation JSONL to xAI Batch API (via batch.add)"
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        help="Path to generation_batch_*.jsonl from prepare_generation_batch.py",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Batch name (default: derived from JSONL filename)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override model in every request (recommended: grok-4 or grok-4.3). "
            "grok-3-latest is not batch-supported."
        ),
    )
    parser.add_argument(
        "--status",
        metavar="BATCH_ID",
        help="Print status for an existing xAI batch id (no upload)",
    )
    args = parser.parse_args()

    if args.status:
        print_batch_status(args.status)
        return

    if not args.jsonl:
        parser.error("jsonl path is required unless --status is used")

    jsonl_path = Path(args.jsonl).expanduser().resolve()
    batch_name = args.name or jsonl_path.stem
    submit_batch(jsonl_path, batch_name, model_override=args.model)


if __name__ == "__main__":
    main()
