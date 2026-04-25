#!/usr/bin/env python3
"""Create/update a Hugging Face Docker Space and upload this repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from huggingface_hub import HfApi

PLACEHOLDER_MARKERS = ("your-", "your_", "changeme", "example")


def is_real_value(value: str | None) -> bool:
    if value is None:
        return False
    trimmed = value.strip()
    if not trimmed:
        return False
    lowered = trimmed.lower()
    return not any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/update a HF Docker Space and upload the current repository."
    )
    parser.add_argument(
        "--space-id",
        default=os.getenv("HF_SPACE_ID", ""),
        help="Space id in the form '<owner>/<space-name>' (or set HF_SPACE_ID).",
    )
    parser.add_argument(
        "--space-url",
        default=os.getenv("HF_SPACE_PAGE_URL", ""),
        help=(
            "Optional Space page URL like "
            "'https://huggingface.co/spaces/<owner>/<space-name>' "
            "(or set HF_SPACE_PAGE_URL)."
        ),
    )
    parser.add_argument(
        "--repo-dir",
        default=".",
        help="Repository folder to upload (default: current directory).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private (default: public).",
    )
    parser.add_argument(
        "--skip-secrets",
        action="store_true",
        help="Skip setting API_BASE_URL, MODEL_NAME, HF_TOKEN as Space secrets.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN", ""),
        help="HF access token (or set HF_TOKEN).",
    )
    return parser.parse_args()


def read_env_file(repo_dir: Path) -> dict[str, str]:
    env_path = repo_dir / ".env"
    if not env_path.exists():
        return {}

    parsed: dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def pick_var(name: str, env_values: dict[str, str]) -> str | None:
    direct = os.getenv(name)
    if is_real_value(direct):
        return direct
    fallback = env_values.get(name)
    if is_real_value(fallback):
        return fallback
    return None


def set_required_space_secrets(api: HfApi, space_id: str, pairs: Iterable[tuple[str, str]]) -> None:
    for key, value in pairs:
        api.add_space_secret(repo_id=space_id, key=key, value=value)
        print(f"[OK] Set Space secret: {key}")


def parse_space_id_from_page_url(space_url: str) -> str | None:
    parsed = urlparse(space_url.strip())
    if parsed.netloc != "huggingface.co":
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 3 or parts[0] != "spaces":
        return None
    return f"{parts[1]}/{parts[2]}"


def resolve_space_id(space_id: str, space_url: str) -> str:
    if space_id and "/" in space_id:
        return space_id
    parsed = parse_space_id_from_page_url(space_url) if space_url else None
    if parsed:
        return parsed
    raise SystemExit(
        "HF space id missing/invalid. Use --space-id <owner>/<space-name>, "
        "set HF_SPACE_ID, or pass --space-url https://huggingface.co/spaces/<owner>/<space-name>."
    )


def to_runtime_space_url(space_id: str) -> str:
    owner, name = space_id.split("/", 1)
    normalized_name = name.replace("_", "-")
    return f"https://{owner}-{normalized_name}.hf.space"


def main() -> int:
    args = parse_args()

    space_id = resolve_space_id(args.space_id, args.space_url)

    if not is_real_value(args.token):
        raise SystemExit("HF_TOKEN is missing or looks like a placeholder.")

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists() or not repo_dir.is_dir():
        raise SystemExit(f"Repo directory not found: {repo_dir}")

    api = HfApi(token=args.token)
    who = api.whoami()
    actor = who.get("name") or who.get("fullname") or "unknown"
    print(f"[OK] Authenticated to Hugging Face as: {actor}")

    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )
    print(f"[OK] Space ready: https://huggingface.co/spaces/{space_id}")

    ignore_patterns = [
        ".git",
        ".venv",
        ".pytest_cache",
        "__pycache__",
        "*.pyc",
        ".env",
        ".env.*",
        "*.env",
        "*.env.*",
        ".codex",
        ".github",
    ]

    commit_info = api.upload_folder(
        folder_path=str(repo_dir),
        repo_id=space_id,
        repo_type="space",
        commit_message="chore: sync repository for validation",
        ignore_patterns=ignore_patterns,
    )
    print(f"[OK] Upload complete: {commit_info.oid}")

    if not args.skip_secrets:
        env_values = read_env_file(repo_dir)
        secret_pairs: list[tuple[str, str]] = []
        for key in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
            value = pick_var(key, env_values)
            if value is not None:
                secret_pairs.append((key, value))

        if secret_pairs:
            set_required_space_secrets(api, space_id, secret_pairs)
        else:
            print("[WARN] No non-placeholder runtime variables found for Space secrets.")

    space_url = to_runtime_space_url(space_id)
    print(f"[NEXT] Ping URL: {space_url}")
    print(f"[NEXT] Run: ./scripts/validate-submission.sh {space_url} {repo_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
