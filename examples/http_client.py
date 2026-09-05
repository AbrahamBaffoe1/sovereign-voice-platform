"""Minimal HTTP client that proves the text-to-speech API contract end to end."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx


def main() -> None:
    """Parse command-line options, call the local TTS endpoint, fail on non-2xx responses, and write
    the returned WAV bytes to disk. This example intentionally contains no retry magic so API
    failures stay visible while debugging."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--text", default="Hello from my local voice platform.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", type=Path, default=Path("speech.wav"))
    args = parser.parse_args()

    headers = {"X-Voice-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=120) as client:
        response = client.post(
            "/v1/speech",
            json={"text": args.text, "language": args.language, "pace": 1.0},
        )
        response.raise_for_status()
        args.output.write_bytes(response.content)
        print(f"wrote {args.output} ({len(response.content)} bytes)")


if __name__ == "__main__":
    main()
