"""Reference WebSocket client for sending one PCM16 voice turn and receiving synthesized WAV audio."""

from __future__ import annotations

import argparse
import asyncio
import json
import wave
from pathlib import Path

import websockets


def read_pcm16_mono(path: Path) -> tuple[int, bytes]:
    """Read a WAV file only when it matches the WebSocket wire contract: mono, signed 16-bit PCM.
    Returning the original sample rate lets the server interpret raw frame bytes without guessing."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("example client requires mono 16-bit PCM WAV")
        return wav.getframerate(), wav.readframes(wav.getnframes())


async def run(args: argparse.Namespace) -> None:
    """Execute one complete WebSocket turn: negotiate turn metadata, stream audio in 100 ms chunks,
    commit the turn, and persist the binary WAV response. The loop exits only after the explicit
    audio_end control frame."""
    sample_rate, pcm = read_pcm16_mono(args.input)
    headers = {"X-Voice-API-Key": args.api_key} if args.api_key else None
    async with websockets.connect(args.url, additional_headers=headers, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "sample_rate": sample_rate,
                    "input_language": args.input_language,
                    "output_language": args.output_language,
                    "voice_id": args.voice_id,
                }
            )
        )
        print(await ws.recv())
        frame_bytes = int(sample_rate * 2 * 0.1)  # 100 ms PCM16 mono chunks
        for offset in range(0, len(pcm), frame_bytes):
            await ws.send(pcm[offset : offset + frame_bytes])
        await ws.send(json.dumps({"type": "commit"}))

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                args.output.write_bytes(message)
                print(f"wrote {args.output} ({len(message)} bytes)")
            else:
                print(message)
                if json.loads(message).get("type") == "audio_end":
                    break


def main() -> None:
    """Parse the example client options and run the async WebSocket exchange in a fresh event loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8080/v1/conversation")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("response.wav"))
    parser.add_argument("--input-language", default="en")
    parser.add_argument("--output-language", default="en")
    parser.add_argument("--voice-id", default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
