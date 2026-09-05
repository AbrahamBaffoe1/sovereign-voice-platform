# WebSocket Protocol v1

Endpoint: `/v1/conversation`

## Client -> server

### `start`

JSON control frame. Opens one voice turn.

```json
{
  "type": "start",
  "sample_rate": 16000,
  "input_language": "en",
  "output_language": "en",
  "voice_id": null,
  "hotwords": null,
  "system_prompt": null
}
```

Then send binary frames containing mono signed 16-bit little-endian PCM.

### `commit`

```json
{"type":"commit"}
```

Finalizes the buffered turn and starts ASR -> LLM -> TTS.

### `cancel`

```json
{"type":"cancel"}
```

Drops the buffered turn.

## Server -> client

`{"type":"started"}` confirms the turn buffer.

`{"type":"processing"}` means commit was accepted.

`result` contains:

```json
{
  "type": "result",
  "transcript": "...",
  "input_language": "en",
  "response_text": "...",
  "output_language": "en",
  "audio_sample_rate": 24000,
  "timings_ms": {"asr": 0, "llm": 0, "tts": 0, "total": 0}
}
```

The next binary WebSocket message is a complete WAV file. `{"type":"audio_end"}` terminates the server output for that turn.

Errors are JSON: `{"type":"error","error":"..."}`.
