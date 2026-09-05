"""Filesystem-backed storage for voice metadata and locally retained reference recordings."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.domain.models import VoiceProfile, VoiceProfilePublic


class VoiceRegistry:
    """Filesystem-backed speaker registry.

    Reference-audio enrollment is intentionally explicit: callers must attest that
    they have permission to use the recording. The service stores only local files.
    """

    def __init__(self, root: Path) -> None:
        """Establish the local voice root directory. Voice data never requires a database for the
        initial single-node deployment."""
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_public(self) -> list[VoiceProfilePublic]:
        """Scan stored profile files, return only validated public fields, skip damaged entries instead
        of failing the whole listing, and sort newest voices first."""
        profiles: list[VoiceProfilePublic] = []
        for metadata_file in self.root.glob("*/profile.json"):
            try:
                payload = json.loads(metadata_file.read_text(encoding="utf-8"))
                profiles.append(VoiceProfilePublic(**{k: payload[k] for k in VoiceProfilePublic.model_fields}))
            except (OSError, ValueError, KeyError):
                continue
        return sorted(profiles, key=lambda item: item.created_at, reverse=True)

    def get(self, voice_id: str) -> VoiceProfile | None:
        """Safely resolve one voice ID to its internal profile. Path separators are rejected up front
        so a caller cannot use the ID field for directory traversal."""
        if not voice_id or "/" in voice_id or "\\" in voice_id:
            return None
        profile_path = self.root / voice_id / "profile.json"
        if not profile_path.exists():
            return None
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        ref = payload.get("reference_audio_path")
        return VoiceProfile(
            id=payload["id"],
            name=payload["name"],
            language=payload.get("language"),
            kind=payload["kind"],
            created_at=payload["created_at"],
            reference_audio_path=(self.root / voice_id / ref) if ref else None,
            nemo_speaker_id=payload.get("nemo_speaker_id"),
            metadata=payload.get("metadata", {}),
        )

    def enroll_reference_audio(
        self,
        *,
        name: str,
        language: str | None,
        audio_bytes: bytes,
        consent_attested: bool,
    ) -> VoiceProfilePublic:
        """Create an opaque voice ID, store already-normalized reference WAV bytes with restrictive
        permissions where supported, write metadata including consent attestation, and return only
        the public projection."""
        if not consent_attested:
            raise ValueError("voice enrollment requires consent_attested=true")
        if not (1 <= len(name.strip()) <= 80):
            raise ValueError("voice name must contain 1-80 characters")
        voice_id = uuid.uuid4().hex
        target = self.root / voice_id
        target.mkdir(mode=0o700, parents=True, exist_ok=False)
        audio_name = "reference.wav"
        audio_path = target / audio_name
        audio_path.write_bytes(audio_bytes)
        try:
            os.chmod(audio_path, 0o600)
        except OSError:
            pass
        created_at = datetime.now(UTC).isoformat()
        payload = {
            "id": voice_id,
            "name": name.strip(),
            "language": language,
            "kind": "reference_audio",
            "created_at": created_at,
            "reference_audio_path": audio_name,
            "nemo_speaker_id": None,
            "metadata": {"consent_attested": True},
        }
        (target / "profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return VoiceProfilePublic(**{k: payload[k] for k in VoiceProfilePublic.model_fields})
