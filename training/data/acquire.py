"""One-command acquisition of approved bootstrap speech corpora with immutable provenance receipts."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.services.corpus_audio import decode_audio
from training.common.language_profile import load_language_profile
from training.common.manifest import file_sha256, normalize_transcript
from training.data.bootstrap_plan import BootstrapPlan, Task
from training.data.catalog import DataSource, SourceCatalog

_METADATA_FIELDS = [
    "audio",
    "text",
    "speaker",
    "dialect",
    "source_id",
    "consent_attested",
    "transcript_reviewed",
    "governance_approved",
    "upstream_validated",
    "training_only",
    "source_license",
    "source_revision",
    "governance_basis",
    "source_dataset",
    "source_config",
    "source_split",
    "sample_rate",
]


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Summary of one source acquisition, including the exact locked upstream revision."""

    source_id: str
    role: str
    revision: str
    imported: int
    skipped: int
    hours: float
    sample_rate: int
    metadata_path: str
    receipt_path: str


def _now() -> str:
    """Return timezone-aware UTC timestamps for immutable acquisition receipts."""
    return datetime.now(UTC).isoformat()


def _safe_name(value: str) -> str:
    """Convert an upstream ID into a filesystem-safe stable token without losing uniqueness."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned[:100] or hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _field(row: dict[str, Any], source: DataSource, semantic: str) -> Any:
    """Read one catalog-mapped provider field; absent optional fields remain None rather than guessed."""
    name = source.fields.get(semantic)
    return row.get(name) if name else None


def target_sample_rate(
    *,
    language: str,
    task: Task,
    profiles_dir: Path = Path("training/configs/languages"),
) -> int:
    """Resolve the canonical model-facing sample rate from the version-controlled language profile."""
    profile = load_language_profile(profiles_dir / f"{language}.yaml")
    if profile.code != language:
        raise ValueError(f"profile {profile.source_path} declares {profile.code!r}, expected {language!r}")
    rate = profile.asr.sample_rate if task == "asr" else profile.tts.sample_rate
    if rate < 8000:
        raise ValueError(f"invalid {task} sample rate for {language}: {rate}")
    return rate


def _raw_audio_bytes(value: Any) -> bytes:
    """Extract encoded audio bytes from Hugging Face decode-disabled values and compatible fallbacks."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return Path(value).read_bytes()
    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw is not None:
            return bytes(raw)
        path = value.get("path")
        if path:
            return Path(str(path)).read_bytes()
        array = value.get("array")
        rate = value.get("sampling_rate")
        if array is not None and rate:
            buffer = io.BytesIO()
            sf.write(buffer, np.asarray(array, dtype=np.float32), int(rate), format="WAV", subtype="PCM_16")
            return buffer.getvalue()
    get_samples = getattr(value, "get_all_samples", None)
    if callable(get_samples):
        samples = get_samples()
        data = getattr(samples, "data", None)
        rate = getattr(samples, "sample_rate", None)
        if data is not None and rate:
            array = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
            if array.ndim > 1:
                array = array.mean(axis=0)
            buffer = io.BytesIO()
            sf.write(buffer, np.asarray(array, dtype=np.float32), int(rate), format="WAV", subtype="PCM_16")
            return buffer.getvalue()
    raise ValueError(f"unsupported provider audio value: {type(value).__name__}")


def _write_normalized_wav(
    payload: bytes,
    destination: Path,
    *,
    max_seconds: float,
    target_rate: int,
) -> float:
    """Decode provider audio, resample to the task policy rate, and persist deterministic mono PCM16 WAV."""
    samples, sample_rate = decode_audio(payload, max_seconds=max_seconds, target_rate=target_rate)
    if sample_rate != target_rate:
        raise ValueError(f"audio normalizer returned {sample_rate} Hz, expected {target_rate} Hz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, samples, sample_rate, format="WAV", subtype="PCM_16")
    return len(samples) / sample_rate


def _lock_path(output_root: Path, source: DataSource) -> Path:
    """Store one source lock outside downloaded audio so it survives metadata rebuilds."""
    return output_root / "locks" / f"{source.source_id}.json"


def resolve_revision(
    source: DataSource,
    *,
    output_root: Path,
    refresh_lock: bool = False,
    token: str | None = None,
) -> str:
    """Resolve a full Hugging Face commit SHA and freeze it before row iteration begins."""
    if source.provider != "huggingface" or not source.repo_id:
        raise ValueError(f"source {source.source_id} is not a Hugging Face source")
    lock_path = _lock_path(output_root, source)
    if lock_path.exists() and not refresh_lock:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if payload.get("repo_id") != source.repo_id:
            raise ValueError(f"lock {lock_path} belongs to a different repository")
        revision = str(payload.get("revision") or "")
        if revision:
            return revision
    if source.revision:
        revision = source.revision
    else:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError("install the 'data' extra to resolve Hugging Face dataset revisions") from exc
        revision = str(HfApi(token=token).dataset_info(source.repo_id).sha or "")
    if source.requires_revision_pin and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(f"source {source.source_id} did not resolve to a full 40-character commit SHA")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_id": source.source_id,
                "repo_id": source.repo_id,
                "revision": revision,
                "locked_at": _now(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return revision


def _load_hf_rows(source: DataSource, *, revision: str, token: str | None) -> Iterable[dict[str, Any]]:
    """Stream a configured subset/split with media decoding disabled for dependency-light acquisition."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install the 'data' extra before acquiring Hugging Face speech datasets") from exc
    kwargs: dict[str, Any] = {
        "path": source.repo_id,
        "split": source.split,
        "revision": revision,
        "streaming": True,
        "token": token,
    }
    if source.config and source.config != "default":
        kwargs["name"] = source.config
    dataset = load_dataset(**kwargs)
    decode = getattr(dataset, "decode", None)
    if callable(decode):
        dataset = decode(False)
    return dataset


def _upstream_id(row: dict[str, Any], source: DataSource, index: int) -> str:
    """Use a provider ID when available and otherwise derive a deterministic row token."""
    value = _field(row, source, "id")
    if value is not None and str(value).strip():
        return str(value).strip()
    return f"row-{index:09d}"


def _release_stream(iterator: object, rows: object) -> None:
    """Release abandoned streaming iterators before interpreter finalization."""
    del iterator
    del rows
    gc.collect()


def _metadata_audio_exists(metadata_path: Path) -> bool:
    """Verify a completed source receipt still points at present normalized audio before reusing it."""
    try:
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                value = (row.get("audio") or "").strip()
                if not value or not Path(value).exists():
                    return False
    except (OSError, csv.Error):
        return False
    return True


def _reuse_completed_source(
    *,
    source: DataSource,
    role: str,
    revision: str,
    target_rate: int,
    max_samples: int | None,
    metadata_path: Path,
    receipt_path: Path,
) -> AcquisitionResult | None:
    """Reuse a fully verified source at the same revision/rate, providing source-level resumability."""
    if not metadata_path.exists() or not receipt_path.exists():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("resolved_revision") != revision:
            return None
        if int(receipt.get("sample_rate", 0)) != target_rate:
            return None
        if receipt.get("requested_max_samples") != max_samples:
            return None
        expected_hash = str(receipt.get("metadata_sha256") or "")
        if not expected_hash or file_sha256(metadata_path) != expected_hash:
            return None
        if not _metadata_audio_exists(metadata_path):
            return None
        return AcquisitionResult(
            source_id=source.source_id,
            role=role,
            revision=revision,
            imported=int(receipt.get("imported", 0)),
            skipped=int(receipt.get("skipped", 0)),
            hours=float(receipt.get("hours", 0.0)),
            sample_rate=target_rate,
            metadata_path=str(metadata_path),
            receipt_path=str(receipt_path),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def acquire_source(
    source: DataSource,
    *,
    language: str,
    task: Task,
    role: str,
    output_root: Path,
    target_sample_rate: int = 16000,
    max_samples: int | None = None,
    refresh_lock: bool = False,
    force_reacquire: bool = False,
    max_clip_seconds: float = 120.0,
    token: str | None = None,
) -> AcquisitionResult:
    """Stream one approved source into normalized WAV files, metadata, a lock, and an audit receipt."""
    if language not in source.languages:
        raise ValueError(f"source {source.source_id} does not support language {language}")
    if not source.supports(task):
        raise ValueError(f"source {source.source_id} is not approved for {task}")
    if role == "train" and not source.allows("production"):
        raise ValueError(f"source {source.source_id} cannot enter a production training corpus")
    if role == "eval" and source.usage not in {"evaluation", "production"}:
        raise ValueError(f"source {source.source_id} cannot enter evaluation")
    if source.provider != "huggingface":
        raise ValueError(f"acquisition provider is not implemented for {source.source_id}: {source.provider}")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive when provided")
    if target_sample_rate < 8000:
        raise ValueError("target_sample_rate must be at least 8000 Hz")

    revision = resolve_revision(
        source,
        output_root=output_root,
        refresh_lock=refresh_lock,
        token=token,
    )
    target = output_root / language / task / role / source.source_id
    audio_dir = target / "audio"
    metadata_path = target / "metadata.csv"
    rejected_path = target / "rejected.jsonl"
    receipt_path = target / "SOURCE_RECEIPT.json"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not force_reacquire:
        reused = _reuse_completed_source(
            source=source,
            role=role,
            revision=revision,
            target_rate=target_sample_rate,
            max_samples=max_samples,
            metadata_path=metadata_path,
            receipt_path=receipt_path,
        )
        if reused is not None:
            return reused

    seen: set[str] = set()
    imported = 0
    skipped = 0
    hours = 0.0
    rows = _load_hf_rows(source, revision=revision, token=token)
    iterator = iter(rows)

    try:
        with (
            metadata_path.open("w", encoding="utf-8", newline="") as handle,
            rejected_path.open("w", encoding="utf-8") as reject_handle,
        ):
            writer = csv.DictWriter(handle, fieldnames=_METADATA_FIELDS)
            writer.writeheader()
            for index, row in enumerate(iterator):
                if max_samples is not None and imported >= max_samples:
                    break
                try:
                    text = normalize_transcript(str(_field(row, source, "text") or ""))
                    if not text:
                        raise ValueError("empty transcript")
                    payload = _raw_audio_bytes(_field(row, source, "audio"))
                    upstream_id = _upstream_id(row, source, index)
                    provisional = audio_dir / f"{_safe_name(upstream_id)}.wav"
                    duration = _write_normalized_wav(
                        payload,
                        provisional,
                        max_seconds=max_clip_seconds,
                        target_rate=target_sample_rate,
                    )
                    digest = file_sha256(provisional)
                    if digest in seen:
                        provisional.unlink(missing_ok=True)
                        raise ValueError("duplicate audio within source acquisition")
                    seen.add(digest)
                    final_path = audio_dir / f"{_safe_name(upstream_id)}-{digest[:12]}.wav"
                    if final_path != provisional:
                        provisional.replace(final_path)
                    speaker_value = _field(row, source, "speaker")
                    dialect_value = _field(row, source, "dialect")
                    speaker = str(speaker_value).strip() if speaker_value is not None else ""
                    dialect = str(dialect_value).strip() if dialect_value is not None else "unknown"
                    training_only = source.training_only or not speaker
                    writer.writerow(
                        {
                            "audio": str(final_path.resolve()),
                            "text": text,
                            "speaker": speaker,
                            "dialect": dialect or "unknown",
                            "source_id": f"{source.source_id}:{upstream_id}",
                            "consent_attested": "false",
                            "transcript_reviewed": "false",
                            "governance_approved": str(source.governance_approved).lower(),
                            "upstream_validated": str(source.upstream_validated).lower(),
                            "training_only": str(training_only).lower(),
                            "source_license": source.license,
                            "source_revision": revision,
                            "governance_basis": f"licensed-external:{source.license}",
                            "source_dataset": source.repo_id or "",
                            "source_config": source.config or "default",
                            "source_split": source.split,
                            "sample_rate": target_sample_rate,
                        }
                    )
                    imported += 1
                    hours += duration / 3600.0
                except Exception as exc:
                    skipped += 1
                    reject_handle.write(
                        json.dumps(
                            {
                                "row": index,
                                "source_id": source.source_id,
                                "error": f"{type(exc).__name__}:{str(exc)[:500]}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        _release_stream(iterator, rows)

    metadata_sha = file_sha256(metadata_path)
    receipt = {
        "schema_version": 2,
        "source": asdict(source),
        "language": language,
        "task": task,
        "role": role,
        "resolved_revision": revision,
        "sample_rate": target_sample_rate,
        "requested_max_samples": max_samples,
        "imported": imported,
        "skipped": skipped,
        "hours": round(hours, 6),
        "metadata_sha256": metadata_sha,
        "rejected_path": str(rejected_path),
        "acquired_at": _now(),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return AcquisitionResult(
        source_id=source.source_id,
        role=role,
        revision=revision,
        imported=imported,
        skipped=skipped,
        hours=hours,
        sample_rate=target_sample_rate,
        metadata_path=str(metadata_path),
        receipt_path=str(receipt_path),
    )


def merge_metadata(paths: list[Path], output: Path) -> int:
    """Merge source manifests without changing per-row provenance or silently deduplicating labels."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=_METADATA_FIELDS)
        writer.writeheader()
        for path in paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    writer.writerow({field: row.get(field, "") for field in _METADATA_FIELDS})
                    count += 1
    return count


def dry_run_plan(
    *,
    language: str,
    task: Task,
    catalog: SourceCatalog,
    plan: BootstrapPlan,
    include_eval: bool,
) -> list[dict[str, Any]]:
    """Return the exact acquisition plan without importing HF libraries or touching the network."""
    result: list[dict[str, Any]] = []
    for planned in plan.planned(language=language, task=task, include_eval=include_eval):
        source = catalog.get(planned.source_id)
        result.append(
            {
                "source_id": source.source_id,
                "role": planned.role,
                "repo_id": source.repo_id,
                "revision": source.revision or "resolve-and-lock",
                "config": source.config,
                "split": source.split,
                "data_files_glob": source.data_files_glob,
                "license": source.license,
                "gated": source.gated,
                "optional": source.optional,
                "training_only": source.training_only,
            }
        )
    return result


def acquire_language(
    *,
    language: str,
    task: Task,
    catalog_path: Path,
    plan_path: Path,
    output_root: Path,
    profiles_dir: Path = Path("training/configs/languages"),
    include_eval: bool = False,
    max_samples: int | None = None,
    refresh_lock: bool = False,
    force_reacquire: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    """Acquire every planned source and write separate combined train/evaluation metadata files."""
    catalog = SourceCatalog(catalog_path)
    plan = BootstrapPlan(plan_path)
    sample_rate = target_sample_rate(language=language, task=task, profiles_dir=profiles_dir)
    results: list[AcquisitionResult] = []
    failures: list[dict[str, str]] = []
    train_metadata: list[Path] = []
    eval_metadata: list[Path] = []
    for planned in plan.planned(language=language, task=task, include_eval=include_eval):
        source = catalog.get(planned.source_id)
        try:
            result = acquire_source(
                source,
                language=language,
                task=task,
                role=planned.role,
                output_root=output_root,
                target_sample_rate=sample_rate,
                max_samples=max_samples,
                refresh_lock=refresh_lock,
                force_reacquire=force_reacquire,
                token=token,
            )
            results.append(result)
            target_list = train_metadata if planned.role == "train" else eval_metadata
            target_list.append(Path(result.metadata_path))
        except Exception as exc:
            if not source.optional:
                raise
            failures.append({"source_id": source.source_id, "error": f"{type(exc).__name__}:{exc}"})
    combined_dir = output_root / language / task
    combined_dir.mkdir(parents=True, exist_ok=True)
    train_combined = combined_dir / "metadata.csv"
    eval_combined = combined_dir / "evaluation_metadata.csv"
    train_rows = merge_metadata(train_metadata, train_combined) if train_metadata else 0
    eval_rows = merge_metadata(eval_metadata, eval_combined) if eval_metadata else 0
    summary = {
        "schema_version": 2,
        "language": language,
        "task": task,
        "sample_rate": sample_rate,
        "sources": [asdict(result) for result in results],
        "optional_failures": failures,
        "train_rows": train_rows,
        "evaluation_rows": eval_rows,
        "train_metadata": str(train_combined) if train_rows else None,
        "evaluation_metadata": str(eval_combined) if eval_rows else None,
    }
    (combined_dir / "ACQUISITION_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    """Acquire one language/task bootstrap corpus or print the exact dry-run plan."""
    parser = argparse.ArgumentParser(description="Acquire approved public bootstrap speech datasets")
    parser.add_argument("--language", choices=["tw", "gaa", "ee", "ha"], required=True)
    parser.add_argument("--task", choices=["asr", "tts"], required=True)
    parser.add_argument("--catalog", type=Path, default=Path("training/configs/source_catalog.yaml"))
    parser.add_argument("--plan", type=Path, default=Path("training/configs/bootstrap_corpora.yaml"))
    parser.add_argument("--profiles-dir", type=Path, default=Path("training/configs/languages"))
    parser.add_argument("--output-root", type=Path, default=Path("data/bootstrap"))
    parser.add_argument("--include-eval", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--refresh-lock", action="store_true")
    parser.add_argument("--force-reacquire", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    catalog = SourceCatalog(args.catalog)
    plan = BootstrapPlan(args.plan)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "language": args.language,
                    "task": args.task,
                    "sample_rate": target_sample_rate(
                        language=args.language,
                        task=args.task,
                        profiles_dir=args.profiles_dir,
                    ),
                    "sources": dry_run_plan(
                        language=args.language,
                        task=args.task,
                        catalog=catalog,
                        plan=plan,
                        include_eval=args.include_eval,
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = acquire_language(
        language=args.language,
        task=args.task,
        catalog_path=args.catalog,
        plan_path=args.plan,
        output_root=args.output_root,
        profiles_dir=args.profiles_dir,
        include_eval=args.include_eval,
        max_samples=args.max_samples,
        refresh_lock=args.refresh_lock,
        force_reacquire=args.force_reacquire,
        token=os.environ.get("HF_TOKEN") or None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
