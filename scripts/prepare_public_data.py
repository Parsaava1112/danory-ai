"""Prepare bounded public Persian corpora with per-source failure isolation."""
import os

# Disable Xet-backed transfer before importing Hugging Face libraries.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_DATASETS_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
import shutil

from datasets import load_dataset
from huggingface_hub import hf_hub_download

SOURCES = {
    "naab": {
        "id": "SLPL/naab",
        "license_note": "Composite corpus; review provenance/privacy."
    },
    "fineweb2_hq_fa": {
        "id": "mohammadkrb/fineweb2-HQ-persian",
        "license_note": "ODC-BY; preserve attribution."
    },
    "cs_farsi": {
        "id": "Kamyar-zeinalipour/CS_Farsi",
        "license_note": "CC-BY-SA-4.0; attribution/share-alike may apply."
    },
    "farsinstruct": {
        "id": "ParsiAI/FarsInstruct",
        "license_note": "Apache-2.0; review source metadata."
    },
    "persian_gk": {
        "id": "PersianML/persian-general-knowledge",
        "license_note": "CC-BY-4.0; preserve attribution."
    },
}


def clean(text):
    s = unicodedata.normalize(
        "NFC",
        str(text or "")
    ).replace("ي", "ی").replace("ك", "ک")

    s = s.replace("ى", "ی").replace("ـ", "")
    s = s.replace("\u200f", "").replace("\u200e", "")

    s = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        " ",
        s
    )

    return re.sub(r"\s+", " ", s).strip()


def is_persian_quality(s, min_ratio=.35):
    letters = [c for c in s if c.isalpha()]

    if len(letters) < 12:
        return False

    fa = sum("\u0600" <= c <= "\u06ff" for c in letters)

    return fa / max(1, len(letters)) >= min_ratio


def export_text(
    dataset,
    out,
    limit,
    seen,
    source,
    min_chars=40,
    max_chars=12000,
    append=False
):
    count = 0
    chars = 0

    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open(
        "a" if append else "w",
        encoding="utf-8"
    ) as f:

        for row in dataset:

            raw = row.get("text", "") if isinstance(row, dict) else ""
            text = clean(raw)

            if len(text) < min_chars:
                continue

            if not is_persian_quality(text):
                continue

            text = text[:max_chars]

            digest = hashlib.sha256(
                text.encode()
            ).hexdigest()

            if digest in seen:
                continue

            seen.add(digest)

            f.write(
                json.dumps(
                    {
                        "text": text,
                        "source": source
                    },
                    ensure_ascii=False
                ) + "\n"
            )

            count += 1
            chars += len(text)

            if count % 10000 == 0:
                print(
                    f"{source}: {count:,} accepted rows",
                    flush=True
                )

            if limit and count >= limit:
                break

    print(
        f"{source}: wrote {count:,} rows / "
        f"{chars:,} chars -> {out}",
        flush=True
    )

    return count


def to_messages(row):

    msgs = row.get("messages")

    if isinstance(msgs, list):

        out = []

        for m in msgs:

            if not isinstance(m, dict):
                continue

            role = str(
                m.get("role", m.get("from", ""))
            ).lower()

            role = {
                "human": "user",
                "gpt": "assistant"
            }.get(role, role)

            content = m.get(
                "content",
                m.get("value", "")
            )

            if (
                role in ("system", "user", "assistant")
                and isinstance(content, str)
                and content.strip()
            ):

                out.append({
                    "role": role,
                    "content": clean(content)
                })

        if (
            any(m["role"] == "user" for m in out)
            and any(m["role"] == "assistant" for m in out)
        ):
            return out

    prompt = row.get(
        "prompt",
        row.get("question", row.get("inputs", ""))
    )

    answer = row.get(
        "assistant",
        row.get(
            "answer",
            row.get(
                "outputs",
                row.get("output", row.get("response", ""))
            )
        )
    )

    if (
        isinstance(prompt, str)
        and isinstance(answer, str)
        and len(answer.strip()) >= 8
    ):

        return [
            {
                "role": "user",
                "content": clean(prompt)
            },
            {
                "role": "assistant",
                "content": clean(answer)
            }
        ]

    return None


def load_stream_with_retries(
    dataset_id,
    split="train",
    retries=3
):

    last_exc = None

    for attempt in range(1, retries + 1):

        try:

            print(
                f"Loading {dataset_id} "
                f"({attempt}/{retries})...",
                flush=True
            )

            return load_dataset(
                dataset_id,
                split=split,
                streaming=True
            )

        except Exception as exc:

            last_exc = exc

            print(
                f"WARN: load failed for {dataset_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True
            )

            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))

    raise RuntimeError(
        f"Could not load {dataset_id} after {retries} attempts"
    ) from last_exc


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--naab-train-rows",
        type=int,
        default=150000
    )

    p.add_argument(
        "--naab-val-rows",
        type=int,
        default=12000
    )

    p.add_argument(
        "--fineweb-rows",
        type=int,
        default=100000
    )

    p.add_argument(
        "--sft-rows",
        type=int,
        default=20000
    )

    p.add_argument(
        "--naab-train-shards",
        default="0"
    )

    p.add_argument(
        "--naab-test-shards",
        default="2"
    )

    p.add_argument(
        "--include-sft",
        action="store_true"
    )

    p.add_argument(
        "--out",
        default="data/public"
    )

    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for v in (
        a.naab_train_rows,
        a.naab_val_rows,
        a.fineweb_rows,
        a.sft_rows
    ):

        if v < 0:
            raise ValueError(
                "row limits must be non-negative"
            )

    train = out / "train.jsonl"
    val = out / "validation.jsonl"

    seen = set()
    train_count = 0
    source_status = {}

    shard_dir = out / "_naab_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    def naab_lines(split, raw_indexes):

        indexes = [
            int(x.strip())
            for x in raw_indexes.split(",")
            if x.strip()
        ]

        upper = 125 if split == "train" else 2

        if any(i < 0 or i > upper for i in indexes):
            raise ValueError(
                f"Invalid {split} shard index; "
                f"expected 0..{upper}"
            )

        if len(indexes) > 3:
            raise ValueError(
                "Select at most three NAAB shards per run"
            )

        paths = []

        for i in indexes:

            filename = (
                f"data/train-{i:05d}-of-00126.txt"
                if split == "train"
                else f"data/test-{i:05d}-of-00003.txt"
            )

            paths.append(
                Path(
                    hf_hub_download(
                        repo_id="SLPL/naab",
                        filename=filename,
                        repo_type="dataset",
                        local_dir=str(shard_dir)
                    )
                )
            )

        def rows():

            for path in paths:

                with path.open(
                    encoding="utf-8",
                    errors="replace"
                ) as f:

                    for line in f:

                        if line.strip():
                            yield {"text": line}

        return rows()

    try:

        if a.naab_val_rows:

            try:

                train_count += export_text(
                    naab_lines(
                        "test",
                        a.naab_test_shards
                    ),
                    val,
                    a.naab_val_rows,
                    seen,
                    "SLPL/naab:test"
                )

                source_status["naab_validation"] = "ok"

            except Exception as exc:

                source_status["naab_validation"] = (
                    f"failed: {type(exc).__name__}: {exc}"
                )

                print(
                    f"WARN: NAAB validation skipped: {exc}",
                    flush=True
                )

        if a.naab_train_rows:

            try:

                train_count += export_text(
                    naab_lines(
                        "train",
                        a.naab_train_shards
                    ),
                    train,
                    a.naab_train_rows,
                    seen,
                    "SLPL/naab"
                )

                source_status["naab_train"] = "ok"

            except Exception as exc:

                source_status["naab_train"] = (
                    f"failed: {type(exc).__name__}: {exc}"
                )

                print(
                    f"WARN: NAAB train skipped: {exc}",
                    flush=True
                )

        if a.fineweb_rows:

            try:

                ds = load_stream_with_retries(
                    SOURCES["fineweb2_hq_fa"]["id"]
                )

                train_count += export_text(
                    ds,
                    train,
                    a.fineweb_rows,
                    seen,
                    SOURCES["fineweb2_hq_fa"]["id"],
                    append=train.exists()
                )

                source_status["fineweb2_hq_fa"] = "ok"

            except Exception as exc:

                source_status["fineweb2_hq_fa"] = (
                    f"failed: {type(exc).__name__}: {exc}"
                )

                print(
                    f"WARN: FineWeb2 Persian skipped: {exc}",
                    flush=True
                )

    finally:

        shutil.rmtree(
            shard_dir,
            ignore_errors=True
        )
    sft_count = 0
    validation_count = 0
    sft_sources = {}

    if a.include_sft:

        sft = out / "sft.jsonl"
        sft_val = out / "sft_validation.jsonl"

        seen_s = set()

        limits = [
            max(1, a.sft_rows // 2),
            min(
                5897,
                max(1, a.sft_rows // 4)
            ),
            max(1, a.sft_rows // 4)
        ]

        sources_order = [
            ("farsinstruct", "Apache-2.0"),
            ("persian_gk", "CC-BY-4.0"),
            ("cs_farsi", "CC-BY-SA-4.0")
        ]

        with (
            sft.open("w", encoding="utf-8") as f,
            sft_val.open("w", encoding="utf-8") as vf
        ):

            for (source_key, license_name), limit in zip(
                sources_order,
                limits
            ):

                source_id = SOURCES[source_key]["id"]
                accepted = 0

                try:

                    dataset = load_stream_with_retries(
                        source_id
                    )

                    for row in dataset:

                        msgs = to_messages(row)

                        if not msgs:
                            continue

                        key = hashlib.sha256(
                            json.dumps(
                                msgs,
                                ensure_ascii=False,
                                sort_keys=True
                            ).encode()
                        ).hexdigest()

                        if key in seen_s:
                            continue

                        seen_s.add(key)

                        record = {
                            "messages": msgs,
                            "source": source_id,
                            "license": license_name
                        }

                        target = (
                            vf
                            if int(key[:8], 16) % 10 == 0
                            else f
                        )

                        target.write(
                            json.dumps(
                                record,
                                ensure_ascii=False
                            ) + "\n"
                        )

                        accepted += 1
                        sft_count += 1

                        if accepted >= limit:
                            break

                    sft_sources[source_key] = (
                        f"ok: {accepted} rows"
                    )

                    print(
                        f"{source_id}: accepted "
                        f"{accepted:,} SFT records",
                        flush=True
                    )

                except Exception as exc:

                    sft_sources[source_key] = (
                        f"failed: {type(exc).__name__}: {exc}; "
                        f"partial rows={accepted}"
                    )

                    print(
                        f"WARN: {source_id} skipped/partial: {exc}",
                        flush=True
                    )

        validation_count = sum(
            1
            for _ in sft_val.open(
                encoding="utf-8"
            )
        )

        print(
            f"SFT total rows: {sft_count:,}; "
            f"held-out validation rows: {validation_count:,}",
            flush=True
        )

        if sft_count == 0:
            raise RuntimeError(
                "SFT preparation yielded zero rows; "
                "aborting before training"
            )

    manifest = {

        "sources": SOURCES,

        "train_rows_requested": (
            a.naab_train_rows + a.fineweb_rows
        ),

        "train_rows_written": train_count,

        "validation_rows_requested": a.naab_val_rows,

        "sft_enabled": bool(a.include_sft),

        "sft_rows_written": sft_count,

        "sft_validation_rows_written": validation_count,

        "source_status": source_status,

        "sft_source_status": sft_sources,

        "notes": [
            "Only public sources are downloaded; "
            "private/local datasets are never accessed.",
            "Exact deduplication within SFT generation; "
            "validation split selected deterministically.",
            "Review provenance, privacy, attribution "
            "and license terms before redistribution "
            "or commercial use."
        ]
    }

    (out / "data_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        "Public data preparation completed.",
        flush=True
    )

    print(
        json.dumps(
            {
                "sft_rows": sft_count,
                "validation_rows": validation_count,
                "sft_sources": sft_sources
            },
            ensure_ascii=False
        ),
        flush=True
    )


if __name__ == "__main__":
    main()

    # Hugging Face streaming + native Arrow extensions can occasionally crash
    # during CPython finalization on CI (exit 134), after outputs are complete.
    # All files are closed and output is flushed before this controlled exit.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
