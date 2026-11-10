"""Legacy sequence helpers; raw-signal DL training runs on CUDA.

The Version 1 trainer does not use ML feature sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceRecord:
    """One fixed-length sequence and its patient/recording provenance."""

    indices: tuple[int, ...]
    patient_id: str
    recording_id: str
    label: int
    start_seconds: float
    end_seconds: float


def build_sequence_records(
    metadata: Sequence[Mapping[str, object]],
    *,
    sequence_length: int,
    drop_incomplete: bool = True,
) -> list[SequenceRecord]:
    """Group rows by recording, sort chronologically, and never cross boundaries."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    groups: dict[tuple[str, str], list[tuple[int, Mapping[str, object]]]] = {}
    for index, item in enumerate(metadata):
        try:
            key = (str(item["patient_id"]), str(item["recording_id"]))
            float(item["start_seconds"])
            float(item["end_seconds"])
            int(item["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid sequence metadata at index {index}") from exc
        groups.setdefault(key, []).append((index, item))

    records: list[SequenceRecord] = []
    for (patient_id, recording_id), rows in sorted(groups.items()):
        rows.sort(key=lambda pair: (float(pair[1]["start_seconds"]), pair[0]))
        for offset in range(0, len(rows), sequence_length):
            chunk = rows[offset : offset + sequence_length]
            if len(chunk) < sequence_length and drop_incomplete:
                continue
            indices = tuple(row[0] for row in chunk)
            records.append(
                SequenceRecord(
                    indices=indices,
                    patient_id=patient_id,
                    recording_id=recording_id,
                    label=int(any(int(row[1]["label"]) for row in chunk)),
                    start_seconds=float(chunk[0][1]["start_seconds"]),
                    end_seconds=float(chunk[-1][1]["end_seconds"]),
                )
            )
    return records


class EEGSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Torch dataset backed by an ML-to-DL feature matrix."""

    def __init__(
        self,
        features: np.ndarray,
        metadata: Sequence[Mapping[str, object]],
        *,
        sequence_length: int,
        drop_incomplete: bool = True,
    ) -> None:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]
        if values.ndim != 2 or values.shape[0] != len(metadata):
            raise ValueError("features must be [rows, features] and align with metadata")
        if not np.isfinite(values).all():
            raise ValueError("features contain non-finite values")
        self.features = values
        self.records = build_sequence_records(
            metadata,
            sequence_length=sequence_length,
            drop_incomplete=drop_incomplete,
        )
        self.labels = np.asarray([record.label for record in self.records], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        values = self.features[list(record.indices)]
        return torch.from_numpy(values), torch.tensor(record.label, dtype=torch.float32)


def subset_metadata(
    metadata: Sequence[Mapping[str, object]],
    patients: Iterable[str],
) -> list[dict[str, object]]:
    """Return metadata for a patient set while preserving source row order."""

    selected = set(patients)
    return [dict(item) for item in metadata if str(item.get("patient_id")) in selected]
