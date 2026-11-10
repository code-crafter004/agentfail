"""Append-only trace collector. Each event is written and flushed immediately,
so an interrupted or crashed run keeps its partial trace."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracing.schema import Event


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TraceCollector:
    def __init__(self, run_id: str, run_dir: str | Path):
        self.run_id = run_id
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, type: str, *, parent: str | None = None, agent_id: str = "agent", **fields: Any) -> Event:
        with self._lock:
            self._seq += 1
            ev = Event(
                run_id=self.run_id,
                event_id=f"EV-{self._seq:03d}",
                sequence_number=self._seq,
                parent_event_id=parent,
                agent_id=agent_id,
                timestamp=utc_now(),
                type=type,
                **fields,
            )
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(ev.model_dump_json(exclude_none=True) + "\n")
                f.flush()
            return ev

    def events(self) -> list[Event]:
        return read_events(self.path)

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.dir / name
        data = obj.model_dump() if hasattr(obj, "model_dump") else obj
        p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return p


def read_events(path: str | Path) -> list[Event]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(Event.model_validate_json(line))
    return out
