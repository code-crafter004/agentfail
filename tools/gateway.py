"""Tool Gateway. The only way an agent touches tools.

Validates tool name, permissions and argument schema, checks paths, routes the
call, and logs request and result events. Framework adapters must call
gateway.call(...) so every framework produces identical tool traces.
"""
from __future__ import annotations

import ast
import csv
import operator
import time
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, ValidationError

from tracing.collector import TraceCollector

TOOL_VERSION = "1.0"
MAX_PREVIEW_ROWS = 20
MAX_WRITE_BYTES = 100_000
ALLOWED_WRITE_SUFFIXES = {".md", ".txt", ".json", ".csv"}


class ToolError(Exception):
    def __init__(self, code: str, message: str, injected: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.injected = injected


# ---------- argument schemas (used for validation) ----------
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadCsvArgs(_Args):
    file_name: str


class AggregateArgs(_Args):
    file_name: str
    group_column: str
    value_column: str


class CalcArgs(_Args):
    expression: str


class WriteReportArgs(_Args):
    file_name: str
    content: str


class MockApiArgs(_Args):
    endpoint: str
    payload: dict = {}


ARG_MODELS = {
    "read_csv": ReadCsvArgs,
    "aggregate_revenue": AggregateArgs,
    "calculate": CalcArgs,
    "write_report": WriteReportArgs,
    "mock_api_request": MockApiArgs,
}


def _spec(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": props, "required": required},
    }


_S = {"type": "string"}
TOOL_SPECS = {
    "read_csv": _spec(
        "read_csv",
        f"Read an input CSV file. Returns column names, row count and the first {MAX_PREVIEW_ROWS} rows.",
        {"file_name": {**_S, "description": "Input file name, e.g. data.csv"}},
        ["file_name"],
    ),
    "aggregate_revenue": _spec(
        "aggregate_revenue",
        "Sum a numeric column grouped by another column of an input CSV. Returns {group: total}.",
        {
            "file_name": {**_S, "description": "Input CSV file name"},
            "group_column": {**_S, "description": "Column to group by"},
            "value_column": {**_S, "description": "Numeric column to sum"},
        },
        ["file_name", "group_column", "value_column"],
    ),
    "calculate": _spec(
        "calculate",
        "Evaluate a basic arithmetic expression using + - * / ** and parentheses.",
        {"expression": {**_S, "description": "e.g. (12.5 + 3) * 2"}},
        ["expression"],
    ),
    "write_report": _spec(
        "write_report",
        "Write a text or markdown file into the output directory.",
        {
            "file_name": {**_S, "description": "Output file name, e.g. report.md"},
            "content": {**_S, "description": "File content"},
        },
        ["file_name", "content"],
    ),
    "mock_api_request": _spec(
        "mock_api_request",
        "Call a mock internal API endpoint with a JSON payload.",
        {
            "endpoint": {**_S, "description": "e.g. /sales/summary"},
            "payload": {"type": "object", "description": "JSON payload"},
        },
        ["endpoint"],
    ),
}

MOCK_ENDPOINTS = {
    "/sales/summary": {"total_orders": 1240, "currency": "INR", "period": "2025-H1"},
    "/customers/count": {"active_customers": 318},
}

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expr: str) -> float:
    if len(expr) > 200:
        raise ToolError("BAD_EXPRESSION", "Expression too long")

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 10:
                raise ToolError("BAD_EXPRESSION", "Exponent too large")
            try:
                return _BIN[type(n.op)](l, r)
            except ZeroDivisionError:
                raise ToolError("BAD_EXPRESSION", "Division by zero")
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        raise ToolError("BAD_EXPRESSION", "Unsupported expression")

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        raise ToolError("BAD_EXPRESSION", "Invalid syntax")
    return ev(tree)


class ToolGateway:
    def __init__(
        self,
        collector: TraceCollector,
        input_dir: str | Path,
        output_dir: str | Path,
        allowed_tools: list[str],
        fault_schedule: Optional[dict] = None,
        agent_id: str = "agent",
    ):
        self.collector = collector
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.allowed = list(allowed_tools)
        self.fault_schedule = fault_schedule or {}
        self.agent_id = agent_id
        self._calls: dict[str, int] = defaultdict(int)

    # ---- what the LLM sees ----
    def specs(self) -> list[dict]:
        return [TOOL_SPECS[t] for t in self.allowed if t in TOOL_SPECS]

    def openai_tools(self) -> list[dict]:
        return [{"type": "function", "function": s} for s in self.specs()]

    # ---- main entry ----
    def call(self, name: str, arguments: Any, parent_event_id: Optional[str] = None) -> dict:
        req = self.collector.emit(
            "tool_call", parent=parent_event_id, agent_id=self.agent_id, tool_name=name, arguments=arguments
        )
        start = time.perf_counter()
        meta: dict = {}
        try:
            if name not in TOOL_SPECS:
                raise ToolError("UNKNOWN_TOOL", f"Tool '{name}' does not exist")
            if name not in self.allowed:
                raise ToolError("TOOL_NOT_ALLOWED", f"Tool '{name}' is not allowed for this task")
            if not isinstance(arguments, dict):
                raise ToolError("SCHEMA_INVALID", "Arguments must be a JSON object")
            try:
                args = ARG_MODELS[name](**arguments)
            except ValidationError as e:
                first = e.errors()[0]
                loc = ".".join(str(x) for x in first["loc"]) or "arguments"
                raise ToolError("SCHEMA_INVALID", f"{loc}: {first['msg']}")
            self._calls[name] += 1
            self._maybe_fault(name)
            result = getattr(self, f"_t_{name}")(args)
            payload = {"status": "ok", "result": result}
            self.collector.emit(
                "tool_result", parent=req.event_id, agent_id=self.agent_id, tool_name=name,
                status="ok", result=result, duration_ms=_ms(start),
            )
        except ToolError as e:
            if e.injected:
                meta["injected_fault"] = True
            payload = {"status": "error", "error": {"code": e.code, "message": e.message}}
            self.collector.emit(
                "tool_result", parent=req.event_id, agent_id=self.agent_id, tool_name=name,
                status="error", error=payload["error"], duration_ms=_ms(start), meta=meta,
            )
        except Exception as e:  # unexpected tool bug, never crash the run
            payload = {"status": "error", "error": {"code": "TOOL_ERROR", "message": str(e)[:200]}}
            self.collector.emit(
                "tool_result", parent=req.event_id, agent_id=self.agent_id, tool_name=name,
                status="error", error=payload["error"], duration_ms=_ms(start),
            )
        return payload

    # ---- fault injection ----
    def _maybe_fault(self, name: str) -> None:
        cfg = self.fault_schedule.get(name)
        if cfg and self._calls[name] <= int(cfg.get("fail_first_n", 0)):
            raise ToolError(cfg.get("error_code", "TIMEOUT"), cfg.get("message", "Request timed out"), injected=True)

    # ---- path safety ----
    def _safe(self, base: Path, file_name: str, must_exist: bool) -> Path:
        if not file_name or "\x00" in file_name:
            raise ToolError("PATH_DENIED", "Invalid file name")
        p = (base / file_name).resolve()  # resolve() follows symlinks
        if not p.is_relative_to(base):
            raise ToolError("PATH_DENIED", "Path is outside the allowed directory")
        if must_exist and not p.is_file():
            raise ToolError("FILE_NOT_FOUND", f"File '{file_name}' not found")
        return p

    # ---- tools ----
    def _t_read_csv(self, a: ReadCsvArgs) -> dict:
        p = self._safe(self.input_dir, a.file_name, True)
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            cols = list(rows[0].keys()) if rows else []
        return {
            "columns": cols,
            "row_count": len(rows),
            "rows": rows[:MAX_PREVIEW_ROWS],
            "truncated": len(rows) > MAX_PREVIEW_ROWS,
        }

    def _t_aggregate_revenue(self, a: AggregateArgs) -> dict:
        p = self._safe(self.input_dir, a.file_name, True)
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            for c in (a.group_column, a.value_column):
                if c not in cols:
                    raise ToolError("UNKNOWN_COLUMN", f"Requested column {c} is absent")
            totals: dict[str, Decimal] = defaultdict(Decimal)
            for row in reader:
                try:
                    totals[row[a.group_column]] += Decimal(row[a.value_column])
                except (InvalidOperation, TypeError):
                    raise ToolError("BAD_VALUE", f"Non numeric value in column {a.value_column}")
        return {"groups": {k: float(v.quantize(Decimal("0.01"))) for k, v in sorted(totals.items())}}

    def _t_calculate(self, a: CalcArgs) -> dict:
        return {"value": safe_eval(a.expression)}

    def _t_write_report(self, a: WriteReportArgs) -> dict:
        if Path(a.file_name).suffix.lower() not in ALLOWED_WRITE_SUFFIXES:
            raise ToolError("PATH_DENIED", "File type not allowed")
        if len(a.content.encode("utf-8")) > MAX_WRITE_BYTES:
            raise ToolError("TOOL_ERROR", "Content too large")
        p = self._safe(self.output_dir, a.file_name, False)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(a.content, encoding="utf-8")
        return {"written": a.file_name, "bytes": len(a.content.encode("utf-8"))}

    def _t_mock_api_request(self, a: MockApiArgs) -> dict:
        if a.endpoint not in MOCK_ENDPOINTS:
            raise ToolError("NOT_FOUND", f"Endpoint {a.endpoint} not found (404)")
        return {"endpoint": a.endpoint, "data": MOCK_ENDPOINTS[a.endpoint]}


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
