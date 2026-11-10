import json
import shutil
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sample_data.make_all import generate_all  # noqa: E402


class ScriptedModel:
    """Fake chat model that replays a script of AIMessages."""

    def __init__(self, script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def tool_msg(name, args, id="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id, "type": "tool_call"}])


@pytest.fixture
def project(tmp_path):
    """A temp project root with tasks, generated data and hidden ground truth."""
    shutil.copytree(ROOT / "benchmark_tasks", tmp_path / "benchmark_tasks")
    truths = generate_all(tmp_path / "sample_data", tmp_path / "ground_truth")
    return tmp_path, truths["TASK-001"]
