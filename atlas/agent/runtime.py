
"""Persistent LangGraph runtime for local development."""

import os
import sqlite3
from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from atlas.agent.graph import build_graph


@lru_cache(maxsize=1)
def get_agent():
    data_dir = Path(os.getenv("ATLAS_DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(data_dir / "agent_checkpoints.sqlite"),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    return build_graph(checkpointer)
