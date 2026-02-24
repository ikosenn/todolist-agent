"""Initialize the database for LangGraph checkpoint and store using PostgreSQL.

Run once before first use to create required tables. Uses POSTGRES_URL from the
environment (or from a .env file in the current directory if present).
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore


def main() -> int:
    """Create LangGraph checkpoint and store tables in PostgreSQL."""
    load_dotenv()
    url = os.environ.get("POSTGRES_URL")
    if not url:
        print(
            "POSTGRES_URL is not set. Set it in the environment or in a .env file.",
            file=sys.stderr,
        )
        return 1

    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    try:
        with PostgresSaver.from_conn_string(url) as checkpointer:
            checkpointer.setup()
        with PostgresStore.from_conn_string(url) as store:
            store.setup()
    except Exception as e:
        print(f"Database setup failed: {e}", file=sys.stderr)
        return 1

    print("Database initialized successfully (LangGraph checkpoint and store tables).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
