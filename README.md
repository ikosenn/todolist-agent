### Todolist Agent

This project is an AI-powered conversational todo assistant built with LangChain and LangGraph. It maintains a running conversation with a user, learns about their preferences and profile over time, and keeps their todo list up to date automatically based on natural language instructions.

The agent is designed to:
- **Track user profile information** (e.g., preferences, habits).
- **Maintain a structured todo list** extracted from free-form conversation.
- **Adapt instructions for managing todos** based on how the user likes to work.
- **Persist both short‑term and long‑term memory** in a Postgres database so that context is preserved across sessions.

The main entrypoint is `todolist_agent/main.py`, which starts an interactive CLI loop where you can chat with the agent.

---

### Tools and Technologies Used

- **Python**: Core language for the agent and application code.
- **LangChain / LangChain Core**: Provides LLM orchestration and chat model initialization.
- **LangGraph**: Used to build the stateful agent graph, manage tool-calling flows, and stream responses.
- **Trustcall**: Handles structured extraction of profile and todo information from conversations.
- **Postgres (via `langgraph-checkpoint-postgres` and `langgraph-store-postgres`)**:
  - **Short‑term memory** (graph checkpoints) is stored using `AsyncPostgresSaver`.
  - **Long‑term memory** (profile, todos, and instructions) is stored using `AsyncPostgresStore`.
  - This means all conversational context and memory is persisted in Postgres and can be reused across runs.
- **psycopg**: Postgres driver used under the hood by the Postgres integrations.
- **Docker / Docker Compose**: Used to run a Postgres instance for local development.
- **Ruff**: Linting and formatting (see `Makefile` targets `format` and `lint`).
- **python-dotenv**: Loads configuration such as database URLs from a `.env` file.

---

### Running the Project

#### 1. Prerequisites

- **Python** `>= 3.13`
- **Docker** and **docker-compose**
- An OpenAI API key (or compatible endpoint) configured for the model name used in `main.py` (currently `gpt-5-nano`).

#### 2. Configure Environment Variables

Create your `.env` file from the provided example:

```bash
cp .env.example .env
```

Then edit `.env` and set:

- `OPENAI_API_KEY` to your API key.
- `POSTGRES_URL` to match your local Postgres instance (the default in `.env.example` assumes Postgres on `localhost`).

The same `POSTGRES_URL` value is used for both the **store** (long‑term memory) and the **checkpointer** (short‑term memory).

#### 3. Start Postgres with Docker Compose

Make sure you have a `docker-compose.yml` that defines a Postgres service (for example exposing port `5432`). Then start the database:

```bash
docker-compose up -d
```

Once Postgres is running, set the Postgres URL in your `.env` file. For example:

```env
POSTGRES_URL=postgresql://todolist_agent:todolist_agent@localhost:5432/todolist_agent
```

#### 4. Install Dependencies (with `uv sync`)

From the project root, using `uv`:

```bash
uv sync
```

#### 5. Initialize the Database

Run the database initialization script exposed as a project script in `pyproject.toml`:

```bash
uv run python -m todolist_agent.setup_db
```

#### 6. Run the Agent

Start the interactive CLI agent:

```bash
uv run python -m todolist_agent.main
```

You will be prompted to:
- Enter natural language messages for the agent.
- The agent will update your todo list, profile, and instructions, storing both short‑term and long‑term memory in Postgres so that subsequent interactions can build on past context.

To stop the application, press `Ctrl+C`.

---

### Development Helpers

- **Format and lint** the codebase with Ruff using the `Makefile`:

```bash
make format
make lint
```

# todolist-agent
Agents used to manage todolist
