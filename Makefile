.PHONY: format
format:
	ruff check todolist_agent --fix
	ruff format todolist_agent

.PHONY: lint
lint:
	ruff check todolist_agent
	ruff format todolist_agent --check
