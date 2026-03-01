# Airline Customer Support Agent

This repository is an airline customer support agent project.
It supports MCP tool calling.
It includes both Python and TypeScript implementations.

## Directory Overview

- `python/agent/` Main agent program with CLI and Web UI
- `python/mcp_airline/` MCP service for airline domain tasks
- `data/` Airline data and policy data

## Setup

1. Prepare Python 3.11 or higher
2. Install dependencies
3. Set your model API key

Example command:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Start the MCP airline service
2. Start the agent with CLI or Web UI
3. Enable RAG mode when needed

Example commands:

```bash
python ingest_policy.py
agent-cli --rag http://localhost:3000/mcp
```

## Notes

- Do not upload sensitive files such as `.env`
- This repository is for course project and experiments
