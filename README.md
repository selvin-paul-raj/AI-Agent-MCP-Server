
# 🧠 Modular AI Agent MCP Server

A scalable, production-grade AI system that connects multiple dynamic agents into a single unified interface using **Model Context Protocol (MCP)**. Designed for extensibility, agent plug-in support, token tracking, cost monitoring, and fallback strategies.

---

## 📌 Features

- ✅ **Agent Router**: Dynamically route user queries to optimal agents.
- 🔁 **Callbacks & Fallbacks**: Retry logic, graceful degradation.
- 🎛️ **Plugin System**: Add/remove agents with zero downtime.
- 📦 **Prompt Engine**: Modular Jinja2 templates for flexible prompt generation.
- 📊 **Token & Cost Tracker**: Based on LangChain's `metadata` and `callbacks`.
- 📁 **Multi-format I/O**: YAML/JSON input/output support.
- ⚙️ **Agent Sandbox**: Secure, isolated execution layer.
- 📈 **Built-in Healthcheck & Validation**.
- ☁️ **FastAPI + Uvicorn**: High-performance HTTP server.

---

## 📐 Architecture

```



````

---

## 🛠️ Technologies Used

| Tool         | Purpose                                 |
|--------------|------------------------------------------|
| `Python`     | Core programming language                |
| `uv`         | Fast dependency installer                |
| `Gemini API` | AI model provider (via LangChain)        |
| `LangGraph`  | Agent orchestration                      |
| `LangChain`  | LLM wrapper, metadata & tracking         |
| `FastAPI`    | Web framework for MCP server             |
| `Uvicorn`    | ASGI server                              |
| `Typer`      | CLI for developer controls               |
| `Jinja2`     | Prompt templating                        |
| `Pydantic`   | Data modeling and validation             |
| `Multiprocessing` | Agent execution isolation         |
| `Tenacity`   | Retry logic for fallback and resilience  |
| `PyYAML`     | YAML input/output config files           |

---

## ⚙️ Installation

```bash
uv pip install -r requirements.txt
````

> **Note**: Make sure your `.env` includes Gemini API Key and any environment agent configs.

---

## 🚀 Usage

```bash
python main.py --prompt "Find latest update about Company X" --agent agent2
```

Or use CLI interactive:

```bash
python cli.py
```

---

## 🧩 Add New Agent

1. Create a new agent file in `agents/agent_x.py`
2. Register in `plugin_loader.py`
3. Add dynamic prompt template in `prompts/agent_x.jinja`
4. Done ✅

---

## 🧪 Token + Cost Tracking

Powered by **LangChain metadata** & custom callback handlers:

* Tracks input/output token usage per agent
* Logs estimated cost based on Gemini pricing
* Supports YAML/JSON reports per run

---

## 🛡️ Fallback & Retry Strategy

* Built-in retry (3x by default) using `tenacity`
* Optional fallback agents (`agent_backup`)
* Custom exceptions and timeout handling

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 🧱 Folder Structure

```
.
├── agents/
│   ├── agent1.py
│   ├── agent2.py
│   └── agentN.py
├── prompts/
│   ├── agent1.jinja
│   └── ...
├── core/
│   ├── router.py
│   ├── fallback.py
│   ├── plugin_loader.py
│   ├── context_manager.py
│   └── token_tracker.py
├── schemas/
│   └── types.py
├── config/
│   ├── settings.yaml
│   └── prompts.yaml
├── cli.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 📤 Output Example

```json
{
  "agent": "agent2",
  "task": "Summarization",
  "tokens_used": 215,
  "cost_estimate_usd": 0.00012,
  "output": "Company X plans to expand its AI division in 2025..."
}
```

---

## 📣 Contributing

Feel free to fork, extend new agents, or raise issues!

---

## 📄 License

[MIT License](LICENSE)

---
