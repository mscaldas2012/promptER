# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then fill in credentials
```

**Run:**
```bash
streamlit run app.py
```

**Test:**
```bash
python tests/testTelemetry.py
```

## Architecture

**promptER** is a Streamlit multi-page app for AI prompt engineering. Three pages map to three workflows:

- `main_page.py` — **Prompt Refiner**: user pastes a raw prompt, selects a framework, and the LLM returns structured JSON with `review_comments`, `suggested_improvements`, and `revised_prompt`
- `chat_page.py` — **Playground**: interactive chat to test a (possibly pre-filled) system prompt
- `evaluation_page.py` — **LLM-as-a-Judge**: scores an assistant response on structure, content, and tone using another LLM call

`app.py` is the entry point; it loads `.env`, manages `st.session_state` for page routing, and renders the sidebar navigation.

### LLM Provider System (`llm_factory.py`)

Factory + Strategy pattern:
- `LLMProvider` (abstract base) → `OllamaProvider`, `AzureOpenAIProvider`, `EDAVOpenAIProvider`
- `LLMProviderFactory.get_provider(name)` returns the right concrete provider
- All providers implement `get_llm_response(messages, model_params)` and wrap calls with Langfuse spans + structured JSON logs to `llm_calls.log`

### Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Versioned prompt configs (system/assistant role content, model params) for the refiner and evaluator |
| `providers.json` | Default provider and model lists for each backend |
| `framework_options.json` | Prompt framework definitions (RTF, PECRA, OSCAR, CRISP, TAG, etc.) injected into refiner prompts |
| `.env` | Credentials for Azure OpenAI, EDAV OpenAI, and Langfuse |

### Observability (`logging_config.py`)

Three-layer approach: structured JSON logs (`llm_calls.log`) → Langfuse cloud traces → OpenTelemetry spans. The custom `OtelSpanEventHandler` bridges Python log records to OTEL spans so extra fields (tokens, latency, model, prompt version) flow into both sinks automatically.

### Adding a New LLM Provider

1. Subclass `LLMProvider` in `llm_factory.py`
2. Add its credentials to `.env`
3. Register it in `LLMProviderFactory.get_provider()`
4. Add model list to `providers.json`

### Adding a New Prompt Framework

Add an entry to `framework_options.json` — the refiner page picks it up automatically and injects the `instructions` field into the system prompt.
