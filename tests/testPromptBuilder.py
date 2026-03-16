"""
Integration test: run the meta-prompt-generator against the openai_compatible provider.

Uses a real Langfuse client when credentials are available (so traces appear in Langfuse),
and falls back to a mock when they are not set.

Requires OPENAI_COMPATIBLE_BASE_URL (and optionally OPENAI_COMPATIBLE_API_KEY) in .env.

Usage:
    python tests/testPromptBuilder.py
"""

import json
import os
import re
import sys
from unittest.mock import MagicMock

# Allow running from repo root or from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from llm_factory import OpenAICompatibleProvider
from logging_config import llm_logger
from prompt_loader import load_prompt_config

PROVIDER = "openai_compatible"
MODEL = "qwen3.5-9b-optiq"
PROMPT_USE_CASE = "meta-prompt-generator"
PROMPT_MODEL_NAME = "default"

SAMPLE_PROMPT = (
    "You are a helpful assistant. Answer user questions clearly and concisely."
)


def _make_langfuse_client():
    """Return a real Langfuse client if credentials are available, else a silent mock."""
    try:
        from langfuse import Langfuse
        client = Langfuse()
        client.auth_check()
        print("✓ Langfuse client authenticated — traces will appear in Langfuse")
        return client
    except Exception as e:
        print(f"⚠ Langfuse unavailable ({e}) — traces will not be recorded")
        mock = MagicMock()
        mock.start_generation.return_value = MagicMock()
        return mock


def apply_framework_substitutions(prompt_roles: dict) -> dict:
    """Replace framework placeholders with free-form defaults, matching main_page logic."""
    framework_note = "Free Form / No Specific Framework"
    replacements = {
        "{{framework}}": framework_note,
        "{{framework_label}}": framework_note,
        "{{framework_description}}": "",
        "{{framework_instructions}}": "",
        "{{framework_key}}": "free_form",
    }
    updated = {}
    for role, content in prompt_roles.items():
        if isinstance(content, str):
            for placeholder, value in replacements.items():
                content = content.replace(placeholder, value)
        updated[role] = content
    return updated


def run_prompt_builder_test():
    print(f"Provider : {PROVIDER}")
    print(f"Model    : {MODEL}")
    print(f"Use case : {PROMPT_USE_CASE}")
    print()

    # --- load prompt config (Langfuse → config.yaml fallback) ---
    prompt_config = load_prompt_config(PROMPT_USE_CASE)
    assert prompt_config, f"Prompt config '{PROMPT_USE_CASE}' not found in Langfuse or config.yaml"

    prompt_model_config = prompt_config["models"][PROMPT_MODEL_NAME]
    prompt_roles = apply_framework_substitutions(
        dict(prompt_model_config.get("prompt_roles", {}))
    )

    log_extra = {
        "use_case": PROMPT_USE_CASE,
        "prompt_id": prompt_config.get("id", PROMPT_USE_CASE),
        "prompt_version": prompt_config.get("version", "unknown"),
        "prompt_framework": "Free Form / No Specific Framework",
        "model_params": prompt_model_config.get("model_params", {}),
    }

    # --- call the model ---
    langfuse_client = _make_langfuse_client()
    provider = OpenAICompatibleProvider(langfuse_client)

    raw = provider.get_llm_response(SAMPLE_PROMPT, MODEL, prompt_roles, llm_logger, log_extra)

    # shutdown() blocks until all queued events are sent — needed for short-lived scripts
    # where flush() would return before the HTTP send completes.
    langfuse_client.shutdown()

    # --- assertions ---
    assert isinstance(raw, str) and raw.strip(), "Response must be a non-empty string"
    print("✓ Response is a non-empty string")

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    assert json_match, f"Response does not contain a JSON object.\nRaw:\n{raw}"
    print("✓ Response contains a JSON object")

    parsed = json.loads(json_match.group())
    expected_keys = {"review_comments", "suggested_improvements", "revised_prompt"}
    missing = expected_keys - parsed.keys()
    assert not missing, (
        f"Response JSON missing expected keys: {missing}\n"
        f"Keys found: {sorted(parsed.keys())}"
    )
    print(f"✓ All required keys present: {sorted(parsed.keys())}")

    # --- print summary ---
    print("\n--- review_comments ---")
    print(parsed["review_comments"])
    print("\n--- suggested_improvements ---")
    print(parsed["suggested_improvements"])
    print("\n--- revised_prompt (first 400 chars) ---")
    print(str(parsed["revised_prompt"])[:400])

    return parsed


if __name__ == "__main__":
    result = run_prompt_builder_test()
    print("\n✓ Test passed.")
