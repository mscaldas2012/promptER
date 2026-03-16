import os
import yaml
import json
import logging
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def _try_langfuse_fetch(langfuse_client, lf_name):
    if not langfuse_client or not lf_name:
        return None

    try:
        # many Langfuse SDKs expose a get_prompt(name, label=...) helper
        return langfuse_client.get_prompt(lf_name)
    except Exception as e:
        # Inspect common HTTP/SDK error shapes to give helpful messages
        msg = str(e)
        status = getattr(e, "status_code", None)
        if status in (401, 403):
            logger.debug("Langfuse prompt '%s' not accessible (status %s) — falling back to config.yaml", lf_name, status)
            return None
        if status == 404 or "404" in msg or "not found" in msg.lower():
            logger.debug("Langfuse prompt '%s' not found (404) — falling back to config.yaml", lf_name)
            return None
        if "401" in msg or "Unauthorized" in msg or "Invalid credentials" in msg:
            logger.debug("Langfuse prompt '%s' not accessible (auth) — falling back to config.yaml", lf_name)
            return None
        # Other errors — re-raise so caller can decide
        raise


def json_safe_str(x):
    try:
        return json.dumps(x)
    except Exception:
        return str(x)


def load_prompt_config(prompt_key):
    """Load a prompt configuration by local prompt_key.

    Attempt to load from Langfuse (using mapped names). If that fails
    (including auth errors), fall back to reading `config.yaml` from
    the repo root. Returns the prompt config dict or None if not found.
    """
    lf_name = prompt_key

    # Try Langfuse client
    langfuse_client = None
    try:
        langfuse_client = Langfuse()
    except Exception as e:
        logger.debug("Langfuse client init failed: %s", e)

    if lf_name and langfuse_client:
        try:
            fetched = _try_langfuse_fetch(langfuse_client, lf_name)
            if fetched:
                logger.info("Loaded prompt '%s' from Langfuse.", lf_name)
                # Normalize Langfuse prompt shape into the config.yaml-like structure.
                # `fetched` may be a Langfuse SDK object (TextPromptClient / ChatPromptClient)
                # or a plain dict depending on SDK version — handle both.
                data = fetched

                def _get(obj, *keys, default=None):
                    """Retrieve a value by key from a dict or attribute from an object."""
                    for k in keys:
                        try:
                            val = obj[k] if isinstance(obj, dict) else getattr(obj, k, None)
                            if val is not None and val != "":
                                return val
                        except (KeyError, TypeError):
                            pass
                    return default

                # unwrap common dict wrapper {"data": {...}}
                if isinstance(data, dict) and isinstance(data.get("data"), dict):
                    data = data["data"]

                # If there's already a prompt_roles block, use that directly
                prompt_roles_raw = _get(data, "prompt_roles")
                if isinstance(prompt_roles_raw, dict):
                    prompt_roles = prompt_roles_raw
                else:
                    # Langfuse v3 TextPromptClient: .prompt is the raw string; v3 ChatPromptClient: list
                    raw_prompt = _get(data, "prompt", "system", "instructions", "content", "template", "text", default="")
                    raw_assistant = _get(data, "assistant", "assistant_instructions", "output_format", default="")
                    system_text = raw_prompt if isinstance(raw_prompt, str) else json_safe_str(raw_prompt)
                    assistant_text = raw_assistant if isinstance(raw_assistant, str) else json_safe_str(raw_assistant)
                    prompt_roles = {"system": system_text, "assistant": assistant_text}

                # Langfuse v3 .config dict may carry model_params
                extra_config = _get(data, "config", default={})
                model_params = (extra_config or {}).get("model_params", {}) if isinstance(extra_config, dict) else {}

                cfg = {
                    "id": _get(data, "id", "name", default=lf_name),
                    "version": _get(data, "version", "label", default="unknown"),
                    "purpose": _get(data, "purpose", "description", default=""),
                    "models": {
                        "default": {
                            "prompt_roles": prompt_roles,
                            "model_params": model_params,
                        }
                    }
                }
                return cfg
        except Exception as e:
            logger.debug("Langfuse fetch raised: %s", e)

    # Fallback to config.yaml
    try:
        here = os.getcwd()
        cfg_path = os.path.join(here, "config.yaml")
        with open(cfg_path) as config_file:
            data = yaml.safe_load(config_file) or {}
            prompts = data.get("prompts", {})
            prompt_cfg = prompts.get(prompt_key)
            if prompt_cfg:
                logger.info("Loaded prompt '%s' from config.yaml.", prompt_key)
                return prompt_cfg
    except Exception as e:
        logger.debug("Failed to load config.yaml fallback: %s", e)

    logger.warning("Prompt '%s' not found in Langfuse or config.yaml.", prompt_key)
    return None

