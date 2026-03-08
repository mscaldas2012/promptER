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
        if hasattr(e, "status_code") and e.status_code in (401, 403):
            logger.warning("Langfuse auth error (status %s) when fetching '%s': %s", e.status_code, lf_name, msg)
            return None
        if "401" in msg or "Unauthorized" in msg or "Invalid credentials" in msg:
            logger.warning("Langfuse auth issue when fetching '%s': %s", lf_name, msg)
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
                # Normalize Langfuse prompt shape into the config.yaml-like structure
                data = fetched
                # unwrap common wrapper
                if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
                    data = data["data"]

                # Try to find system/assistant content
                def _pick(*keys):
                    for k in keys:
                        if isinstance(data, dict) and k in data and data[k]:
                            return data[k]
                    return ""

                # If there's already a prompt_roles block, use that directly
                if isinstance(data, dict) and "prompt_roles" in data and isinstance(data["prompt_roles"], dict):
                    prompt_roles = data["prompt_roles"]
                else:
                    # Common places where a Langfuse prompt stores main text
                    possible_system = _pick("system", "instructions", "prompt", "content", "template", "text")
                    possible_assistant = _pick("assistant", "assistant_instructions", "output_format")
                    system_text = possible_system if isinstance(possible_system, str) else json_safe_str(possible_system)
                    assistant_text = possible_assistant if isinstance(possible_assistant, str) else json_safe_str(possible_assistant)
                    prompt_roles = {"system": system_text, "assistant": assistant_text}

                # Build a config structure similar to config.yaml
                cfg = {
                    "id": data.get("id") or data.get("name") or lf_name,
                    "version": data.get("version") or data.get("label") or "unknown",
                    "purpose": data.get("purpose") or data.get("description") or "",
                    "models": {
                        "default": {
                            "prompt_roles": prompt_roles,
                            "model_params": data.get("model_params", {}),
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

