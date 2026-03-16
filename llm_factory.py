import os

import ollama
from azure.identity import ClientSecretCredential
from langfuse import Langfuse
from openai import AzureOpenAI, OpenAI


class LLMProvider:
    def __init__(self, langfuse_client):
        self.langfuse = langfuse_client

    def _build_messages(self, user_prompt, roles):
        messages = []
        assistant_prefill = None
        for role, content in roles.items():
            if not content:
                continue
            if role == 'assistant':
                assistant_prefill = content  # defer until after user message
            else:
                messages.append({'role': role, 'content': content})
        messages.append({'role': 'user', 'content': user_prompt})
        if assistant_prefill:
            messages.append({'role': 'assistant', 'content': assistant_prefill})
        return messages

    def _traced_call(self, provider_name, model, user_prompt, messages, log_extra, llm_logger, call_fn):
        """
        call_fn(messages) -> (response_text, input_tokens, output_tokens)
        Creates a Langfuse span (trace root) with a nested generation (the LLM call).
        Uses span.start_observation(as_type='generation') — the non-deprecated v3 API.
        Langfuse v3 requires a root span to produce a visible trace; a root-level
        generation alone does not surface as a trace in the Langfuse UI.
        """
        version = log_extra.get("prompt_version")

        span = self.langfuse.start_span(
            name=log_extra.get("use_case", f"{provider_name}-trace"),
            input={"user_input": user_prompt},
            metadata=log_extra,
            version=version,
        )
        generation = span.start_observation(
            as_type="generation",
            name=f"{provider_name}-generation",
            input=messages,
            model=model,
            metadata=log_extra,
            version=version,
            model_parameters=log_extra.get("model_params"),
        )

        try:
            response_text, input_tokens, output_tokens = call_fn(messages)

            generation.update(
                output=response_text,
                usage_details={"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
            )
            generation.end()
            span.update(output={"result": response_text})
            span.end()
            self.langfuse.flush()

            llm_logger.info("LLM call successful", extra={
                **log_extra,
                "provider": provider_name,
                "model_used": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "outcome": "success",
                "input_text": user_prompt,
                "output_text": response_text,
            })
            return response_text
        except Exception as e:
            generation.update(level="ERROR", status_message=str(e))
            generation.end()
            span.end()
            self.langfuse.flush()
            llm_logger.error("LLM call failed", extra={
                **log_extra,
                "provider": provider_name,
                "model_used": model,
                "outcome": "error",
                "error_message": str(e),
            })
            raise

    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        messages = self._build_messages(user_prompt, roles)
        def call_fn(msgs):
            r = ollama.chat(model=model, messages=msgs)
            msg = r.get('message') or {}
            return msg.get('content', ''), r.get('prompt_eval_count', 0), r.get('eval_count', 0)
        return self._traced_call("ollama", model, user_prompt, messages, log_extra, llm_logger, call_fn)


class EDAVOpenAIProvider(LLMProvider):
    def __init__(self, langfuse_client):
        super().__init__(langfuse_client)
        TENANT_ID = os.getenv("EDAV_TENANT_ID")
        CLIENT_ID = os.getenv("EDAV_CLIENT_ID")
        CLIENT_SECRET = os.getenv("EDAV_CLIENT_SECRET")
        credential = ClientSecretCredential(
            tenant_id=TENANT_ID,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
        scope = os.getenv("EDAV_SCOPE_TOKEN_AUDIENCE")
        if not scope:
            raise ValueError("EDAV_SCOPE_TOKEN_AUDIENCE is missing; cannot request an access token.")
        try:
            token = credential.get_token(scope).token
            print(f"Token with scope {scope} acquired successfully: {token[:25]}....")
            MAAS_SUBSCRIPTION_KEY = os.getenv("EDAV_SUBSCRIPTION_KEY")
            self.client = AzureOpenAI(
                api_version=os.getenv("EDAV_AZURE_OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("EDAV_AZURE_OPENAI_ENDPOINT"),
                azure_ad_token=token,
                default_headers={"Ocp-Apim-Subscription-Key": MAAS_SUBSCRIPTION_KEY},
            )
        except Exception as e:
            raise RuntimeError(f"Error acquiring token for scope {scope}: {e}") from e

    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        messages = self._build_messages(user_prompt, roles)
        def call_fn(msgs):
            r = self.client.chat.completions.create(model=model, messages=msgs)
            return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens
        return self._traced_call("edav_openai", model, user_prompt, messages, log_extra, llm_logger, call_fn)


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, langfuse_client):
        super().__init__(langfuse_client)
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )

    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        messages = self._build_messages(user_prompt, roles)
        def call_fn(msgs):
            r = self.client.chat.completions.create(model=model, messages=msgs)
            return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens
        return self._traced_call("azure_openai", model, user_prompt, messages, log_extra, llm_logger, call_fn)


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible API (LMStudio, vLLM, etc.)."""
    def __init__(self, langfuse_client):
        super().__init__(langfuse_client)
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "lm-studio"),
        )

    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        messages = self._build_messages(user_prompt, roles)
        def call_fn(msgs):
            r = self.client.chat.completions.create(model=model, messages=msgs)
            input_tokens = r.usage.prompt_tokens if r.usage else 0
            output_tokens = r.usage.completion_tokens if r.usage else 0
            return r.choices[0].message.content, input_tokens, output_tokens
        return self._traced_call("openai_compatible", model, user_prompt, messages, log_extra, llm_logger, call_fn)


class LLMProviderFactory:
    def __init__(self):
        self.langfuse = Langfuse()

    def get_provider(self, provider_name):
        if provider_name == "ollama":
            return OllamaProvider(self.langfuse)
        elif provider_name == "azure_openai":
            return AzureOpenAIProvider(self.langfuse)
        elif provider_name == "edav_openai":
            return EDAVOpenAIProvider(self.langfuse)
        elif provider_name == "openai_compatible":
            return OpenAICompatibleProvider(self.langfuse)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
