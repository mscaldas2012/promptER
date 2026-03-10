import os

import ollama
from azure.identity import ClientSecretCredential
from langfuse import Langfuse
from openai import AzureOpenAI, OpenAI


class LLMProvider:
    def __init__(self, langfuse_client):
        self.langfuse = langfuse_client

    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        try:
            messages = []
            for role, content in roles.items():
                if content:
                    messages.append({'role': role, 'content': content})
            messages.append({'role': 'user', 'content': user_prompt})

            version = log_extra.get("prompt_version")
            model_params = log_extra.get("model_params")

            langfuse_span = self.langfuse.start_span(
                name=log_extra.get("use_case", "ollama-trace"),
                input={"user_input": user_prompt},
                metadata=log_extra,
                version=version
            )

            langfuse_generation = langfuse_span.start_generation(
                name="ollama-generation",
                input=messages,
                model=model,
                metadata=log_extra,
                version=version,
                model_parameters=model_params
            )

            response = ollama.chat(
                model=model,
                messages=messages,
            )

            input_tokens = response.get('prompt_eval_count', 0)
            output_tokens = response.get('eval_count', 0)
            response_text = response['message']['content']

            langfuse_generation.update(
                output=response_text,
                usage_details={"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
            )
            langfuse_generation.end()
            
            langfuse_span.update(output={"result": response_text})
            langfuse_span.end()
            
            self.langfuse.flush()

            llm_logger.info(
                "LLM call successful",
                extra={
                    **log_extra,
                    "provider": "ollama",
                    "model_used": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "outcome": "success",
                    "input_text": user_prompt,
                    "output_text": response_text,
                },
            )
            return response_text
        except Exception as e:
            llm_logger.error(
                "LLM call failed",
                extra={
                    **log_extra,
                    "provider": "ollama",
                    "model_used": model,
                    "outcome": "error",
                    "error_message": str(e),
                },
            )
            raise e



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
        try:
            messages = []
            for role, content in roles.items():
                if content:
                    messages.append({'role': role, 'content': content})
            messages.append({'role': 'user', 'content': user_prompt})

            version = log_extra.get("prompt_version")
            model_params = log_extra.get("model_params")

            langfuse_span = self.langfuse.start_span(
                name=log_extra.get("use_case", "edav-openai-trace"),
                input={"user_input": user_prompt},
                metadata=log_extra,
                version=version
            )

            langfuse_generation = langfuse_span.start_generation(
                name="edav-openai-generation",
                input=messages,
                model=model,
                metadata=log_extra,
                version=version,
                model_parameters=model_params
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            response_text = response.choices[0].message.content

            langfuse_generation.update(
                output=response_text,
                usage_details={"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
            )
            langfuse_generation.end()

            langfuse_span.update(output={"result": response_text})
            langfuse_span.end()
            
            self.langfuse.flush()

            llm_logger.info(
                "LLM call successful",
                extra={
                    **log_extra,
                    "provider": "edav_openai",
                    "model_used": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "outcome": "success",
                    "input_text": user_prompt,
                    "output_text": response_text,
                },
            )
            return response_text
        except Exception as e:
            llm_logger.error(
                "LLM call failed",
                extra={
                    **log_extra,
                    "provider": "edav_openai",
                    "model_used": model,
                    "outcome": "error",
                    "error_message": str(e),
                },
            )
            raise e


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, langfuse_client):
        super().__init__(langfuse_client)
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        )

    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        try:
            messages = []
            for role, content in roles.items():
                if content:
                    messages.append({'role': role, 'content': content})
            messages.append({'role': 'user', 'content': user_prompt})

            version = log_extra.get("prompt_version")
            model_params = log_extra.get("model_params")

            langfuse_span = self.langfuse.start_span(
                name=log_extra.get("use_case", "azure-openai-trace"),
                input={"user_input": user_prompt},
                metadata=log_extra,
                version=version
            )

            langfuse_generation = langfuse_span.start_generation(
                name="azure-openai-generation",
                input=messages,
                model=model,
                metadata=log_extra,
                version=version,
                model_parameters=model_params
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            response_text = response.choices[0].message.content

            langfuse_generation.update(
                output=response_text,
                usage_details={"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
            )
            langfuse_generation.end()

            langfuse_span.update(output={"result": response_text})
            langfuse_span.end()
            
            self.langfuse.flush()

            llm_logger.info(
                "LLM call successful",
                extra={
                    **log_extra,
                    "provider": "azure_openai",
                    "model_used": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "outcome": "success",
                    "input_text": user_prompt,
                    "output_text": response_text,
                },
            )
            return response_text
        except Exception as e:
            llm_logger.error(
                "LLM call failed",
                extra={
                    **log_extra,
                    "provider": "azure_openai",
                    "model_used": model,
                    "outcome": "error",
                    "error_message": str(e),
                },
            )
            raise e


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible API (LMStudio, vLLM, etc.)."""
    def __init__(self, langfuse_client):
        super().__init__(langfuse_client)
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "lm-studio"),
        )

    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        try:
            messages = []
            for role, content in roles.items():
                if content:
                    messages.append({'role': role, 'content': content})
            messages.append({'role': 'user', 'content': user_prompt})

            version = log_extra.get("prompt_version")
            model_params = log_extra.get("model_params")

            langfuse_span = self.langfuse.start_span(
                name=log_extra.get("use_case", "openai-compatible-trace"),
                input={"user_input": user_prompt},
                metadata=log_extra,
                version=version
            )

            langfuse_generation = langfuse_span.start_generation(
                name="openai-compatible-generation",
                input=messages,
                model=model,
                metadata=log_extra,
                version=version,
                model_parameters=model_params
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            response_text = response.choices[0].message.content

            langfuse_generation.update(
                output=response_text,
                usage_details={"prompt_tokens": input_tokens, "completion_tokens": output_tokens}
            )
            langfuse_generation.end()

            langfuse_span.update(output={"result": response_text})
            langfuse_span.end()

            self.langfuse.flush()

            llm_logger.info(
                "LLM call successful",
                extra={
                    **log_extra,
                    "provider": "openai_compatible",
                    "model_used": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "outcome": "success",
                    "input_text": user_prompt,
                    "output_text": response_text,
                },
            )
            return response_text
        except Exception as e:
            llm_logger.error(
                "LLM call failed",
                extra={
                    **log_extra,
                    "provider": "openai_compatible",
                    "model_used": model,
                    "outcome": "error",
                    "error_message": str(e),
                },
            )
            raise e


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
