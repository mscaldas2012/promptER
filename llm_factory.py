import ollama
import time
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import ClientSecretCredential
from openai import api_version, AzureOpenAI
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider

# Initialize OpenTelemetry
trace.set_tracer_provider(TracerProvider())
metrics.set_meter_provider(MeterProvider())
tracer = trace.get_tracer(__name__)

meter = metrics.get_meter(__name__)
llm_call_counter = meter.create_counter(
    "llm.calls",
    description="Number of LLM calls",
    unit="1"
)
llm_latency_histogram = meter.create_histogram(
    "llm.latency",
    description="LLM call latency",
    unit="ms"
)
llm_tokens_counter = meter.create_counter(
    "llm.tokens",
    description="Token usage",
    unit="1"
)


class LLMProvider:
    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        with tracer.start_as_current_span("ollama.chat") as span:
            start_time = time.time()
            span.set_attribute("llm.provider", "ollama")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.user_prompt", user_prompt)

            try:
                messages = []
                for role, content in roles.items():
                    if content:
                        messages.append({'role': role, 'content': content})
                messages.append({'role': 'user', 'content': user_prompt})

                response = ollama.chat(
                    model=model,
                    messages=messages,
                )
                latency = time.time() - start_time
                latency_ms = latency * 1000

                input_tokens = response.get('prompt_eval_count', 0)
                output_tokens = response.get('eval_count', 0)

                # Set span attributes
                span.set_attribute("llm.input_tokens", input_tokens)
                span.set_attribute("llm.output_tokens", output_tokens)
                span.set_attribute("llm.latency_ms", latency_ms)
                span.set_attribute("llm.response", response['message']['content'])

                # Record metrics
                llm_call_counter.add(1, {"provider": "ollama", "model": model, "outcome": "success"})
                llm_latency_histogram.record(latency_ms, {"provider": "ollama", "model": model})
                llm_tokens_counter.add(input_tokens, {"provider": "ollama", "model": model, "token_type": "input"})
                llm_tokens_counter.add(output_tokens, {"provider": "ollama", "model": model, "token_type": "output"})

                llm_logger.info(
                    "LLM call successful",
                    extra={
                        **log_extra,
                        "provider": "ollama",
                        "model_used": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency": latency,
                        "outcome": "success",
                        "input_text": user_prompt,
                        "output_text": response['message']['content'],
                    },
                )
                return response['message']['content']
            except Exception as e:
                latency = time.time() - start_time
                latency_ms = latency * 1000

                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.set_attribute("llm.error", str(e))
                span.set_attribute("llm.latency_ms", latency_ms)

                llm_call_counter.add(1, {"provider": "ollama", "model": model, "outcome": "error"})
                llm_latency_histogram.record(latency_ms, {"provider": "ollama", "model": model})

                llm_logger.error(
                    "LLM call failed",
                    extra={
                        **log_extra,
                        "provider": "ollama",
                        "model_used": model,
                        "latency": latency,
                        "outcome": "error",
                        "error_message": str(e),
                    },
                )
                raise e



class EDAVOpenAIProvider(LLMProvider):
    def __init__(self):
        TENANT_ID = os.getenv("EDAV_TENANT_ID")

        # From the customer SP
        CLIENT_ID = os.getenv("EDAV_CLIENT_ID")
        CLIENT_SECRET = os.getenv("EDAV_CLIENT_SECRET")

        # Authenticate using Consumer Dev SP credentials
        credential = ClientSecretCredential(
            tenant_id=TENANT_ID,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )

        # Retrieve the token using Provider SP as the scope
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
        with tracer.start_as_current_span("azure_openai.chat") as span:
            start_time = time.time()
            span.set_attribute("llm.provider", "azure_openai")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.user_prompt", user_prompt)

            try:
                messages = []
                for role, content in roles.items():
                    if content:
                        messages.append({'role': role, 'content': content})
                messages.append({'role': 'user', 'content': user_prompt})

                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                latency = time.time() - start_time
                latency_ms = latency * 1000

                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                response_text = response.choices[0].message.content

                # Set span attributes
                span.set_attribute("llm.input_tokens", input_tokens)
                span.set_attribute("llm.output_tokens", output_tokens)
                span.set_attribute("llm.latency_ms", latency_ms)
                span.set_attribute("llm.response", response_text)

                # Record metrics
                llm_call_counter.add(1, {"provider": "edav_openai", "model": model, "outcome": "success"})
                llm_latency_histogram.record(latency_ms, {"provider": "edav_openai", "model": model})
                llm_tokens_counter.add(input_tokens,
                                       {"provider": "edav_openai", "model": model, "token_type": "input"})
                llm_tokens_counter.add(output_tokens,
                                       {"provider": "edav_openai", "model": model, "token_type": "output"})

                llm_logger.info(
                    "LLM call successful",
                    extra={
                        **log_extra,
                        "provider": "edav_openai",
                        "model_used": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency": latency,
                        "outcome": "success",
                        "input_text": user_prompt,
                        "output_text": response_text,
                    },
                )
                return response_text
            except Exception as e:
                latency = time.time() - start_time
                latency_ms = latency * 1000

                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.set_attribute("llm.error", str(e))
                span.set_attribute("llm.latency_ms", latency_ms)

                llm_call_counter.add(1, {"provider": "edav_openai", "model": model, "outcome": "error"})
                llm_latency_histogram.record(latency_ms, {"provider": "edav_openai", "model": model})

                llm_logger.error(
                    "LLM call failed",
                    extra={
                        **log_extra,
                        "provider": "edav_openai",
                        "model_used": model,
                        "latency": latency,
                        "outcome": "error",
                        "error_message": str(e),
                    },
                )
                raise e


class AzureOpenAIProvider(LLMProvider):
    def __init__(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        self.client = ChatCompletionsClient(
            endpoint=f"{endpoint}/openai/deployments/gpt-4o/",
            api_version="2025-01-01-preview",
            credential=AzureKeyCredential(api_key)
        )


    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        with tracer.start_as_current_span("azure_openai.chat") as span:
            start_time = time.time()
            span.set_attribute("llm.provider", "azure_openai")
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.user_prompt", user_prompt)

            try:
                messages = []
                for role, content in roles.items():
                    if content:
                        messages.append({'role': role, 'content': content})
                messages.append({'role': 'user', 'content': user_prompt})

                response = self.client.complete(
                    model=model,
                    messages=messages,
                )
                latency = time.time() - start_time
                latency_ms = latency * 1000

                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                response_text = response.choices[0].message.content

                # Set span attributes
                span.set_attribute("llm.input_tokens", input_tokens)
                span.set_attribute("llm.output_tokens", output_tokens)
                span.set_attribute("llm.latency_ms", latency_ms)
                span.set_attribute("llm.response", response_text)

                # Record metrics
                llm_call_counter.add(1, {"provider": "azure_openai", "model": model, "outcome": "success"})
                llm_latency_histogram.record(latency_ms, {"provider": "azure_openai", "model": model})
                llm_tokens_counter.add(input_tokens,
                                       {"provider": "azure_openai", "model": model, "token_type": "input"})
                llm_tokens_counter.add(output_tokens,
                                       {"provider": "azure_openai", "model": model, "token_type": "output"})

                llm_logger.info(
                    "LLM call successful",
                    extra={
                        **log_extra,
                        "provider": "azure_openai",
                        "model_used": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency": latency,
                        "outcome": "success",
                        "input_text": user_prompt,
                        "output_text": response_text,
                    },
                )
                return response_text
            except Exception as e:
                latency = time.time() - start_time
                latency_ms = latency * 1000

                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.set_attribute("llm.error", str(e))
                span.set_attribute("llm.latency_ms", latency_ms)

                llm_call_counter.add(1, {"provider": "azure_openai", "model": model, "outcome": "error"})
                llm_latency_histogram.record(latency_ms, {"provider": "azure_openai", "model": model})

                llm_logger.error(
                    "LLM call failed",
                    extra={
                        **log_extra,
                        "provider": "azure_openai",
                        "model_used": model,
                        "latency": latency,
                        "outcome": "error",
                        "error_message": str(e),
                    },
                )
                raise e


class LLMProviderFactory:
    def get_provider(self, provider_name):
        if provider_name == "ollama":
            return OllamaProvider()
        elif provider_name == "azure_openai":
            return AzureOpenAIProvider()
        elif provider_name == "edav_openai":
            return EDAVOpenAIProvider()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
