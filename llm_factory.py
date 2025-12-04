
import ollama
import openai
import time
import os
import threading
import json

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

class LLMProvider:
    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        raise NotImplementedError

class OllamaProvider(LLMProvider):
    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        start_time = time.time()
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
            llm_logger.info(
                "LLM call successful",
                extra={
                    **log_extra,
                    "provider": "ollama",
                    "model_used": model,
                    "input_tokens": response.get('prompt_eval_count'),
                    "output_tokens": response.get('eval_count'),
                    "latency": latency,
                    "outcome": "success",
                    "input_text": user_prompt,
                    "output_text": response['message']['content'],
                },
            )
            return response['message']['content']
        except Exception as e:
            latency = time.time() - start_time
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

class AzureOpenAIProvider(LLMProvider):
    _instrumentation_lock = threading.Lock()
    _instrumented = False

    def __init__(self):
        with self._instrumentation_lock:
            if not self._instrumented:
                self._instrument_openai()
                AzureOpenAIProvider._instrumented = True

        # Configure the Azure exporter in a background thread to avoid blocking.
        telemetry_thread = threading.Thread(target=self._configure_exporter)
        telemetry_thread.daemon = True
        telemetry_thread.start()

        self.client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

    def _instrument_openai(self):
        """
        Applies OpenTelemetry instrumentation for OpenAI, capturing input and output.
        This method is synchronous and should be called before any OpenAI clients are created.
        """
        print("Applying OpenAI instrumentation hooks.")
        def request_hook(span, request_kwargs):
            if span and span.is_recording():
                messages = request_kwargs.get("messages", [])
                span.set_attribute("llm.input_messages", json.dumps(messages))

        def response_hook(span, request_kwargs, response):
            if span and span.is_recording():
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content
                    span.set_attribute("llm.output_message", content)

        OpenAIInstrumentor().instrument(
            request_hook=request_hook,
            response_hook=response_hook
        )

    def _configure_exporter(self):
        """
        Configures and attaches the Azure Monitor exporter to the global TracerProvider.
        This runs in a background thread to avoid blocking application startup.
        """
        try:
            connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
            if connection_string:
                print("Configuring Azure Monitor exporter in the background...")
                
                provider = trace.get_tracer_provider()
                
                # If no TracerProvider is configured, set one up.
                if isinstance(provider, trace.ProxyTracerProvider):
                    provider = TracerProvider()
                    trace.set_tracer_provider(provider)
                    print("Initialized new TracerProvider.")

                exporter = AzureMonitorTraceExporter.from_connection_string(connection_string)
                span_processor = BatchSpanProcessor(exporter)
                provider.add_span_processor(span_processor)
                print("Azure Monitor exporter added successfully.")
            else:
                print("APPLICATIONINSIGHTS_CONNECTION_STRING not set. Skipping exporter configuration.")
        except Exception as e:
            print(f"Error configuring Azure Monitor exporter in the background: {e}")

    def get_llm_response(self, user_prompt, model, roles, llm_logger, log_extra):
        start_time = time.time()
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
            llm_logger.info(
                "LLM call successful",
                extra={
                    **log_extra,
                    "provider": "azure_openai",
                    "model_used": model,
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "latency": latency,
                    "outcome": "success",
                    "input_text": user_prompt,
                    "output_text": response.choices[0].message.content,
                },
            )

            return response.choices[0].message.content
        except Exception as e:
            latency = time.time() - start_time
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
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
