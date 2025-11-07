
import ollama
import openai
import time
import os

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
    def __init__(self):
        self.client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

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
