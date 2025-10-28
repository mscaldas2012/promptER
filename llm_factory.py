
import ollama
import openai
import time
import os

class LLMProvider:
    def get_refined_prompt(self, initial_prompt, model, system_prompt_content, llm_logger, log_extra):
        raise NotImplementedError

class OllamaProvider(LLMProvider):
    def get_refined_prompt(self, initial_prompt, model, system_prompt_content, llm_logger, log_extra):
        start_time = time.time()
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        '''role''': '''system''',
                        '''content''': system_prompt_content,
                    },
                    {
                        '''role''': '''user''',
                        '''content''': initial_prompt,
                    },
                ],
            )
            latency = time.time() - start_time
            llm_logger.info(
                "LLM call successful",
                extra={
                    **log_extra,
                    "provider": "ollama",
                    "model_used": model,
                    "tokens_used": response.get('''eval_count'''),
                    "latency": latency,
                    "outcome": "success",
                },
            )
            return response['''message''']['''content''']
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
            api_version="2023-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def get_refined_prompt(self, initial_prompt, model, system_prompt_content, llm_logger, log_extra):
        start_time = time.time()
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": initial_prompt},
                ],
            )
            latency = time.time() - start_time
            llm_logger.info(
                "LLM call successful",
                extra={
                    **log_extra,
                    "provider": "azure_openai",
                    "model_used": model,
                    "tokens_used": response.usage.total_tokens,
                    "latency": latency,
                    "outcome": "success",
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
