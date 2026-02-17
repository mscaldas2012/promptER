
import inspect
from langfuse._client.span import LangfuseGeneration

try:
    sig_update = inspect.signature(LangfuseGeneration.update)
    print(f"Signature of LangfuseGeneration.update: {sig_update}")
except Exception as e:
    print(f"Error getting update signature: {e}")

try:
    sig_end = inspect.signature(LangfuseGeneration.end)
    print(f"Signature of LangfuseGeneration.end: {sig_end}")
except Exception as e:
    print(f"Error getting end signature: {e}")
