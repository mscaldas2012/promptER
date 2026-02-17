
import inspect
from langfuse import Langfuse
from langfuse._client.span import LangfuseSpan

try:
    sig_start = inspect.signature(Langfuse.start_span)
    print(f"Signature of Langfuse.start_span: {sig_start}")
except Exception as e:
    print(f"Error getting start_span signature: {e}")

try:
    sig_update = inspect.signature(LangfuseSpan.update)
    print(f"Signature of LangfuseSpan.update: {sig_update}")
except Exception as e:
    print(f"Error getting LangfuseSpan.update signature: {e}")
