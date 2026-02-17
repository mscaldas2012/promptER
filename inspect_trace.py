
import inspect
from langfuse import Langfuse

try:
    sig = inspect.signature(Langfuse.trace)
    print(f"Signature of Langfuse.trace: {sig}")
except Exception as e:
    print(f"Error getting signature: {e}")
