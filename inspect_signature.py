
import inspect
from langfuse import Langfuse

try:
    sig = inspect.signature(Langfuse.start_generation)
    print(f"Signature of start_generation: {sig}")
except Exception as e:
    print(f"Error getting signature: {e}")
