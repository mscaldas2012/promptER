
import inspect
from langfuse._client.span import LangfuseSpan

print("Methods of LangfuseSpan:")
for name, obj in inspect.getmembers(LangfuseSpan):
    if not name.startswith("_") and inspect.isfunction(obj):
        print(f"  {name}")
