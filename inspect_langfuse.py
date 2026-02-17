
import langfuse

import inspect
from dotenv import load_dotenv


load_dotenv(".env")
try:
    print(f"Langfuse version: {langfuse.version.__version__}")
except:
    print("Could not determine Langfuse version.")

print("\nMethods of Langfuse class:")
try:
    client_class = langfuse.Langfuse
    for name, obj in inspect.getmembers(client_class):
        if not name.startswith("_"):
            print(f"  {name}")
except Exception as e:
    print(f"Error inspecting Langfuse class: {e}")

print("\nMethods of Langfuse instance:")
try:
    client = langfuse.Langfuse()
    for name, obj in inspect.getmembers(client):
        if not name.startswith("_") and inspect.ismethod(obj):
            print(f"  {name}")
except Exception as e:
    print(f"Error inspecting Langfuse instance: {e}")
