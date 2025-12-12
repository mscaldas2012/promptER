import os
import time

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

load_dotenv()
# Set up tracing
tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)

connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
exporter = AzureMonitorTraceExporter(connection_string=connection_string)
span_processor = BatchSpanProcessor(exporter)
tracer_provider.add_span_processor(span_processor)

tracer = trace.get_tracer(__name__)

# Create a test trace
with tracer.start_as_current_span("test-span") as span:
    span.set_attribute("test.attribute", "hello-world")
    print("Test span created")
    time.sleep(1)

# Force flush
print("Flushing data...")
tracer_provider.shutdown()
print("Done! Check Application Insights in 1-2 minutes")