import time

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider


def run_tracing_smoke_test():
    """Create a span with the default tracer provider to ensure telemetry wiring works."""
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("test.attribute", "hello-world")
        time.sleep(0.1)
        return span.is_recording()


if __name__ == "__main__":
    span_recording = run_tracing_smoke_test()
    print(f"Span recording without Azure exporter: {span_recording}")
