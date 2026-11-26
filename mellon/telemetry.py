"""Ours: where a graph run's seconds go, as OpenTelemetry histograms (explicit buckets, delta temporality).

The fork records through the OpenTelemetry API only. A process that has set up the SDK already (H3's container on
Modal, whose app exports its own) gets the fork's histograms through its provider. Otherwise, when the SDK's own
OTEL_EXPORTER_OTLP_ENDPOINT is set (a Mac's service, pointed at the estate collector), configure() sets the SDK up
and turns on aiohttp's maintained instrumentation: the server's requests (/graph, /cache, ...) and the client's,
which is how s3fs reaches the store, so every request to R2 is timed with no code here. Without the variable, nothing
is sent and nothing fails. Export runs on the SDK's own thread every 10 s and nothing here waits on it: with the
collector unreachable a flush blocked 7 s (refused) to 10 s (no answer), measured on SDK 1.45.2, so a run never flushes;
a failed export is the SDK's one logged error.

- mellon.graph.duration: one posted graph from its start to its end, by outcome
- mellon.node.duration: one node of it, by folder and outcome (ran here, read from the store, or remote: run where
  nodes.json places it)
- mellon.store.duration and mellon.store.size (store.py's): one operation on the store, by op (check, a run's keys and
  whether each is stored; keys; read, a run's entries in one pass; refs, resolve, put); the bytes moved
- mellon.remote.duration: a placed node's call, by phase (check, spawn, wait; poll, the wait less the call's
  own queue and run, as the call reports them)
"""

import os
import socket
import time
from contextlib import contextmanager

from opentelemetry import metrics

from mellon.store import SECONDS, store_s  # the store's own instruments are store.py's, which runs alone too

meter = metrics.get_meter("mellon")
graph_s = meter.create_histogram(
    "mellon.graph.duration", unit="s", description="One posted graph. By outcome.",
    explicit_bucket_boundaries_advisory=SECONDS,
)
node_s = meter.create_histogram(
    "mellon.node.duration", unit="s", description="One node of a graph. By folder, outcome (ran, stored, remote).",
    explicit_bucket_boundaries_advisory=SECONDS,
)
remote_s = meter.create_histogram(
    "mellon.remote.duration", unit="s", description="A placed node's call. By phase.",
    explicit_bucket_boundaries_advisory=SECONDS,
)


@contextmanager
def timed(histogram, **attributes):
    """The block's seconds in `histogram`; the caller may add attributes to the dict it is given."""
    start = time.perf_counter()
    try:
        yield attributes
    finally:
        histogram.record(time.perf_counter() - start, attributes)


def configure() -> None:
    """The SDK, exporting to OTEL_EXPORTER_OTLP_ENDPOINT, unless this process has a provider of its own or nowhere
    to send to."""
    from opentelemetry.sdk.metrics import MeterProvider

    if isinstance(metrics.get_meter_provider(), MeterProvider) or not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.aiohttp_server import AioHttpServerInstrumentor
    from opentelemetry.sdk.metrics import Histogram
    from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    exporter = OTLPMetricExporter(preferred_temporality={Histogram: AggregationTemporality.DELTA})
    resource = Resource.create({"service.name": "h3-mellon", "host.name": socket.gethostname()})
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(exporter, 10_000)])
    )
    AioHttpServerInstrumentor().instrument()
    AioHttpClientInstrumentor().instrument()
