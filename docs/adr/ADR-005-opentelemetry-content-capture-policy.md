# ADR-005: OpenTelemetry semantic conventions and content-capture policy

- **Status**: Accepted
- **Date**: 2026-09-04
- **Deciders**: GroundGraph maintainers
- **Related**: `docs/plan.md` §10, `AGENTS.md` §6, `ADR-001`

## Context

GroundGraph needs correlated operational traces, logs, and metrics for every
request without exporting source-document contents, prompts, answers,
credentials, or hidden reasoning. Telemetry configuration must also be
testable: unit tests must not make network calls to a locally configured OTLP
collector or mutate global OpenTelemetry state.

## Decision

1. Every FastAPI request receives a validated `x-request-id`; its value is
   also used as the correlation ID. Invalid or absent values are replaced by a
   server-generated identifier.
2. The API binds request ID, correlation ID, trace ID, and span ID to
   `structlog.contextvars` for the request lifetime, logs a structured
   completion/failure event, then clears the context.
3. API tracing uses `FastAPIInstrumentor` with an application-local
   `TracerProvider`. It records stable HTTP metadata (`http.request.method`,
   route template, response status) and safe GroundGraph correlation IDs.
4. Application startup may export spans by OTLP. Unit tests disable OTLP by
   default and can inject an in-memory `SpanExporter`; the application shuts
   its provider down at lifespan exit. Providers are never installed globally.
5. Raw request/response bodies and all `Authorization`, `Cookie`,
   `Set-Cookie`, proxy-authorization, password, secret, and token attributes
   are excluded from spans and general logs. The telemetry helper accepts only
   scalar allowlisted metadata after key-based sanitization.
6. Prometheus metrics record request count, server errors, duration, and each
   readiness dependency's health. `/metrics` exposes the application registry
   in Prometheus format.
7. Readiness results expose only dependency name, boolean health, and a stable
   reason code. Exception details remain internal and never enter an API
   response or telemetry payload.

## Consequences

- Operators can correlate a request across JSON logs, traces, and metrics using
  safe IDs without retaining raw request content.
- An injected exporter allows deterministic trace assertions without localhost
  OTLP retries or process-global tracer warnings.
- Debugging requires looking up protected source content through its governed
  storage path rather than reading it from telemetry.
- New telemetry attributes require review: they must be scalar, necessary, and
  pass `sanitize_attributes()` before export.

## Alternatives considered

- **Capture complete requests and responses in Phoenix**: rejected because it
  violates the default content-capture prohibition in `plan.md` §10.3.
- **Use the global OpenTelemetry provider**: rejected because repeated app
  construction in tests causes global-state warnings and exporter thread leaks.
- **Use a plain readiness boolean**: rejected because operators need a safe,
  stable dependency-level reason code without raw upstream exception details.

## References

- `docs/plan.md` §0.1, §10.1-§10.4, §12.4
- OpenTelemetry semantic conventions for HTTP server spans
- OpenTelemetry FastAPI instrumentation documentation
