# Lazy imports to avoid circular imports
def __getattr__(name):
    if name == "ingest_documents":
        from .ingestion import ingest_documents
        return ingest_documents
    elif name in ["console", "print_header", "print_result", "log", "debug", "info", "warning", "error", "critical"]:
        from .utils import console, print_header, print_result, log, debug, info, warning, error, critical
        return locals().get(name)
    elif name in ["TokenTracker", "track_llm_call"]:
        from .utils.tracking import TokenTracker, track_llm_call
        return locals().get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ingest_documents", "console", "print_header", "print_result", "log", "debug", "info", "warning", "error", "critical", "TokenTracker", "track_llm_call"]
