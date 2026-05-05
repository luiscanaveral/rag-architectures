from rich.console import Console

console = Console()

def print_header(title: str):
    console.rule(f"[bold blue]{title}")

def print_result(label: str, value: str):
    console.print(f"[bold green]{label}:[/bold green] {value}")

# Lazy imports for logging and tracking
def __getattr__(name):
    if name in ["log", "debug", "info", "warning", "error", "critical"]:
        from utils.logging import log, debug, info, warning, error, critical
        return locals().get(name)
    elif name in ["TokenTracker", "track_llm_call"]:
        from utils.tracking import TokenTracker, track_llm_call
        return locals().get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
