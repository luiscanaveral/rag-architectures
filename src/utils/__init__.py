from rich.console import Console
from utils.logging import log, debug, info, warning, error, critical
from utils.tracking import TokenTracker, track_llm_call

console = Console()

def print_header(title: str):
    console.rule(f"[bold blue]{title}")

def print_result(label: str, value: str):
    console.print(f"[bold green]{label}:[/bold green] {value}")
