from rich.console import Console

console = Console()

def print_header(title: str):
    console.rule(f"[bold blue]{title}")

def print_result(label: str, value: str):
    console.print(f"[bold green]{label}:[/bold green] {value}")
