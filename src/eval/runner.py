import os
import sys
import json
import time as time_module
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()

from utils.config import get_embeddings, get_vectorstore
from utils.semantic_cache import SemanticCache
from eval.dataset import SINGLE_TURN_TEST_CASES

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from langchain_openai import ChatOpenAI

console = Console()
REPORT_DIR = Path(".logs/reports")


def _make_report_path() -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return REPORT_DIR / f"eval.{ts}.log"


def _append_report(report_path: Path, entry: dict):
    with open(report_path, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _make_judge():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaModel(model=model_name, base_url=base_url)
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name)


ARCHITECTURES: dict[str, callable] = {}


def _register_all():
    from rag_architectures.simple.app import run_simple_rag
    ARCHITECTURES["simple"] = run_simple_rag

    for mod_name, fn_name, label in [
        ("rag_architectures.standard.app", "run_standard_rag", "standard"),
        ("rag_architectures.contextual.app", "run_contextual_rag", "contextual"),
        ("rag_architectures.corrective.app", "run_corrective_rag", "corrective"),
        ("rag_architectures.conversational.app", "run_conversational_rag", "conversational"),
        ("rag_architectures.fusion.app", "run_fusion_rag", "fusion"),
        ("rag_architectures.agentic.app", "run_agentic_rag", "agentic"),
        ("rag_architectures.graph.app", "run_graph_rag", "graph"),
    ]:
        import importlib
        mod = importlib.import_module(mod_name)
        ARCHITECTURES[label] = getattr(mod, fn_name)


def eval_architectures(
    arch_names: list[str] | None = None,
    test_names: list[str] | None = None,
    k: int = 5,
):
    _register_all()

    report_path = _make_report_path()
    judge = _make_judge()

    faithfulness = FaithfulnessMetric(model=judge, threshold=0.5)
    answer_relevancy = AnswerRelevancyMetric(model=judge, threshold=0.5)
    contextual_precision = ContextualPrecisionMetric(model=judge, threshold=0.5)

    targets = [n for n in ARCHITECTURES if arch_names is None or n in arch_names]
    cases = [tc for tc in SINGLE_TURN_TEST_CASES if test_names is None or tc["name"] in test_names]

    vectordb = get_vectorstore()

    summary: dict[str, dict[str, list[float]]] = {}

    for arch in targets:
        SemanticCache(embedding_func=get_embeddings()).clear()
        console.rule(f"[bold blue]{arch.upper()} RAG")

        for tc in cases:
            name = tc["name"]
            query = tc["input"]
            expected = tc["expected_output"]

            result = ARCHITECTURES[arch](query)
            actual_output = result["answer"]

            docs = vectordb.similarity_search(query, k=k)
            retrieval_context = [d.page_content for d in docs]

            test_case = LLMTestCase(
                input=query,
                actual_output=actual_output,
                expected_output=expected,
                retrieval_context=retrieval_context,
            )

            table = Table(title=f"{arch} — {name}", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Score", style="green")
            table.add_column("Threshold", style="yellow")
            table.add_column("Pass", style="bold")

            entries: list[dict] = []

            for metric in [faithfulness, answer_relevancy, contextual_precision]:
                label = metric.__class__.__name__.replace("Metric", "")
                try:
                    metric.measure(test_case)
                    passed = metric.is_successful()
                    score = metric.score
                    table.add_row(
                        label,
                        f"{score:.3f}",
                        f"{metric.threshold}",
                        "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
                    )
                    entries.append({"metric": label, "score": score, "threshold": metric.threshold, "passed": passed})
                except Exception as e:
                    table.add_row(label, "ERROR", f"{metric.threshold}", f"[red]{e!s}[/red]")
                    entries.append({"metric": label, "score": None, "threshold": metric.threshold, "passed": False, "error": str(e)})

            console.print(table)
            console.print()

            _append_report(report_path, {
                "timestamp": datetime.now().isoformat(),
                "architecture": arch,
                "test_case": name,
                "query": query,
                "expected_output": expected,
                "actual_output": actual_output,
                "metrics": {e["metric"]: {"score": e["score"], "passed": e["passed"]} for e in entries},
            })

            summary.setdefault(arch, {}).setdefault("all", [])
            for e in entries:
                summary.setdefault(arch, {}).setdefault(e["metric"], [])
                if e["score"] is not None:
                    summary[arch][e["metric"]].append(e["score"])
                    summary[arch]["all"].append(e["score"])

    # Summary
    console.rule("[bold yellow]SUMMARY")
    console.print(f"Report saved to: [cyan]{report_path}[/cyan]\n")

    summary_table = Table(title="Average Scores per Architecture", show_header=True)
    summary_table.add_column("Architecture", style="cyan")
    summary_table.add_column("Avg Score", style="green")
    summary_table.add_column("Avg Faithfulness", style="green")
    summary_table.add_column("Avg AnswerRelevancy", style="green")
    summary_table.add_column("Avg ContextualPrecision", style="green")

    for arch in targets:
        s = summary[arch]
        avg_all = sum(s["all"]) / len(s["all"]) if s["all"] else 0
        avg_f = sum(s["Faithfulness"]) / len(s["Faithfulness"]) if s.get("Faithfulness") else 0
        avg_a = sum(s["AnswerRelevancy"]) / len(s["AnswerRelevancy"]) if s.get("AnswerRelevancy") else 0
        avg_c = sum(s["ContextualPrecision"]) / len(s["ContextualPrecision"]) if s.get("ContextualPrecision") else 0
        summary_table.add_row(arch, f"{avg_all:.3f}", f"{avg_f:.3f}", f"{avg_a:.3f}", f"{avg_c:.3f}")

    console.print(summary_table)
    console.print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG architectures with DeepEval")
    parser.add_argument("--arch", nargs="*", help="Architectures to evaluate (default: all)")
    parser.add_argument("--test", nargs="*", help="Test cases to run (default: all)")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved documents (default: 5)")
    args = parser.parse_args()

    eval_architectures(arch_names=args.arch, test_names=args.test, k=args.k)
