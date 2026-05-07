import os
import json
import base64
import threading
from datetime import datetime, timezone
from uuid import uuid4
from urllib.request import Request, urlopen
from urllib.error import URLError
from langchain_core.callbacks import BaseCallbackHandler


def _extract_model_name(serialized: dict, kwargs: dict) -> str:
    metadata = kwargs.get("metadata", {})
    ls_model = metadata.get("ls_model_name")
    if ls_model:
        return str(ls_model)
    for src in [kwargs.get("invocation_params", {}), serialized]:
        for key in ("model_name", "model", "model_id"):
            val = src.get(key)
            if val:
                return str(val)
    return serialized.get("name", "unknown")


def _extract_usage(response) -> dict:
    usage = {}
    try:
        gen_info = {}
        if hasattr(response, "generations") and response.generations:
            gen_list = response.generations[0]
            if gen_list and hasattr(gen_list[0], "generation_info"):
                gen_info = gen_list[0].generation_info or {}
        if hasattr(response, "llm_output") and response.llm_output:
            llm = response.llm_output
            if "token_usage" in llm:
                t = llm["token_usage"]
                return {"input": t.get("prompt_tokens", 0), "output": t.get("completion_tokens", 0),
                        "total": t.get("total_tokens", 0), "unit": "TOKENS"}
            if "usage" in llm:
                u = llm["usage"]
                return {"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0),
                        "total": u.get("total_tokens", 0), "unit": "TOKENS"}
        if "prompt_eval_count" in gen_info:
            prompt = gen_info.get("prompt_eval_count", 0)
            output = gen_info.get("eval_count", 0)
            return {"input": prompt, "output": output, "total": prompt + output, "unit": "TOKENS"}
        if "token_usage" in gen_info:
            t = gen_info["token_usage"]
            return {"input": t.get("prompt_tokens", 0), "output": t.get("completion_tokens", 0),
                    "total": t.get("total_tokens", 0), "unit": "TOKENS"}
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for g in gen_list:
                    if hasattr(g, "message") and g.message and hasattr(g.message, "usage_metadata"):
                        u = g.message.usage_metadata
                        return {"input": u.get("input_tokens", 0), "output": u.get("output_tokens", 0),
                                "total": u.get("total_tokens", 0), "unit": "TOKENS"}
    except Exception:
        pass
    return usage


def _extract_output_text(response) -> str:
    try:
        if hasattr(response, "generations") and response.generations:
            parts = []
            for gen_list in response.generations:
                for g in gen_list:
                    if hasattr(g, "text"):
                        parts.append(g.text)
            return "\n".join(parts)
    except Exception:
        pass
    return ""


class LangfuseRestCallback(BaseCallbackHandler):
    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.host = os.getenv("LANGFUSE_HOST", "http://localhost:4000")
        self._events: list[dict] = []
        self._trace_id: str | None = None
        self._generation_id: str | None = None
        self._model: str = "unknown"
        self._input: str = ""

    def _auth_header(self) -> str:
        token = f"{self.public_key}:{self.secret_key}"
        return "Basic " + base64.b64encode(token.encode()).decode()

    def _flush(self):
        if not self._events:
            return
        events = self._events
        self._events = []
        try:
            body = json.dumps({"batch": events}).encode()
            req = Request(
                f"{self.host}/api/public/ingestion",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": self._auth_header(),
                },
                method="POST",
            )
            urlopen(req, timeout=5)
        except Exception:
            pass

    def _event(self, type_: str, body: dict):
        self._events.append({
            "id": uuid4().hex,
            "type": type_,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "body": body,
        })

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        self._trace_id = self._trace_id or uuid4().hex
        self._generation_id = uuid4().hex
        self._model = _extract_model_name(serialized, kwargs)
        self._input = prompts[0] if prompts else ""

        self._event("trace-create", {"id": self._trace_id, "name": self._model})
        self._event("generation-create", {
            "id": self._generation_id,
            "trace_id": self._trace_id,
            "name": self._model,
            "model": self._model,
            "input": self._input,
        })

    def on_llm_end(self, response, **kwargs):
        if not self._generation_id:
            return
        text = _extract_output_text(response)
        usage = _extract_usage(response)

        body = {
            "id": self._generation_id,
            "trace_id": self._trace_id,
            "model": self._model,
            "output": text,
        }
        if usage:
            body["usage"] = usage
        self._event("generation-update", body)
        threading.Thread(target=self._flush, daemon=True).start()
