import os
import json
import base64
import threading
from datetime import datetime, timezone
from uuid import uuid4
from urllib.request import Request, urlopen
from urllib.error import URLError
from langchain_core.callbacks import BaseCallbackHandler


class LangfuseRestCallback(BaseCallbackHandler):
    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.host = os.getenv("LANGFUSE_HOST", "http://localhost:4000")
        self._events: list[dict] = []
        self._trace_id: str | None = None
        self._generation_id: str | None = None

    def _auth_header(self) -> str:
        token = f"{self.public_key}:{self.secret_key}"
        return "Basic " + base64.b64encode(token.encode()).decode()

    def _flush(self):
        if not self._events:
            return
        events = self._events
        self._events = []
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
        try:
            urlopen(req, timeout=5)
        except URLError:
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
        self._event("trace-create", {"id": self._trace_id, "name": "LLMChain"})
        self._event("generation-create", {
            "id": self._generation_id,
            "trace_id": self._trace_id,
            "name": "LLM",
            "input": prompts[0] if prompts else "",
        })

    def on_llm_end(self, response, **kwargs):
        if not self._generation_id:
            return
        text = ""
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                if gen_list and hasattr(gen_list[0], "text"):
                    text += gen_list[0].text
        self._event("generation-update", {
            "id": self._generation_id,
            "trace_id": self._trace_id,
            "output": text,
        })
        threading.Thread(target=self._flush, daemon=True).start()
