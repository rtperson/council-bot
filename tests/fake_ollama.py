"""
A stand-in for a local Ollama server, for tests.

Speaks the parts of the API the council uses: the OpenAI-compatible
/v1/chat/completions endpoint (what CrewAI calls) and /v1/models (what the
command line checks before a run). Every request is recorded, and a `reply`
function decides each response, so a test can play a well-behaved model or
one that never stops calling tools.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


def text_reply(content: str) -> dict:
    return {"role": "assistant", "content": content}


def tool_reply(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


class FakeOllama:
    def __init__(self, reply: Callable[[dict], dict], models=("fake-model",),
                 prompt_tokens: int = 1000, completion_tokens: int = 100):
        self.reply = reply
        self.models = list(models)
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.requests: list[dict] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def _send(self, body: dict, status: int = 200):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.rstrip("/") == "/v1/models":
                    self._send({"object": "list", "data": [{"id": m, "object": "model"} for m in server.models]})
                else:
                    self._send({"error": "not found"}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                message = server.reply(body)
                self._send({
                    "id": f"chatcmpl-{len(server.requests)}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }],
                    "usage": {
                        "prompt_tokens": server.prompt_tokens,
                        "completion_tokens": server.completion_tokens,
                        "total_tokens": server.prompt_tokens + server.completion_tokens,
                    },
                })

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
