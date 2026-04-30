"""FastAPI + WebSocket server powering EVA's interactive interface.

Endpoints
---------

* ``GET /`` — the voice-first single-page app (static ``index.html``).
* ``GET /api/status`` — current genome, parameter count, RAM estimate,
  device, step number.
* ``GET /api/memory/episodes?limit=50`` — recent episodes from the
  persistent store.
* ``GET /api/memory/insights?limit=50`` — emergent insights.
* ``GET /api/memory/thoughts?limit=200`` — raw thought stream (noise +
  signal mixed; the UI tags by category).
* ``GET /api/evolution`` — genome history.
* ``GET /api/tools`` — available tools and their documentation.
* ``POST /api/chat`` — one-shot chat turn (text). The response streams
  back on the ``/ws`` websocket too so the UI can render tokens live.
* ``WS   /ws`` — multiplexed live stream: thoughts, insights, safety
  alerts, token deltas.

The server intentionally stays framework-light so the install footprint
is small enough for 8 GiB hosts. Voice input/output lives in the browser
(``SpeechRecognition`` + ``speechSynthesis``) — no server-side model.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from eva.core.baby_brain import BabyBrain, detect_device
from eva.core.tokenizer import EVATokenizer
from eva.emotions.affect import AffectiveState
from eva.evolution.evolver import ArchitectureGenome, EvolutionEvent, Evolver
from eva.memory.persistent import (
    Insight,
    PersistentMemoryStore,
    StoredEpisode,
)
from eva.tools.builtins import register_default_tools
from eva.tools.registry import ToolRegistry
from eva.tools.usage_tracker import ToolUsageTracker

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    workspace: Path
    vocab_size: int = 256
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 6
    max_seq_len: int = 512
    max_ram_gb: float = 8.0
    device: str = "auto"


class EVARuntime:
    """In-process EVA that the UI talks to."""

    def __init__(self, cfg: RuntimeConfig) -> None:
        self.cfg = cfg
        self.workspace = cfg.workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self.tokenizer = EVATokenizer()
        # Use the real tokenizer vocab size — avoids filler tokens leaking
        # into the response stream.
        effective_vocab = self.tokenizer.vocab_size

        self.device = detect_device(cfg.device)
        self.brain = BabyBrain(
            vocab_size=effective_vocab,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            dtype_str="float32",
            device=self.device,
            max_seq_len=cfg.max_seq_len,
        )

        self.genome = ArchitectureGenome(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
        )
        self.store = PersistentMemoryStore(self.workspace / "memory.db")

        self.evolver = Evolver(
            brain=self.brain,
            genome=self.genome,
            max_ram_gb=cfg.max_ram_gb,
            on_event=self._on_evolution_event,
        )

        self.tools = ToolRegistry()
        register_default_tools(
            self.tools, workspace_root=self.workspace / "tool_workspace"
        )
        self.tool_tracker = ToolUsageTracker()

        self.affect = AffectiveState()
        self._step = 0
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

        # Seed genome history.
        self.store.record_genome(
            step=0,
            genes=self.genome.to_dict(),
            parameter_count=self.brain.parameter_count,
        )

    # ------------------------------------------------------------------
    # pub/sub
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=128)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(q)

    def _broadcast(self, event: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            if q.full():
                continue
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def _on_evolution_event(self, event: EvolutionEvent) -> None:
        self.store.record_genome(
            step=event.step,
            genes=event.genome_after,
            parameter_count=self.brain.parameter_count,
        )
        self._broadcast(
            {"type": "evolution", "event": event.__dict__}
        )

    # ------------------------------------------------------------------
    # conversational turn
    # ------------------------------------------------------------------

    def _record_thought(
        self, category: str, content: str, context: Optional[dict] = None
    ) -> None:
        self.store.record_thought(
            step=self._step,
            category=category,
            content=content,
            context=context or {},
        )
        self._broadcast(
            {
                "type": "thought",
                "step": self._step,
                "category": category,
                "content": content,
                "context": context or {},
            }
        )

    def chat(self, text: str, max_new_tokens: int = 80) -> str:
        """Run one conversational turn and push thoughts + insights."""

        self._step += 1
        self._record_thought("input", text, {"source": "human"})

        prompt_ids = self.tokenizer.encode(text, source="human")
        ctx = prompt_ids[-self.cfg.max_seq_len :]
        generated: list[int] = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                tokens = torch.tensor(
                    [ctx + generated], dtype=torch.long, device=self.device
                )
                probs = self.brain.predict_next(tokens)
                probs = probs.squeeze(0).float()
                temperature = max(
                    0.1, float(self.genome.exploration_temperature)
                )
                probs = torch.softmax(torch.log(probs + 1e-10) / temperature, -1)
                next_id = int(torch.multinomial(probs, 1).item())
                if next_id == 3:  # EOS
                    break
                generated.append(next_id)

        response = self.tokenizer.decode(generated).strip()
        if not response:
            response = "(silence)"
        self._record_thought("output", response, {"source": "self"})

        # Synthetic surprise/importance since we don't do training here.
        surprise = float(torch.rand(()).item())
        importance = min(1.0, 0.3 + surprise)
        self.store.record_episode(
            StoredEpisode(
                step=self._step,
                action=generated[0] if generated else 0,
                outcome=prompt_ids[-1] if prompt_ids else 0,
                surprise=surprise,
                emotional_importance=importance,
                source_tag="human",
            )
        )

        # Feed the evolver — plateaus eventually trigger growth.
        self.evolver.step(self._step, reward=surprise)

        # Heuristic insight detection: long, non-trivial response.
        if len(response) > 20 and surprise > 0.75:
            insight = Insight(
                step=self._step,
                kind="response_surprise",
                description=response[:120],
                confidence=min(0.95, surprise),
                data={"prompt": text},
            )
            self.store.record_insight(insight)
            self._broadcast({"type": "insight", "insight": insight.__dict__})

        self._broadcast(
            {
                "type": "turn",
                "step": self._step,
                "prompt": text,
                "response": response,
                "surprise": surprise,
            }
        )
        return response

    # ------------------------------------------------------------------
    # status / introspection
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "step": self._step,
            "device": str(self.device),
            "parameter_count": self.brain.parameter_count,
            "memory_estimate_mb": self.brain.estimate_memory_bytes() // (1024 ** 2),
            "genome": self.genome.to_dict(),
            "affect": self.affect.to_dict(),
            "tools": self.tools.discover(),
            "ram_budget_gb": self.cfg.max_ram_gb,
        }

    def tool_docs(self) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        for name in self.tools.list_all():
            info = self.tools.get_documentation(name)
            if info is None:
                continue
            docs.append(
                {
                    "name": info.name,
                    "description": info.description,
                    "available": name in self.tools.discover(),
                    "parameters": [
                        p.__dict__ for p in info.parameters
                    ],
                    "examples": list(info.examples),
                }
            )
        return docs


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------


def build_app(runtime: EVARuntime):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "fastapi is not installed. Run `pip install .[ui]` or "
            "`pip install fastapi uvicorn`."
        ) from exc

    app = FastAPI(title="EVA", docs_url="/api/docs")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return runtime.status()

    @app.get("/api/memory/episodes")
    def episodes(limit: int = 50) -> dict[str, Any]:
        return {"episodes": runtime.store.recent_episodes(limit)}

    @app.get("/api/memory/insights")
    def insights(limit: int = 50) -> dict[str, Any]:
        return {"insights": runtime.store.recent_insights(limit)}

    @app.get("/api/memory/thoughts")
    def thoughts(limit: int = 200) -> dict[str, Any]:
        return {"thoughts": runtime.store.recent_thoughts(limit)}

    @app.get("/api/evolution")
    def evolution() -> dict[str, Any]:
        return {"history": runtime.store.genome_history()}

    @app.get("/api/tools")
    def tools() -> dict[str, Any]:
        return {"tools": runtime.tool_docs()}

    @app.post("/api/chat")
    async def chat(payload: dict[str, Any]) -> JSONResponse:
        text = str(payload.get("text", "")).strip()
        if not text:
            return JSONResponse(
                {"error": "empty text"}, status_code=400
            )
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, runtime.chat, text
        )
        return JSONResponse({"response": response})

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        q = runtime.subscribe()
        try:
            await websocket.send_json(
                {"type": "status", "status": runtime.status()}
            )
            while True:
                event = await q.get()
                await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            runtime.unsubscribe(q)

    if STATIC_DIR.exists():
        app.mount(
            "/", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui"
        )

    return app


def run_server(
    runtime: EVARuntime,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Blocking entrypoint used by ``run.py ui``."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "uvicorn is not installed. Run `pip install uvicorn`."
        ) from exc

    app = build_app(runtime)
    uvicorn.run(app, host=host, port=port, log_level="info")
