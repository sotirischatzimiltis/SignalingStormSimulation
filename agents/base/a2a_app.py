"""
Shared A2A boilerplate so the JSON-RPC route / TaskStore / AgentCard wiring
(confirmed against the actually-installed a2a-sdk 1.1.0, not guessed) isn't
copy-pasted across the three agent processes.

Design choice: every inter-agent exchange in this system is a short
request/response -- the callee's executor reaches a terminal task state
within the same call (an instant ack, or after a brief stub-reason() delay).
There are no long-held streaming connections between agents. Long-running
work (tactical's trigger-poll loop, the simulator itself) runs as
independent background asyncio tasks / OS threads inside each process, and
only emits discrete, separately-instrumented A2A messages at meaningful
points (episode kickoff, escalation, episode-complete, validate-recovery,
results) -- exactly what the overhead report needs to count.

All payloads are structured (protobuf Struct via Part.data /
new_data_message), never JSON-encoded into a text part -- a2a-sdk supports
this natively (see a2a.helpers.new_data_message / get_data_parts).
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from starlette.applications import Starlette

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.types.a2a_pb2 import Role, SendMessageRequest

from instrumentation import db
from instrumentation.recorder import InstrumentedRecorder


def make_agent_card(
    name: str, description: str, host: str, port: int, skill_id: str, skill_description: str,
    advertise_host: str | None = None,
) -> AgentCard:
    """`host` is only used as the fallback advertised address (and, by every
    existing caller, also passed separately to uvicorn as the BIND host).
    These are not always the same thing: under --backend docker, agents must
    bind 0.0.0.0 to accept connections from other containers, but "0.0.0.0"
    is not a dialable address -- a peer that resolves THIS agent's own
    self-reported AgentInterface.url (a2a-sdk's create_client() does this
    for the actual send_message() call, even though the initial agent-card
    GET uses whatever url the caller was given directly) would then try to
    connect to "http://0.0.0.0:<port>" and fail. `advertise_host` lets
    __main__.py pass the real reachable address (the container's service
    name) independently of the bind host."""
    skill = AgentSkill(
        id=skill_id,
        name=skill_id,
        description=skill_description,
        input_modes=["application/json"],
        output_modes=["application/json"],
        tags=["resilience-orchestration"],
        examples=[],
    )
    return AgentCard(
        name=name,
        description=description,
        version="0.1.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=False),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=f"http://{advertise_host or host}:{port}")
        ],
        skills=[skill],
    )


def build_agent_app(
    executor: AgentExecutor, agent_card: AgentCard, recorder: InstrumentedRecorder
) -> Starlette:
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=agent_card
    )
    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(handler, "/"))

    @asynccontextmanager
    async def lifespan(app: Starlette):
        # asyncpg pools are bound to the event loop they're created in --
        # uvicorn.run() below owns its own loop, so the pool can only be
        # created here (inside that loop, via this ASGI lifespan hook), not
        # in the sync main() that constructs `recorder` before this app
        # even exists. See instrumentation/db.py's module docstring.
        pool = await db.get_pool()
        recorder.set_pool(pool)
        try:
            yield
        finally:
            await pool.close()

    return Starlette(routes=routes, lifespan=lifespan)


def run_agent_process(app: Starlette, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")


def get_request_data(context: RequestContext) -> dict:
    """Pulls the structured JSON payload out of the incoming message."""
    parts = get_data_parts(context.message.parts)
    return parts[0] if parts else {}


def extract_response_data(chunk) -> dict:
    """chunk: a2a_pb2.StreamResponse. Looks in whichever oneof field
    ('task', 'message', 'status_update', 'artifact_update') is populated for
    the data payload our own executors place in the final status message."""
    which = chunk.WhichOneof("payload")
    parts = []
    if which == "message":
        parts = get_data_parts(chunk.message.parts)
    elif which == "task":
        task = chunk.task
        if task.HasField("status") and task.status.HasField("message"):
            parts = get_data_parts(task.status.message.parts)
        if not parts:
            for art in task.artifacts:
                parts.extend(get_data_parts(art.parts))
    elif which == "status_update":
        status = chunk.status_update.status
        if status.HasField("message"):
            parts = get_data_parts(status.message.parts)
    elif which == "artifact_update":
        parts = get_data_parts(chunk.artifact_update.artifact.parts)
    return parts[0] if parts else {}


def _is_connection_error(exc: BaseException) -> bool:
    """Walks __cause__ (a2a/httpx wrap the real httpx.ConnectError in their
    own exception types) looking for an actual connection-level failure,
    rather than string-matching error messages."""
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        cur = cur.__cause__
    return False


async def send_data_message(
    target_url: str, data: dict, recorder: InstrumentedRecorder, name: str,
    max_attempts: int = 3, retry_backoff_s: float = 0.5,
) -> dict:
    """Resolves the remote agent's card, sends one structured-data message,
    awaits the full (non-streaming) response, returns the extracted JSON
    payload. Used for every inter-agent exchange in this system.

    Overwrites the resolved agent card's own advertised URL with target_url
    before creating the client -- see the comment at that line for why (the
    agent-card GET succeeding while the subsequent POST fails with "All
    connection attempts failed" / a DNS lookup error was the --backend docker
    bug this fixes). Also retries on connection-level failures (a fresh
    httpx.AsyncClient each attempt) as general defensive robustness; does not
    retry on application-level errors (those propagate immediately, same as
    before)."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        # timeout=None: some calls (e.g. start_episode) legitimately block for
        # the whole episode duration (minutes), since the coordinator only
        # completes that task once the episode is actually done -- see
        # coordinator/executor.py.
        async with httpx.AsyncClient(timeout=None) as httpx_client:
            try:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=target_url)
                agent_card = await resolver.get_agent_card()
                # create_client() dials whatever URL the agent card itself
                # declares (supported_interfaces[*].url), NOT target_url --
                # which breaks the moment a peer is reachable at a different
                # address for different callers (e.g. --backend docker: the
                # host-side orchestrator reaches an agent via 127.0.0.1:<port>,
                # but that same agent must advertise its container service
                # name, e.g. "coordinator", for sibling containers to dial
                # it). Overwrite with target_url -- the address THIS caller
                # already knows is correct -- so every caller's own view of
                # the network is authoritative over the peer's self-report.
                for iface in agent_card.supported_interfaces:
                    iface.url = target_url
                config = ClientConfig(streaming=False, httpx_client=httpx_client)
                client = await create_client(agent=agent_card, client_config=config)
            except Exception as exc:  # noqa: BLE001 - connection failure resolving/creating the client
                last_exc = exc
                if not _is_connection_error(exc) or attempt == max_attempts:
                    raise
                await asyncio.sleep(retry_backoff_s * attempt)
                continue

            message = new_data_message(data, role=Role.ROLE_USER)
            request = SendMessageRequest(message=message)
            result: dict = {}
            try:
                async with recorder.record("a2a", name, data) as rec:
                    async for chunk in client.send_message(request):
                        result = extract_response_data(chunk)
                    rec.payload_out = result
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_connection_error(exc) or attempt == max_attempts:
                    raise
                await asyncio.sleep(retry_backoff_s * attempt)
            finally:
                await client.close()
    raise last_exc  # pragma: no cover - loop always returns or raises above


async def wait_until_ready(probe, timeout_s: float = 30.0, interval_s: float = 0.3) -> None:
    """Generic async readiness retry loop. `probe` is a zero-arg async
    callable that raises on failure. Used for both MCP (ClientSession
    handshake) and A2A (agent-card fetch) readiness -- protocol-level
    checks, not a fabricated /healthz endpoint."""
    deadline = time.monotonic() + timeout_s
    last_exc: Any = None
    while time.monotonic() < deadline:
        try:
            await probe()
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            await asyncio.sleep(interval_s)
    raise TimeoutError(f"not ready after {timeout_s}s: {last_exc}")


async def probe_agent_ready(url: str) -> None:
    async with httpx.AsyncClient(timeout=5.0) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        await resolver.get_agent_card()
