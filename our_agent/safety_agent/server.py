from __future__ import annotations
import asyncio
from pathlib import Path
import uuid
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from .agent import SafetyAgent
from .wire import pack, unpack, METADATA, policy_observation


def make_handler(goal, config, upstream_url, output_dir, perception=None):
    async def handler(client):
        # Each connection is a new episode; remote payloads cannot reset budgets.
        directory = Path(output_dir) / (goal.question + "_" + uuid.uuid4().hex)
        agent = SafetyAgent(goal, config, directory)
        upstream = None
        try:
            try:
                upstream = await connect(upstream_url, open_timeout=config.backend_timeout_s, max_size=32 * 1024**2, compression=None)
                meta = unpack(await asyncio.wait_for(upstream.recv(), config.backend_timeout_s))
                if not isinstance(meta, dict) or meta.get("action_dim") != 10 or meta.get("state_dim") != 25:
                    raise ValueError("backend metadata mismatch")
            except Exception:
                agent.stop("backend_connect_failure")
            await client.send(pack({**METADATA, "question": goal.question, "control_hz": config.control_hz,
                                    "assurance_config_provenance": config.provenance}))
            async for message in client:
                try:
                    raw = unpack(message)
                except Exception:
                    raw = {}
                clean = agent.prepare(raw, perception=perception)
                if clean is None:
                    result = agent.finish()
                else:
                    try:
                        await asyncio.wait_for(upstream.send(pack(policy_observation(clean))), config.backend_timeout_s)
                        response = unpack(await asyncio.wait_for(upstream.recv(), config.backend_timeout_s))
                    except Exception as error:
                        result = agent.finish(backend_error=type(error).__name__)
                    else:
                        result = agent.finish(response)
                await client.send(pack(result))
        except ConnectionClosed:
            pass
        finally:
            agent.close()
            if upstream is not None:
                await upstream.close()
    return handler


async def run_server(goal, config, upstream, host, port, output, perception=None):
    if config.compensation_enabled:
        raise ValueError("stock WebSocket protocol has no trusted executed-step timing; use in-process adapter")
    async with serve(make_handler(goal, config, upstream, output, perception), host, port,
                     max_size=32 * 1024**2, compression=None):
        print(f"Safety Agent {goal.question}: ws://{host}:{port} -> {upstream}", flush=True)
        await asyncio.Future()
