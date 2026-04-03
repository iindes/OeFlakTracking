"""
agent_runner.py  —  Google ADK Agent Wiring & Execution Engine
==============================================================
Assembles the tracking agent from its parts and provides both an
async API and a sync CLI wrapper.

Architecture
------------
  TrackingAgent
  ├── LlmAgent (google.adk)
  │   ├── model             : gemini-2.0-flash
  │   ├── instruction       : SYSTEM_INSTRUCTION  (with 5 few-shot examples)
  │   ├── tools             : ALL_TOOLS (5 callables — ADK auto-schemas)
  │   ├── output_schema     : TrackingAgentResponse (Pydantic — strict JSON)
  │   └── generate_content_config
  │       ├── temperature   : 0.0   (deterministic — no sampling)
  │       ├── top_p         : 1.0   (full vocab, let temperature do the work)
  │       ├── seed          : 42    (reproducible across identical prompts)
  │       └── max_output_tokens: 4096
  ├── Runner (google.adk)  — manages async event loop
  └── InMemorySessionService — stateless per-query sessions

Determinism guarantee
---------------------
temperature=0.0 + seed=42 pins the Gemini sampler. output_schema enforces
the Pydantic model as a JSON constraint on the final generation turn (ADK
internally sets response_mime_type="application/json" + response_schema
for that one call while leaving the tool-calling turns unconstrained so
tool calls can still be parsed as function-call payloads).

Usage
-----
  # Programmatic (async)
  agent = TrackingAgent(api_key="AIza...")
  result = asyncio.run(agent.query("Run 10-step simulation"))

  # CLI
  python agent_runner.py --demo                  # 5 pre-canned queries
  python agent_runner.py --query "..."           # single query
  python agent_runner.py                         # interactive REPL

  # API key from environment
  export GOOGLE_API_KEY=AIza...
  python agent_runner.py --demo
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import warnings
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agent_tools import ALL_TOOLS
from agent_prompts import SYSTEM_INSTRUCTION, TrackingAgentResponse, FEW_SHOT_EXAMPLES


# ── Constants ────────────────────────────────────────────────────────────────

_MODEL          = "gemini-2.0-flash"
_APP_NAME       = "OeFlakTrack"
_USER_ID        = "operator-1"
_TEMPERATURE    = 0.0    # full determinism — no sampling variance
_TOP_P          = 1.0    # no nucleus truncation; temperature handles diversity
_SEED           = 42     # reproducible PRNG state for identical prompts
_MAX_TOKENS     = 4096   # sufficient for full tool result + structured response


# ── Determinism config ────────────────────────────────────────────────────────

def _build_generation_config() -> types.GenerateContentConfig:
    """
    Build the GenerateContentConfig that pins all sources of LLM randomness.

    temperature=0.0
        Makes the sampler always choose the highest-probability token.
        Eliminates per-run variance on identical inputs.

    top_p=1.0
        Keeps the full vocabulary available; combined with temperature=0
        this is redundant but makes the constraint explicit and forwards-
        compatible if temperature is bumped in future experiments.

    seed=42
        Gemini supports an explicit PRNG seed.  Identical (prompt, seed)
        pairs produce bitwise-identical outputs within a model version.

    max_output_tokens=4096
        Prevents runaway generation if a tool returns an unusually large dict.

    NOTE: response_mime_type / response_schema are NOT set here.
    ADK injects them automatically on the final (post-tool-calls) turn when
    output_schema is supplied to LlmAgent, while leaving intermediate turns
    (tool-call parsing) in normal text mode.
    """
    return types.GenerateContentConfig(
        temperature=_TEMPERATURE,
        top_p=_TOP_P,
        seed=_SEED,
        max_output_tokens=_MAX_TOKENS,
    )


# ── JSON validation ───────────────────────────────────────────────────────────

def _parse_response(text: str) -> TrackingAgentResponse:
    """
    Extract and validate the JSON payload from the agent's final response text.

    ADK with output_schema should guarantee valid JSON, but defensive parsing
    handles edge cases (e.g. model wraps JSON in a markdown code fence).
    """
    # Strip markdown code fences if present
    stripped = text.strip()
    if stripped.startswith("```"):
        lines  = stripped.splitlines()
        inner  = [l for l in lines if not l.startswith("```")]
        stripped = "\n".join(inner).strip()

    try:
        data = json.loads(stripped)
        return TrackingAgentResponse.model_validate(data)
    except Exception as exc:
        # Return a structured error response rather than raising
        return TrackingAgentResponse(
            status="error",
            intent="Agent returned an unparseable response.",
            tool_called="none",
            parameters={},
            result={},
            summary="The agent response could not be parsed as valid JSON.",
            error_message=f"Parse error: {exc} | Raw: {text[:300]}",
        )


# ── Core agent class ──────────────────────────────────────────────────────────

class TrackingAgent:
    """
    Google ADK-powered 3-D radar tracking agent.

    Parameters
    ----------
    api_key : str, optional
        Gemini API key. Falls back to GOOGLE_API_KEY environment variable.
    model : str
        Gemini model identifier. Default: "gemini-2.0-flash".
    verbose : bool
        If True, print intermediate tool-call events to stdout.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = _MODEL,
        verbose: bool = False,
    ) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._verbose = verbose

        if not self._api_key:
            raise ValueError(
                "No Gemini API key provided. Set GOOGLE_API_KEY or pass api_key=."
            )

        # ── Assemble the LlmAgent ────────────────────────────────────────
        self._agent = LlmAgent(
            name="OeFlakTrackAgent",
            description=(
                "3-D radar tracking agent that calls EKF, simulation, "
                "benchmarking, and coordinate-conversion tools."
            ),
            model=model,
            instruction=SYSTEM_INSTRUCTION,
            tools=ALL_TOOLS,
            # Pydantic schema — ADK enforces this on the final response turn
            output_schema=TrackingAgentResponse,
            output_key="tracking_response",   # written to session state
            generate_content_config=_build_generation_config(),
        )

        # ── Session service + runner ─────────────────────────────────────
        self._session_svc = InMemorySessionService()
        self._runner = Runner(
            agent=self._agent,
            app_name=_APP_NAME,
            session_service=self._session_svc,
        )

    # ── Async core ────────────────────────────────────────────────────────

    async def query_async(self, prompt: str) -> TrackingAgentResponse:
        """
        Submit a single query and return a validated TrackingAgentResponse.

        Creates a fresh stateless session for each call — no conversation
        history bleeds between queries.
        """
        session = await self._session_svc.create_session(
            app_name=_APP_NAME,
            user_id=_USER_ID,
        )

        user_message = types.Content(
            role="user",
            parts=[types.Part(text=prompt)],
        )

        final_text: str | None = None
        t_start = time.perf_counter()

        async for event in self._runner.run_async(
            user_id=_USER_ID,
            session_id=session.id,
            new_message=user_message,
        ):
            if self._verbose:
                _log_event(event)

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_text = event.content.parts[0].text

        elapsed_ms = (time.perf_counter() - t_start) * 1_000

        if final_text is None:
            return TrackingAgentResponse(
                status="error",
                intent="No response received from the model.",
                tool_called="none",
                parameters={},
                result={},
                summary="The agent produced no final response.",
                error_message=f"Empty response after {elapsed_ms:.1f} ms.",
            )

        response = _parse_response(final_text)

        if self._verbose:
            print(f"\n  [Agent] Round-trip: {elapsed_ms:.1f} ms")

        return response

    # ── Sync convenience wrapper ──────────────────────────────────────────

    def query(self, prompt: str) -> TrackingAgentResponse:
        """Synchronous wrapper around query_async for non-async callers."""
        return asyncio.run(self.query_async(prompt))

    # ── Batch mode ────────────────────────────────────────────────────────

    async def batch_async(
        self, prompts: list[str]
    ) -> list[TrackingAgentResponse]:
        """
        Run a list of queries sequentially (awaiting each) and return results.

        Sequential (not concurrent) to avoid hammering the API rate limit.
        """
        results = []
        for prompt in prompts:
            results.append(await self.query_async(prompt))
        return results

    def batch(self, prompts: list[str]) -> list[TrackingAgentResponse]:
        """Synchronous wrapper around batch_async."""
        return asyncio.run(self.batch_async(prompts))


# ── Event logger (verbose mode) ───────────────────────────────────────────────

def _log_event(event) -> None:
    """Print a compact one-line summary of an ADK event."""
    calls = event.get_function_calls()
    resps = event.get_function_responses()
    if calls:
        for fc in calls:
            print(f"  [Tool→] {fc.name}({json.dumps(dict(fc.args), separators=(',',':'))})")
    if resps:
        for fr in resps:
            snippet = json.dumps(fr.response)[:120]
            print(f"  [←Tool] {fr.name}: {snippet}{'…' if len(json.dumps(fr.response))>120 else ''}")


# ── Report formatter ──────────────────────────────────────────────────────────

def _print_response(resp: TrackingAgentResponse, query: str, idx: int | None = None) -> None:
    """Pretty-print a TrackingAgentResponse to stdout."""
    W = 70
    hdr = f" Query {idx} " if idx is not None else " Result "
    print("\n" + "═" * W)
    print(f"{'═' * ((W - len(hdr)) // 2)}{hdr}{'═' * ((W - len(hdr) + 1) // 2)}")
    print("═" * W)
    print(f"  Q : {query[:W-6]}")
    print(f"  ✦ Status      : {resp.status}")
    print(f"  ✦ Intent      : {resp.intent}")
    print(f"  ✦ Tool        : {resp.tool_called}")
    if resp.parameters:
        params_str = json.dumps(resp.parameters, separators=(",", ":"))
        print(f"  ✦ Parameters  : {params_str[:W-18]}")
    print(f"  ✦ Summary     : {resp.summary}")
    if resp.error_message:
        print(f"  ✦ Error       : {resp.error_message}")
    # Show top-level result keys (not full data to keep output readable)
    if resp.result:
        keys = list(resp.result.keys())
        print(f"  ✦ Result keys : {keys}")
    print("═" * W)


# ── Demo queries ──────────────────────────────────────────────────────────────

_DEMO_QUERIES: list[str] = [
    ex["user"] for ex in FEW_SHOT_EXAMPLES
]


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_runner.py",
        description="OeFlakTrack Google ADK tracking agent CLI",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--demo",
        action="store_true",
        help="Run the 5 pre-canned few-shot demo queries and exit.",
    )
    group.add_argument(
        "--query", "-q",
        metavar="PROMPT",
        help="Run a single query and print the result.",
    )
    group.add_argument(
        "--repl",
        action="store_true",
        help="Start an interactive REPL (default when no flag given).",
    )
    p.add_argument(
        "--api-key",
        metavar="KEY",
        default=None,
        help="Gemini API key. Defaults to GOOGLE_API_KEY env var.",
    )
    p.add_argument(
        "--model",
        default=_MODEL,
        help=f"Gemini model ID (default: {_MODEL}).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print intermediate tool-call events.",
    )
    return p


async def _run_demo(agent: TrackingAgent) -> None:
    print("\n" + "━" * 70)
    print(" OeFlakTrack ADK Agent — Demo Mode (5 few-shot queries)")
    print("━" * 70)
    print(f" Model       : {_MODEL}")
    print(f" Temperature : {_TEMPERATURE}  |  Seed: {_SEED}  |  top_p: {_TOP_P}")
    print(f" Output schema: TrackingAgentResponse (Pydantic, 7 fields)")
    print("━" * 70)

    for i, q in enumerate(_DEMO_QUERIES, start=1):
        print(f"\n[{i}/5] Querying: {q[:65]}…")
        t0   = time.perf_counter()
        resp = await agent.query_async(q)
        dt   = (time.perf_counter() - t0) * 1_000
        _print_response(resp, q, idx=i)
        print(f"  Round-trip: {dt:.0f} ms")

    print("\n" + "━" * 70)
    print(" Demo complete.")
    print("━" * 70)


async def _run_repl(agent: TrackingAgent) -> None:
    print("\n OeFlakTrack ADK Agent — Interactive REPL")
    print(" Type a query, or 'quit' / Ctrl-C to exit.\n")
    while True:
        try:
            prompt = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break
        t0   = time.perf_counter()
        resp = await agent.query_async(prompt)
        dt   = (time.perf_counter() - t0) * 1_000
        _print_response(resp, prompt)
        print(f"  Round-trip: {dt:.0f} ms")


def main() -> None:
    cli  = _build_cli()
    args = cli.parse_args()

    try:
        agent = TrackingAgent(
            api_key=args.api_key,
            model=args.model,
            verbose=args.verbose,
        )
    except ValueError as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.demo:
        asyncio.run(_run_demo(agent))
    elif args.query:
        resp = agent.query(args.query)
        _print_response(resp, args.query)
        # Also dump raw JSON for piping
        print("\n--- JSON payload ---")
        print(resp.model_dump_json(indent=2))
    else:
        # Default: interactive REPL
        asyncio.run(_run_repl(agent))


if __name__ == "__main__":
    main()
