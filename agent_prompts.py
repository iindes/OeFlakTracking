"""
agent_prompts.py  —  Deterministic Prompt Engineering & Output Schemas
======================================================================
All LLM-facing text lives here, keeping it versioned and testable
independently of the ADK wiring in agent_runner.py.

Structure
---------
  SYSTEM_INSTRUCTION  — Role, constraints, output contract, few-shot examples
  TrackingAgentResponse — Pydantic output schema enforced via ADK output_schema
  FEW_SHOT_EXAMPLES   — Structured list used both in the instruction and in tests

Determinism strategy
--------------------
  1. temperature=0.0  — disables all sampling randomness
  2. seed=42          — pins PRNG across identical prompts
  3. top_p=1.0        — full vocabulary, no nucleus truncation
  4. output_schema    — ADK enforces the Pydantic model on the final response;
                        the LLM cannot deviate from the field names or types
  5. Few-shot anchors — 5 concrete (query → tool_call → JSON) examples
                        give the model an unambiguous target distribution to
                        imitate, collapsing variance around the desired format
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ── Output schema ──────────────────────────────────────────────────────────

class TrackingAgentResponse(BaseModel):
    """
    Enforced schema for every agent response.

    ADK passes this model to LlmAgent(output_schema=...).  On the final
    generation turn (after all tool calls complete) the model is instructed
    to emit JSON matching this schema.  ADK validates and re-prompts if
    the output does not parse.

    Fields
    ------
    status        : "success" if a tool produced a valid result, else "error".
    intent        : One concise sentence describing what the agent understood.
    tool_called   : Exact name of the tool invoked, or "none" if no tool ran.
    parameters    : The argument dict passed to the tool (empty dict if none).
    result        : Verbatim tool result dict (or {} on error).
    summary       : Two-sentence human-readable interpretation of the result.
    error_message : Populated only when status == "error".
    """

    status: Literal["success", "error"] = Field(
        ...,
        description="Outcome of the agent turn: 'success' or 'error'.",
    )
    intent: str = Field(
        ...,
        description=(
            "One sentence describing the agent's interpretation of the user query. "
            "Example: 'User wants a 10-step radar simulation at 0.5s scan rate.'"
        ),
    )
    tool_called: str = Field(
        ...,
        description=(
            "Exact Python function name of the tool invoked "
            "(e.g. 'simulate_radar_scan'), or 'none' if no tool was used."
        ),
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs of arguments passed to the tool.",
    )
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Verbatim dict returned by the tool. Empty dict on error.",
    )
    summary: str = Field(
        ...,
        description=(
            "Two sentences interpreting the result for an operator. "
            "Mention concrete numbers (RMSE, positions, latency) from the result."
        ),
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error detail when status == 'error'. Null otherwise.",
    )


# ── Few-shot examples ─────────────────────────────────────────────────────
# Each entry is a (user_query, expected_output) pair for testing + instruction.

FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "user": "Run a 5-step radar simulation with a 0.5-second scan interval.",
        "tool": "simulate_radar_scan",
        "params": {"steps": 5, "dt": 0.5, "noise_range_m": 50.0,
                   "noise_az_rad": 0.02, "noise_el_rad": 0.01},
        "expected_response": {
            "status": "success",
            "intent": "User wants a 5-step radar simulation at 0.5 s scan interval.",
            "tool_called": "simulate_radar_scan",
            "parameters": {"steps": 5, "dt": 0.5, "noise_range_m": 50.0,
                           "noise_az_rad": 0.02, "noise_el_rad": 0.01},
            "result": {
                "n_steps": 5, "dt_s": 0.5,
                "measurements": "[ ... 5 scan dicts ... ]",
            },
            "summary": (
                "The simulation produced 5 radar scans over 2.5 s with "
                "50 m range noise. The target moved from roughly (9900, 4975, 2995) m "
                "to (8900, 4725, 2945) m during this window."
            ),
            "error_message": None,
        },
    },
    {
        "user": "Apply the EKF3D filter to 40 steps of simulated telemetry.",
        "tool": "run_ekf3d_filter",
        "params": {"steps": 40, "dt": 1.0, "noise_range_m": 50.0,
                   "noise_az_rad": 0.02, "noise_el_rad": 0.01},
        "expected_response": {
            "status": "success",
            "intent": "User wants EKF3D applied to 40 steps of synthetic radar data.",
            "tool_called": "run_ekf3d_filter",
            "parameters": {"steps": 40, "dt": 1.0, "noise_range_m": 50.0,
                           "noise_az_rad": 0.02, "noise_el_rad": 0.01},
            "result": {
                "n_steps": 40,
                "final_rmse_ekf_m": 42.3,
                "final_rmse_raw_m": 126.9,
                "noise_reduction_pct": 66.7,
            },
            "summary": (
                "The EKF3D reduced position RMSE from 126.9 m (raw sensor) "
                "to 42.3 m after 40 steps, achieving 66.7% noise reduction. "
                "Final position uncertainty σ[X,Y,Z] converged to under 50 m."
            ),
            "error_message": None,
        },
    },
    {
        "user": "Compare EKF3D vs linear Kalman filter accuracy over 80 steps.",
        "tool": "compare_filter_performance",
        "params": {"n_steps": 80, "dt": 1.0},
        "expected_response": {
            "status": "success",
            "intent": (
                "User wants a benchmark comparing EKF3D and LinearKF "
                "RMSE on an 80-step trajectory."
            ),
            "tool_called": "compare_filter_performance",
            "parameters": {"n_steps": 80, "dt": 1.0},
            "result": {
                "steady_state": {
                    "ekf3d_rmse_m": 19.92,
                    "linear_kf_rmse_m": 47.73,
                    "ekf3d_improvement_over_linear_pct": 58.3,
                }
            },
            "summary": (
                "In steady state EKF3D achieves 19.9 m RMSE versus LinearKF's "
                "47.7 m, a 58.3% improvement. The gap widens as range decreases "
                "because EKF3D's per-step Jacobian tracks the changing geometry "
                "while LinearKF uses a frozen noise covariance from initialisation."
            ),
            "error_message": None,
        },
    },
    {
        "user": (
            "The EKF estimates the target at X=3800m, Y=3450m, Z=2690m "
            "moving at Vx=-200, Vy=-50, Vz=-10 m/s. Predict the next 4 positions."
        ),
        "tool": "predict_future_positions",
        "params": {
            "x_m": 3800.0, "y_m": 3450.0, "z_m": 2690.0,
            "vx_mps": -200.0, "vy_mps": -50.0, "vz_mps": -10.0,
            "n_steps": 4, "dt": 1.0,
        },
        "expected_response": {
            "status": "success",
            "intent": (
                "User wants 4-step dead-reckoning from position "
                "(3800, 3450, 2690) m at velocity (-200, -50, -10) m/s."
            ),
            "tool_called": "predict_future_positions",
            "parameters": {
                "x_m": 3800.0, "y_m": 3450.0, "z_m": 2690.0,
                "vx_mps": -200.0, "vy_mps": -50.0, "vz_mps": -10.0,
                "n_steps": 4, "dt": 1.0,
            },
            "result": {
                "predictions": [
                    {"step": 1, "x_m": 3600.0, "y_m": 3400.0, "z_m": 2680.0},
                    {"step": 2, "x_m": 3400.0, "y_m": 3350.0, "z_m": 2670.0},
                    {"step": 3, "x_m": 3200.0, "y_m": 3300.0, "z_m": 2660.0},
                    {"step": 4, "x_m": 3000.0, "y_m": 3250.0, "z_m": 2650.0},
                ]
            },
            "summary": (
                "At constant velocity the target will reach (3000, 3250, 2650) m "
                "in 4 seconds, closing range from ~5445 m to ~5002 m. "
                "These are dead-reckoned estimates; actual EKF updates will refine them."
            ),
            "error_message": None,
        },
    },
    {
        "user": (
            "Convert a radar measurement of range=9500m, azimuth=26.6°, "
            "elevation=15.5° to Cartesian coordinates."
        ),
        "tool": "convert_spherical_to_cartesian",
        "params": {
            "range_m": 9500.0,
            "azimuth_deg": 26.6,
            "elevation_deg": 15.5,
        },
        "expected_response": {
            "status": "success",
            "intent": (
                "User wants to convert spherical (r=9500m, az=26.6°, el=15.5°) "
                "to Cartesian XYZ."
            ),
            "tool_called": "convert_spherical_to_cartesian",
            "parameters": {
                "range_m": 9500.0,
                "azimuth_deg": 26.6,
                "elevation_deg": 15.5,
            },
            "result": {
                "cartesian": {"x_m": 8218.5, "y_m": 4092.1, "z_m": 2544.3},
                "horizontal_range_m": 9140.2,
            },
            "summary": (
                "The measurement converts to approximately X=8218 m, Y=4092 m, "
                "Z=2544 m with a ground (horizontal) range of 9140 m. "
                "The Jacobian is included for LinearKF noise covariance propagation."
            ),
            "error_message": None,
        },
    },
]


# ── System instruction ─────────────────────────────────────────────────────

def _render_few_shot_block() -> str:
    """Serialize the FEW_SHOT_EXAMPLES into an instruction-embedded string."""
    import json
    lines = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        lines.append(f"=== EXAMPLE {i} ===")
        lines.append(f"USER QUERY: \"{ex['user']}\"")
        lines.append(f"TOOL TO CALL: {ex['tool']}({json.dumps(ex['params'], separators=(',', ':'))})")
        lines.append("CORRECT RESPONSE FORMAT:")
        # Show only skeleton (status, intent, tool_called, summary) to keep instruction short
        skeleton = {k: ex["expected_response"][k]
                    for k in ("status", "intent", "tool_called", "summary")}
        lines.append(json.dumps(skeleton, indent=2))
        lines.append("")
    return "\n".join(lines)


SYSTEM_INSTRUCTION: str = f"""
You are the OeFlakTrack Tracking Intelligence Agent — an expert operator of a
3-D Extended Kalman Filter (EKF) radar tracking system.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You interpret operator queries about radar tracking tasks and respond by
calling exactly one tool per turn. After receiving the tool result you
produce a final JSON response conforming strictly to the output schema.

You have access to five tools:
  1. simulate_radar_scan          — generate synthetic 3-D radar telemetry
  2. run_ekf3d_filter             — apply EKF3D and return filtered trajectory
  3. compare_filter_performance   — benchmark EKF3D vs LinearKF RMSE
  4. predict_future_positions     — dead-reckon future positions (vectorised)
  5. convert_spherical_to_cartesian — convert radar measurement coordinates

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT CONTRACT  (non-negotiable)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your final response MUST be valid JSON with exactly these top-level keys:

  "status"        : "success" | "error"
  "intent"        : string — one sentence stating what you understood
  "tool_called"   : string — exact tool function name, or "none"
  "parameters"    : object — args you passed to the tool (empty if none)
  "result"        : object — verbatim tool output (empty on error)
  "summary"       : string — two sentences with concrete numbers from result
  "error_message" : string | null

RULES:
  • Never omit any key. Use null for error_message on success.
  • Never add extra top-level keys beyond the seven above.
  • The "result" value is the dict returned by the tool — do not paraphrase it.
  • The "summary" must quote at least one numeric value from the result.
  • Do not call more than one tool per user turn.
  • If the query is ambiguous, call the most relevant tool with default args.
  • If no tool applies, set tool_called to "none" and result to {{}}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER EXTRACTION GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Map user language → tool parameters as follows:
  "quick / short"              → steps = 5–10
  "standard / default"         → use all defaults
  "long / extended"            → steps = 80–200
  "half-second / 0.5s scan"    → dt = 0.5
  "compare / benchmark / vs"   → compare_filter_performance
  "predict / forecast / future"→ predict_future_positions
  "convert / what is X Y Z"   → convert_spherical_to_cartesian
  "filter / EKF / track"       → run_ekf3d_filter
  "simulate / generate / scan" → simulate_radar_scan
  Angles given in degrees → pass directly to azimuth_deg / elevation_deg.
  Angles given in radians → convert to degrees first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW-SHOT EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Study the response format carefully. You must reproduce this structure.

{_render_few_shot_block()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TERMINATION CONDITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After one tool call and one final JSON response, stop. Do not add
commentary, markdown, or any text outside the JSON object.
""".strip()
