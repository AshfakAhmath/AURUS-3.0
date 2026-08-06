"""Groq MCP Agent Service for autonomous tool calling from the dashboard."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

try:
    from groq import Groq
except ImportError:
    Groq = None


# Standard OpenAI/Groq Tool Schemas mapping to AURUS MCP capabilities
AURUS_MCP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_rover",
            "description": "Move AURUS in a direction for a given duration, then stop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": [
                            "forward",
                            "backward",
                            "left",
                            "right",
                            "forward_left",
                            "forward_right",
                            "backward_left",
                            "backward_right",
                        ],
                        "description": "Direction to drive.",
                    },
                    "speed": {
                        "type": "number",
                        "description": "Speed from 0.0 to 1.0 (default 1.0).",
                        "default": 1.0,
                    },
                    "duration": {
                        "type": "number",
                        "description": "Seconds to drive (default 1.0).",
                        "default": 1.0,
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spin_rover",
            "description": "Spin AURUS in place (rotate without translating).",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "default": "left",
                    },
                    "speed": {
                        "type": "number",
                        "default": 1.0,
                    },
                    "duration": {
                        "type": "number",
                        "default": 1.0,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_rover",
            "description": "Immediately stop all motors. Call after movement or for emergency stop.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_sensors",
            "description": "Read all 5 ultrasonic distance sensors in centimeters and check if front/rear are clear.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": "Capture a photo from the camera and check vision target identity.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "speak_text",
            "description": "Make AURUS speak a sentence aloud using text-to-speech through the speaker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The words to say aloud (max 300 chars)."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Store a fact in AURUS persistent SQLite memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The text to remember."},
                    "user_id": {"type": "integer", "default": 1},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memories",
            "description": "Retrieve all stored memories and facts about a person.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "integer", "default": 1}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rover_status",
            "description": "Get current status of AURUS including position, mode, battery/voltage simulation, and health.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perform_animation",
            "description": "Play a predefined physical motion animation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": ["wiggle", "shiver", "spin"],
                        "description": "'wiggle' (happy), 'shiver' (scared), 'spin' (curious).",
                    }
                },
                "required": ["name"],
            },
        },
    },
]


class MCPAgentService:
    def __init__(self, runtime: Any, api_key: str | None = None, model: str | None = None):
        self.runtime = runtime
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        # Llama 3.3 70B Versatile is currently Groq's best tool-calling model
        self.model = model or os.getenv("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")
        self._client = None
        if self.api_key and Groq is not None:
            try:
                self._client = Groq(api_key=self.api_key, timeout=15.0, max_retries=1)
            except Exception as exc:
                print(f"[MCPAgentService] Initialization warning: {exc}")

    @property
    def ready(self) -> bool:
        return self._client is not None

    def execute_command(self, user_command: str) -> dict:
        """Executes a natural language instructions using Groq tool calling loop."""
        if not self._client:
            msg = "⚡ Groq MCP Agent is unavailable. Please check that 'groq' is installed and GROQ_API_KEY is set in your .env file."
            self.runtime.emit(
                "conversation",
                {
                    "source": "agent",
                    "transcript": user_command,
                    "response": msg,
                    "provider": "local",
                    "fallback_reason": "missing key/sdk",
                },
            )
            return {"status": "error", "message": msg}

        # Inform dashboard that agent started thinking
        self.runtime.emit(
            "conversation",
            {
                "source": "agent",
                "transcript": f"[MCP Agent] {user_command}",
                "response": "🧠 Groq Llama-3 agent received objective. Planning tool sequence...",
                "provider": "groq-mcp",
                "fallback_reason": "",
            },
        )

        system_prompt = (
            "You are the autonomous cloud AI brain controlling AURUS, a 4-wheel mecanum rover with "
            "5 ultrasonic sensors, a camera, text-to-speech, and persistent SQLite memory. "
            "When given a goal, invoke the appropriate tools to accomplish it safely. "
            "Always check sensors if safety or obstacles are mentioned. Keep verbal speech concise and warm."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_command[:1000]},
        ]

        iterations = 0
        max_iterations = 6
        final_summary = ""

        try:
            while iterations < max_iterations:
                iterations += 1
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=AURUS_MCP_TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=500,
                )

                response_message = completion.choices[0].message
                tool_calls = response_message.tool_calls

                if not tool_calls:
                    final_summary = (response_message.content or "Task completed.").strip()
                    break

                messages.append(response_message)

                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except Exception:
                        args = {}

                    # Notify dashboard of live tool call
                    self.runtime.emit(
                        "mcp_tool_exec",
                        {
                            "id": tool_call.id,
                            "tool": fn_name,
                            "arguments": args,
                            "status": "calling",
                        },
                    )
                    self.runtime.emit(
                        "conversation",
                        {
                            "source": "tool",
                            "transcript": "",
                            "response": f"⚡ Calling tool: {fn_name}({json.dumps(args)})",
                            "provider": "mcp-tool",
                            "fallback_reason": "",
                        },
                    )

                    # Execute tool against live runtime
                    result_json = self._dispatch_tool(fn_name, args)

                    self.runtime.emit(
                        "mcp_tool_exec",
                        {
                            "id": tool_call.id,
                            "tool": fn_name,
                            "arguments": args,
                            "result": json.loads(result_json) if result_json.startswith("{") else result_json,
                            "status": "completed",
                        },
                    )
                    self.runtime.emit(
                        "conversation",
                        {
                            "source": "tool_res",
                            "transcript": "",
                            "response": f"➔ {fn_name} returned: {result_json[:150]}",
                            "provider": "mcp-res",
                            "fallback_reason": "",
                        },
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": str(result_json),
                        }
                    )

            if not final_summary:
                final_summary = f"Completed agent execution sequence after {iterations} turns."

            self.runtime.emit(
                "conversation",
                {
                    "source": "agent_done",
                    "transcript": "",
                    "response": f"🤖 {final_summary}",
                    "provider": "groq-mcp",
                    "fallback_reason": "",
                },
            )
            return {"status": "ok", "summary": final_summary, "turns": iterations}

        except Exception as exc:
            err_msg = f"Groq Agent execution error: {str(exc)[:250]}"
            self.runtime.emit(
                "conversation",
                {
                    "source": "error",
                    "transcript": "",
                    "response": err_msg,
                    "provider": "groq-mcp",
                    "fallback_reason": "exception",
                },
            )
            return {"status": "error", "message": err_msg}

    def _dispatch_tool(self, name: str, args: dict) -> str:
        """Execute the requested MCP tool using the active robot runtime."""
        try:
            if name == "move_rover":
                direction = str(args.get("direction", "forward"))
                speed = float(args.get("speed", 1.0))
                duration = float(args.get("duration", 1.0))
                speed = max(0.0, min(1.0, speed))
                duration = max(0.1, min(10.0, duration))

                direction_map = {
                    "forward": (speed, 0.0, 0.0),
                    "backward": (-speed, 0.0, 0.0),
                    "left": (0.0, speed, 0.0),
                    "right": (0.0, -speed, 0.0),
                    "forward_left": (speed, speed, 0.0),
                    "forward_right": (speed, -speed, 0.0),
                    "backward_left": (-speed, speed, 0.0),
                    "backward_right": (-speed, -speed, 0.0),
                }
                vels = direction_map.get(direction.lower(), (speed, 0.0, 0.0))
                self.runtime.arbiter.command("mcp-agent", vels[0], vels[1], vels[2], ttl=duration, priority=60)
                time.sleep(duration)
                self.runtime.arbiter.halt("mcp-agent-stop")
                state = self.runtime.driver.get_simulation_state()
                return json.dumps(
                    {
                        "status": "ok",
                        "action": f"Moved {direction} for {duration}s",
                        "simulation_mode": self.runtime.driver.is_simulation,
                        "position": {
                            "x_cm": round(state.get("x", 0.0), 1),
                            "y_cm": round(state.get("y", 0.0), 1),
                        },
                    }
                )

            elif name == "spin_rover":
                direction = str(args.get("direction", "left"))
                speed = float(args.get("speed", 1.0))
                duration = float(args.get("duration", 1.0))
                omega = speed if direction.lower() == "left" else -speed
                self.runtime.arbiter.command("mcp-agent", 0.0, 0.0, omega, ttl=duration, priority=60)
                time.sleep(duration)
                self.runtime.arbiter.halt("mcp-agent-stop")
                return json.dumps({"status": "ok", "action": f"Spun {direction} for {duration}s"})

            elif name == "stop_rover":
                self.runtime.arbiter.halt("mcp-agent")
                return json.dumps({"status": "ok", "message": "All motors stopped."})

            elif name == "read_sensors":
                snapshot = self.runtime.sensor_sampler.get_snapshot()
                return json.dumps(
                    {
                        "status": "ok",
                        "distances_cm": {
                            "fl": round(snapshot.fl, 1),
                            "f": round(snapshot.f, 1),
                            "fr": round(snapshot.fr, 1),
                            "rl": round(snapshot.rl, 1),
                            "rr": round(snapshot.rr, 1),
                        },
                        "front_min_cm": round(snapshot.front_min, 1),
                        "rear_min_cm": round(snapshot.rear_min, 1),
                        "front_clear": snapshot.front_min > 35.0,
                    }
                )

            elif name == "capture_image":
                snapshot = self.runtime.vision.get_snapshot()
                identity = snapshot.identity
                return json.dumps(
                    {
                        "status": "ok",
                        "camera_healthy": snapshot.healthy,
                        "backend": snapshot.backend,
                        "detected_person": {
                            "name": identity.name or "Unknown",
                            "status": identity.status,
                            "confidence": round(float(identity.confidence or 0.0), 2),
                        },
                    }
                )

            elif name == "speak_text":
                text = str(args.get("text", ""))[:300]
                self.runtime.tts.speak(text)
                return json.dumps({"status": "ok", "message": f"Spoke text: {text}"})

            elif name == "remember_fact":
                fact = str(args.get("fact", ""))
                user_id = int(args.get("user_id", 1))
                self.runtime.repository.remember(user_id, fact)
                return json.dumps({"status": "ok", "message": f"Remembered: {fact}"})

            elif name == "recall_memories":
                user_id = int(args.get("user_id", 1))
                facts = self.runtime.repository.memories_for(user_id)
                return json.dumps({"status": "ok", "count": len(facts), "memories": facts[:10]})

            elif name == "get_rover_status":
                health = self.runtime.health()
                state = self.runtime.driver.get_simulation_state()
                return json.dumps(
                    {
                        "status": "ok",
                        "mode": str(self.runtime.arbiter.mode.value),
                        "simulation_mode": self.runtime.driver.is_simulation,
                        "health": health,
                        "position": {
                            "x_cm": round(state.get("x", 0.0), 1),
                            "y_cm": round(state.get("y", 0.0), 1),
                            "heading_deg": round(state.get("theta", 0.0) * 57.2958, 1),
                        },
                    }
                )

            elif name == "perform_animation":
                anim = str(args.get("name", "wiggle")).lower()
                animations = {
                    "wiggle": lambda: self.runtime.driver.wiggle(duration=1.5),
                    "shiver": lambda: self.runtime.driver.shiver(duration=1.2),
                    "spin": lambda: self.runtime.driver.spin(duration=1.5),
                }
                func = animations.get(anim)
                if func:
                    func()
                    time.sleep(1.8)
                    return json.dumps({"status": "ok", "animation": anim})
                return json.dumps({"status": "error", "message": f"Unknown animation: {anim}"})

            else:
                return json.dumps({"status": "error", "message": f"Unknown tool name: {name}"})

        except Exception as exc:
            return json.dumps({"status": "error", "message": f"Tool execution error: {exc}"})
