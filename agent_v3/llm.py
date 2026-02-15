"""LLM provider abstraction for Agent V3.

Supports OpenAI GPT and Google Gemini ER as interchangeable reasoning backends.
"""

import base64
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("agent_v3.llm")


@dataclass
class ToolCall:
    """A single tool/function call from the model."""
    name: str
    args: dict[str, Any]
    id: str | None = None  # OpenAI provides tool_call IDs


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # Provider-specific response for history management


class LLMProvider(ABC):
    """Abstract LLM provider for agent reasoning + tool calling."""

    @abstractmethod
    def create_messages(self, system_prompt: str, task_prompt: str) -> list[Any]:
        """Create the initial message list for a new task."""
        ...

    @abstractmethod
    def chat(self, messages: list[Any], tools: list[dict]) -> LLMResponse:
        """Send messages with tools, return unified response."""
        ...

    @abstractmethod
    def append_assistant_response(self, messages: list[Any], response: LLMResponse) -> None:
        """Append the model's response to the message history."""
        ...

    @abstractmethod
    def append_tool_result(self, messages: list[Any], tool_call: ToolCall,
                           result_str: str) -> None:
        """Append a tool result to the message history."""
        ...

    @abstractmethod
    def append_images(self, messages: list[Any], image_parts: list[Any],
                      text: str | None = None) -> None:
        """Append images (and optional text) as a user message."""
        ...

    @abstractmethod
    def encode_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Any:
        """Encode image bytes for this provider's message format."""
        ...

    @abstractmethod
    def encode_text(self, text: str) -> Any:
        """Encode a text string as a content part for this provider."""
        ...


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider using chat.completions with function calling."""

    def __init__(self, openai_api_key: str, helicone_api_key: str,
                 model: str = "gpt-5.2", debug: bool = False,
                 reasoning_effort: str = "low"):
        from agent.vlm_agent import create_openai_client
        self.client = create_openai_client(openai_api_key, helicone_api_key, debug=debug)
        self.model = model
        self.reasoning_effort = reasoning_effort
        logger.info(f"OpenAI provider initialized: model={model}")

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert tool declarations to OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": decl["name"],
                    "description": decl["description"],
                    "parameters": decl["parameters"],
                },
            }
            for decl in tools
        ]

    def create_messages(self, system_prompt: str, task_prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        openai_tools = self._convert_tools(tools)

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": openai_tools,
            "tool_choice": "auto",
        }

        if "o1" in self.model.lower() or "o3" in self.model.lower():
            request_kwargs["extra_body"] = {
                "reasoning_effort": self.reasoning_effort,
            }

        response = self.client.chat.completions.create(**request_kwargs)

        if not response.choices:
            return LLMResponse(text="Model returned empty response.")

        message = response.choices[0].message
        text = message.content or None

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                tool_calls.append(ToolCall(name=tc.function.name, args=args, id=tc.id))

        return LLMResponse(text=text, tool_calls=tool_calls, raw=message)

    def append_assistant_response(self, messages: list[dict], response: LLMResponse) -> None:
        messages.append(response.raw.model_dump())

    def append_tool_result(self, messages: list[dict], tool_call: ToolCall,
                           result_str: str) -> None:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_str,
        })

    def append_images(self, messages: list[dict], image_parts: list[Any],
                      text: str | None = None) -> None:
        content = list(image_parts)
        if text:
            content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content})

    def encode_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64}",
                "detail": "high",
            },
        }

    def encode_text(self, text: str) -> dict:
        return {"type": "text", "text": text}


class GeminiProvider(LLMProvider):
    """Google Gemini ER provider using generate_content with tool calling."""

    def __init__(self, google_api_key: str,
                 model: str = "gemini-robotics-er-1.5-preview",
                 thinking_budget: int = 1024):
        from google import genai
        from google.genai import types
        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=google_api_key)
        self.model = model
        self.thinking_budget = thinking_budget
        logger.info(f"Gemini provider initialized: model={model}")

    def _convert_tools(self, tools: list[dict]) -> list[Any]:
        """Convert tool declarations to Gemini Tool format."""
        return [self._types.Tool(function_declarations=tools)]

    def create_messages(self, system_prompt: str, task_prompt: str) -> list[Any]:
        types = self._types
        # Gemini has no system role — inject via initial exchange
        return [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_prompt)],
            ),
            types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text="Understood. I'm ready to control the robot arm using my vision "
                    "and tool-calling capabilities. Give me a task and I'll start by "
                    "looking at the scene."
                )],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=task_prompt)],
            ),
        ]

    def chat(self, messages: list[Any], tools: list[dict]) -> LLMResponse:
        types = self._types
        gemini_tools = self._convert_tools(tools)

        response = self.client.models.generate_content(
            model=self.model,
            contents=messages,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                temperature=0.3,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ),
            ),
        )

        if not response.candidates:
            return LLMResponse(text="Model returned empty response.")

        candidate = response.candidates[0]
        model_parts = (candidate.content and candidate.content.parts) or []

        if not model_parts:
            return LLMResponse(text="Model returned empty response.")

        # Collect text
        text_parts = [p.text for p in model_parts if p.text]
        text = "".join(text_parts) if text_parts else None

        # Collect tool calls
        tool_calls = []
        for p in model_parts:
            if p.function_call:
                args = dict(p.function_call.args) if p.function_call.args else {}
                tool_calls.append(ToolCall(name=p.function_call.name, args=args))

        return LLMResponse(text=text, tool_calls=tool_calls, raw=candidate.content)

    def append_assistant_response(self, messages: list[Any], response: LLMResponse) -> None:
        messages.append(response.raw)

    def append_tool_result(self, messages: list[Any], tool_call: ToolCall,
                           result_str: str) -> None:
        types = self._types
        part = types.Part.from_function_response(
            name=tool_call.name,
            response={"result": result_str},
        )
        # Batch tool results — check if last message is already a user turn with
        # function responses. If so, append to it; otherwise create new.
        if (messages and hasattr(messages[-1], "role") and messages[-1].role == "user"
                and messages[-1].parts
                and any(hasattr(p, "function_response") and p.function_response
                        for p in messages[-1].parts)):
            messages[-1].parts.append(part)
        else:
            messages.append(types.Content(role="user", parts=[part]))

    def append_images(self, messages: list[Any], image_parts: list[Any],
                      text: str | None = None) -> None:
        types = self._types
        parts = list(image_parts)
        if text:
            parts.append(types.Part.from_text(text=text))
        messages.append(types.Content(role="user", parts=parts))

    def encode_image(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Any:
        types = self._types
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    def encode_text(self, text: str) -> Any:
        return self._types.Part.from_text(text=text)
