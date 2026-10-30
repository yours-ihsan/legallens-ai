
"""LLM bridge using Groq's OpenAI-compatible API."""

import json
import os
import re

from openai import OpenAI

from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_TIMEOUT_SECONDS,
    MODEL,
)


class LLMError(RuntimeError):
    """Raised when an LLM request or response fails."""
    pass


def run_llm(
    prompt: str,
    model: str = MODEL,
    timeout: int = LLM_TIMEOUT_SECONDS,
    *,
    json_mode: bool = False,
) -> str:
    """Generate a response using Groq."""

    # Prefer the dedicated Groq key, with the existing
    # configuration key as a fallback.
    api_key = (
        os.getenv("GROQ_API_KEY")
        or LLM_API_KEY
    )

    base_url = (
        os.getenv("GROQ_BASE_URL")
        or LLM_BASE_URL
        or "https://api.groq.com/openai/v1"
    )

    selected_model = (
        os.getenv("GROQ_MODEL")
        or model
        or "openai/gpt-oss-20b"
    )

    if not api_key:
        raise LLMError(
            "Groq API key is missing. "
            "Add GROQ_API_KEY to your .env file."
        )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=1,
        )

        # Keep a bounded output budget. Reasoning models can spend part of the
        # completion budget on internal reasoning, so make the budget explicit
        # rather than relying on provider defaults.
        try:
            max_completion_tokens = int(
                os.getenv("GROQ_MAX_COMPLETION_TOKENS", "8192")
            )
        except ValueError as exc:
            raise LLMError(
                "GROQ_MAX_COMPLETION_TOKENS must be a positive integer."
            ) from exc
        if max_completion_tokens < 1:
            raise LLMError(
                "GROQ_MAX_COMPLETION_TOKENS must be a positive integer."
            )

        request_options = {
            "model": selected_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
        }
        if json_mode:
            request_options["response_format"] = {"type": "json_object"}

            # Groq documents these options for GPT-OSS. Hidden reasoning keeps
            # internal reasoning out of the content field expected by our JSON
            # parser, while low effort reserves more of the token budget for
            # the user-facing structured result.
            if selected_model.startswith("openai/gpt-oss-"):
                effort = os.getenv("GROQ_REASONING_EFFORT", "low").strip().lower()
                if effort not in {"low", "medium", "high"}:
                    raise LLMError(
                        "GROQ_REASONING_EFFORT must be low, medium, or high."
                    )
                request_options["extra_body"] = {
                    "reasoning_effort": effort,
                    "reasoning_format": "hidden",
                }

        response = client.chat.completions.create(**request_options)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError(
                f"Groq returned no completion choices (model={selected_model})."
            )

        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message else None

        if isinstance(content, list):
            content = "\n".join(
                str(getattr(part, "text", "") or "")
                for part in content
                if getattr(part, "text", None)
            )

        # A non-empty response can still be truncated JSON. Detect the token
        # limit before handing the text to extract_json().
        finish_reason = getattr(choice, "finish_reason", None) or "unknown"
        if finish_reason == "length":
            raise LLMError(
                "Groq completion reached the token limit "
                f"(model={selected_model}, finish_reason=length). "
                "Increase GROQ_MAX_COMPLETION_TOKENS or reduce "
                "GROQ_REASONING_EFFORT."
            )

        if isinstance(content, str) and content.strip():
            return content.strip()

        # Include safe response metadata so the UI/API error tells us whether
        # the model refused or returned no final text.
        refusal = getattr(message, "refusal", None) if message else None
        details = [f"model={selected_model}", f"finish_reason={finish_reason}"]
        if refusal:
            details.append(f"refusal={str(refusal)[:180]}")

        usage = getattr(response, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        if reasoning_tokens is not None:
            details.append(f"reasoning_tokens={reasoning_tokens}")

        if refusal:
            hint = "The provider/model refused the request."
        else:
            hint = "The provider returned no final text content."
        raise LLMError(
            "Groq returned an empty response (" + ", ".join(details) + "). " + hint
        )

    except LLMError:
        raise

    except Exception as exc:
        # Do not silently switch to OpenCode when Groq fails.
        raise LLMError(
            f"Groq API request failed: {exc}"
        ) from exc


def parse_run_output(
    stdout: str,
    returncode: int = 0,
    stderr: str = "",
) -> str:
    """Compatibility helper for parsing legacy OpenCode JSON output."""

    texts: list[str] = []
    errors: list[str] = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line.startswith("{"):
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if obj.get("type") == "error":
            err = obj.get("error") or {}

            if isinstance(err, dict):
                errors.append(
                    err.get("message", str(err))
                )
            else:
                errors.append(str(err))

        part = obj.get("part") or {}

        if part.get("type") == "text" and part.get("text"):
            texts.append(part["text"].strip())

    if errors:
        raise LLMError(
            f"OpenCode provider error: {errors[0]}"
        )

    if returncode != 0:
        detail = (
            (stderr or "").strip()[:500]
            or "no stderr output"
        )

        raise LLMError(
            f"OpenCode run failed "
            f"(exit {returncode}): {detail}"
        )

    output = "\n".join(
        text for text in texts if text
    )

    if not output:
        raise LLMError(
            "OpenCode produced no text output"
        )

    return output


def extract_json(text: str) -> dict:
    """Extract a JSON object from a model response."""

    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        re.S,
    )

    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end <= start:
        raise LLMError(
            "Model output contains no JSON object"
        )

    try:
        result = json.loads(text[start:end + 1])

        if not isinstance(result, dict):
            raise LLMError(
                "Model output must be a JSON object"
            )

        return result

    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Model output is not valid JSON: {exc}"
        ) from exc
