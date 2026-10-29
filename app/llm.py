
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

        response = client.chat.completions.create(
            model=selected_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise LLMError(
                "Groq returned an empty response."
            )

        return content.strip()

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
