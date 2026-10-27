
"""LLM bridge supporting Gemini API and local OpenCode CLI."""
import json
import re
import shutil
import subprocess

from .config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_TIMEOUT_SECONDS,
    MODEL,
)


class LLMError(RuntimeError):
    pass


def run_llm(
    prompt: str,
    model: str = MODEL,
    timeout: int = LLM_TIMEOUT_SECONDS,
) -> str:
    # Use an OpenAI-compatible API when configured.
    if LLM_BASE_URL:
        if not LLM_API_KEY:
            raise LLMError(
                "LLM API URL is configured, but the API key is missing."
            )

        try:
            from openai import OpenAI

            client = OpenAI(
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                timeout=timeout,
            )
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError("LLM API returned an empty response.")

            return content

        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"LLM API request failed: {exc}"
            ) from exc

    # Keep the existing local OpenCode fallback.
    exe = shutil.which("opencode") or "opencode"
    try:
        proc = subprocess.run(
            [exe, "run", "--format", "json", "-m", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMError(
            f"LLM call timed out after {timeout}s"
        ) from exc
    except OSError as exc:
        raise LLMError(
            f"Could not launch OpenCode CLI: {exc}"
        ) from exc

    return parse_run_output(
        proc.stdout, proc.returncode, proc.stderr
    )


def parse_run_output(
    stdout: str, returncode: int = 0, stderr: str = ""
) -> str:
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
            errors.append(
                err.get("message", str(err))
                if isinstance(err, dict)
                else str(err)
            )

        part = obj.get("part") or {}
        if part.get("type") == "text" and part.get("text"):
            texts.append(part["text"].strip())

    if errors:
        raise LLMError(f"OpenCode provider error: {errors[0]}")

    if returncode != 0:
        detail = (stderr or "").strip()[:500] or "no stderr output"
        raise LLMError(
            f"OpenCode run failed (exit {returncode}): {detail}"
        )

    output = "\n".join(t for t in texts if t)
    if not output:
        raise LLMError("OpenCode produced no text output")

    return output


def extract_json(text: str) -> dict:
    """Extract a JSON object from a model response."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("Model output contains no JSON object")

    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(
            f"Model output is not valid JSON: {exc}"
        ) from exc
