"""Bridge to the local OpenCode CLI.

This environment has NO API keys, so the LLM is reached through:
    opencode run --format json -m <model>   (prompt on stdin)

stdout is NDJSON: one JSON object per line. Assistant text arrives as events
with part.type == "text"; provider failures arrive as type == "error" with a
non-zero exit code. Both parse paths below are unit-tested without the network.
"""
import json
import re
import shutil
import subprocess

from .config import LLM_TIMEOUT_SECONDS, MODEL


class LLMError(RuntimeError):
    pass


def run_llm(prompt: str, model: str = MODEL, timeout: int = LLM_TIMEOUT_SECONDS) -> str:
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
        raise LLMError(f"LLM call timed out after {timeout}s") from exc
    except OSError as exc:
        raise LLMError(f"could not launch opencode CLI: {exc}") from exc
    return parse_run_output(proc.stdout, proc.returncode, proc.stderr)


def parse_run_output(stdout: str, returncode: int = 0, stderr: str = "") -> str:
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
            errors.append(err.get("message", str(err)) if isinstance(err, dict) else str(err))
        part = obj.get("part") or {}
        if part.get("type") == "text" and part.get("text"):
            texts.append(part["text"].strip())

    if errors:
        raise LLMError(f"OpenCode provider error: {errors[0]}")
    if returncode != 0:
        detail = (stderr or "").strip()[:500] or "no stderr output"
        raise LLMError(f"opencode run failed (exit {returncode}): {detail}")
    output = "\n".join(t for t in texts if t)
    if not output:
        raise LLMError("opencode run produced no text output")
    return output


def extract_json(text: str) -> dict:
    """Pull the JSON object out of a model response (tolerates ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("model output contains no JSON object")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"model output is not valid JSON: {exc}") from exc
