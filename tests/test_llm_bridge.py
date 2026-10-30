"""Groq bridge tests use a fake OpenAI-compatible client; no network required."""
from types import SimpleNamespace

from app import llm


class FakeClient:
    last_init = None
    last_request = None
    next_response = None

    def __init__(self, **kwargs):
        type(self).last_init = kwargs
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        type(self).last_request = kwargs
        return type(self).next_response


def _response(content="{\"draft\": \"ok\"}", finish_reason="stop", refusal=None):
    message = SimpleNamespace(content=content, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def test_json_prompt_uses_groq_json_and_gpt_oss_options(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("GROQ_MAX_COMPLETION_TOKENS", "4096")
    monkeypatch.delenv("GROQ_REASONING_EFFORT", raising=False)
    FakeClient.next_response = _response()
    monkeypatch.setattr(llm, "OpenAI", FakeClient)

    result = llm.run_llm("Return a structured response.", json_mode=True)

    assert result == '{"draft": "ok"}'
    assert FakeClient.last_request["response_format"] == {"type": "json_object"}
    assert FakeClient.last_request["max_completion_tokens"] == 4096
    assert FakeClient.last_request["extra_body"] == {
        "reasoning_effort": "low",
        "reasoning_format": "hidden",
    }


def test_free_text_prompt_does_not_force_json(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    FakeClient.next_response = _response("plain text answer")
    monkeypatch.setattr(llm, "OpenAI", FakeClient)

    result = llm.run_llm("Answer briefly in plain text.")

    assert result == "plain text answer"
    assert "response_format" not in FakeClient.last_request


def test_empty_response_explains_token_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    FakeClient.next_response = _response(content=None, finish_reason="length")
    monkeypatch.setattr(llm, "OpenAI", FakeClient)

    try:
        llm.run_llm("Return structured output.", json_mode=True)
    except llm.LLMError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected LLMError for empty completion")

    assert "finish_reason=length" in message
    assert "GROQ_MAX_COMPLETION_TOKENS" in message


def test_truncated_nonempty_response_is_reported_as_token_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
    FakeClient.next_response = _response(content='{"draft":', finish_reason="length")
    monkeypatch.setattr(llm, "OpenAI", FakeClient)

    try:
        llm.run_llm("Return structured output.", json_mode=True)
    except llm.LLMError as exc:
        message = str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected LLMError for truncated completion")

    assert "finish_reason=length" in message
    assert "GROQ_MAX_COMPLETION_TOKENS" in message


def test_provider_errors_are_wrapped(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class BrokenClient(FakeClient):
        def _create(self, **kwargs):
            raise RuntimeError("simulated provider outage")

    monkeypatch.setattr(llm, "OpenAI", BrokenClient)
    try:
        llm.run_llm("Answer in plain text.")
    except llm.LLMError as exc:
        assert "simulated provider outage" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected provider error to become LLMError")
