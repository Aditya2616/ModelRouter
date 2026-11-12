import json

import httpx
import pytest

from modelrouter.pricing import PriceTable, UnknownModelError
from modelrouter.providers import AnthropicProvider, OpenAIProvider, ProviderError

MSGS = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


def handler_factory(responses, seen):
    it = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        r = next(it)
        return r if isinstance(r, httpx.Response) else httpx.Response(200, json=r)

    return httpx.MockTransport(handler)


OPENAI_OK = {"id": "c1", "model": "gpt-x", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop",
             "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -1.5}]}}],
             "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
ANTH_OK = {"id": "m1", "model": "test-model", "content": [{"type": "text", "text": "hel"}, {"type": "text", "text": "lo"}],
           "stop_reason": "max_tokens", "usage": {"input_tokens": 9, "output_tokens": 2}}


def test_openai_request_shape_and_parsing():
    seen = []
    p = OpenAIProvider(api_key="sk-test", transport=handler_factory([OPENAI_OK], seen), want_logprobs=True)
    c = p.complete("gpt-x", MSGS, max_tokens=50)
    req = seen[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert body["messages"] == MSGS and body["max_completion_tokens"] == 50 and body["logprobs"] is True
    assert (c.text, c.prompt_tokens, c.completion_tokens) == ("hello", 12, 3)
    assert c.mean_logprob == pytest.approx(-1.0)


def test_openai_missing_key_refuses_without_network():
    p = OpenAIProvider(api_key="", transport=handler_factory([], []))
    with pytest.raises(ProviderError, match="not set"):
        p.complete("m", MSGS)


def test_openai_retries_429_then_succeeds_and_does_not_leak_key():
    seen = []
    sleeps = []
    t = handler_factory([httpx.Response(429), httpx.Response(500), OPENAI_OK], seen)
    p = OpenAIProvider(api_key="sk-secret", transport=t, sleep=sleeps.append)
    assert p.complete("m", MSGS).text == "hello"
    assert len(seen) == 3 and sleeps == [0.5, 1.0]


def test_openai_400_is_not_retried_and_error_hides_key():
    seen = []
    p = OpenAIProvider(api_key="sk-secret", transport=handler_factory([httpx.Response(400)], seen))
    with pytest.raises(ProviderError) as e:
        p.complete("m", MSGS)
    assert len(seen) == 1 and e.value.status == 400 and "sk-secret" not in str(e.value)


def test_openai_bad_shape():
    p = OpenAIProvider(api_key="k", transport=handler_factory([{"nope": 1}], []))
    with pytest.raises(ProviderError, match="shape"):
        p.complete("m", MSGS)


def test_anthropic_request_shape_and_parsing():
    seen = []
    p = AnthropicProvider(api_key="ak-test", transport=handler_factory([ANTH_OK], seen))
    c = p.complete("test-model", MSGS + [{"role": "user", "content": "again"}], max_tokens=20)
    req, body = seen[0], json.loads(seen[0].content)
    assert str(req.url) == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == "ak-test" and req.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "Be brief." and body["max_tokens"] == 20
    assert body["messages"] == [{"role": "user", "content": "Hi\n\nagain"}]  # system moved, turns merged
    assert (c.text, c.prompt_tokens, c.completion_tokens, c.finish_reason) == ("hello", 9, 2, "length")


def test_anthropic_529_retry_and_timeout():
    seen = []
    p = AnthropicProvider(api_key="k", transport=handler_factory([httpx.Response(529), ANTH_OK], seen), sleep=lambda s: None)
    assert p.complete("m", MSGS).text == "hello"

    def boom(request):
        raise httpx.ReadTimeout("slow", request=request)

    p2 = AnthropicProvider(api_key="k", transport=httpx.MockTransport(boom), sleep=lambda s: None)
    with pytest.raises(ProviderError, match="timed out"):
        p2.complete("m", MSGS)


def test_pricing_arithmetic_and_unknown(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("price_date: '2026-01-01'\nmodels:\n  a: {provider: x, input: 2.0, output: 10.0}\n")
    t = PriceTable.load(f)
    assert t.cost("a", 1_000_000, 500_000) == pytest.approx(7.0)
    assert t.price_date == "2026-01-01"
    with pytest.raises(UnknownModelError):
        t.cost("b", 1, 1)


def test_repo_pricing_file_is_valid_and_dated():
    t = PriceTable.load("config/pricing.yaml")
    assert t.price_date == "2026-11-12" and "change" in t.note.lower()
    assert t.price("gpt-6-luna").output < t.price("gpt-6.1-sol").output
    assert not any(m.startswith("<") for m in t.models)  # the hosted template stays commented out


def test_hosted_tier_without_model_id_is_a_clear_config_error(pricing_path="config/pricing.yaml"):
    from modelrouter.gateway.app import gateway_from_config
    from modelrouter.router import DifficultyRouter

    class Dummy(DifficultyRouter):
        def __init__(self):
            pass

        def score(self, prompt):
            return 0.1

    cfg = {"tiers": {"small": {"provider": "openai"}, "big": {"provider": "anthropic", "model": "m"}}}
    with pytest.raises(ValueError, match="no model id is set"):
        gateway_from_config(cfg, PriceTable.load(pricing_path), Dummy())
