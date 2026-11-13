import pytest

from modelrouter.live import PREFIX, SMALL, parse_letter
from modelrouter.providers import LocalProvider


@pytest.mark.slow
def test_local_model_answers_with_forced_prefix():
    prompt = "Answer with the letter only.\n\nWhat is 2 + 2?\nA. 3\nB. 4\nC. 5\nAnswer:"
    c = LocalProvider(threads=2).complete(SMALL, [{"role": "user", "content": prompt}], max_tokens=4, prefix=PREFIX)
    assert parse_letter(c.text) in {"A", "B", "C"}  # format check only
    assert c.completion_tokens <= 4 and c.mean_logprob is not None


@pytest.mark.slow
def test_batched_answers_match_single_requests_on_most_items():
    prompts = [f"Answer with the letter only.\n\nWhat is {a} + {b}?\nA. {a + b - 1}\nB. {a + b}\nC. {a + b + 1}\nAnswer:"
               for a, b in [(2, 2), (10, 5), (7, 8), (100, 23), (3, 3), (9, 9)]]
    p = LocalProvider(threads=2)
    batched = p.complete_batch(SMALL, prompts, max_tokens=4, prefix=PREFIX, batch_size=3)
    single = [p.complete(SMALL, [{"role": "user", "content": q}], max_tokens=4, prefix=PREFIX) for q in prompts]
    agree = sum(parse_letter(b.text) == parse_letter(s.text) for b, s in zip(batched, single, strict=True))
    assert agree >= len(prompts) - 1  # left padding can flip a near-tie in float16, not more
    assert [b.prompt_tokens for b in batched] == [s.prompt_tokens for s in single]
