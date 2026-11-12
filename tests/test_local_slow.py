import pytest

from modelrouter.live import PREFIX, SMALL, parse_letter
from modelrouter.providers import LocalProvider


@pytest.mark.slow
def test_local_model_answers_with_forced_prefix():
    prompt = "Answer with the letter only.\n\nWhat is 2 + 2?\nA. 3\nB. 4\nC. 5\nAnswer:"
    c = LocalProvider(threads=2).complete(SMALL, [{"role": "user", "content": prompt}], max_tokens=4, prefix=PREFIX)
    assert parse_letter(c.text) in {"A", "B", "C"}  # format check only
    assert c.completion_tokens <= 4 and c.mean_logprob is not None
