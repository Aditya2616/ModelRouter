import io
import pickle

import pandas as pd
import pytest

from modelrouter.safepickle import UnsafePickleError, safe_load


def test_loads_plain_dataframe():
    df = pd.DataFrame({"prompt": pd.Series(["a", "b"], dtype=object), "score": [0.0, 1.0]})
    df.columns = pd.Index(["prompt", "score"], dtype=object)
    out = safe_load(io.BytesIO(pickle.dumps(df)))
    assert out.equals(df)


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def test_blocks_os_system_payload():
    with pytest.raises(UnsafePickleError):
        safe_load(io.BytesIO(pickle.dumps(_Evil())))


class _EvilBuiltin:
    def __reduce__(self):
        return (eval, ("1+1",))


def test_blocks_builtin_eval():
    with pytest.raises(UnsafePickleError):
        safe_load(io.BytesIO(pickle.dumps(_EvilBuiltin())))
