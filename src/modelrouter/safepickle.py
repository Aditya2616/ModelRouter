"""Allow-list unpickler.

RouterBench is distributed as a pickled pandas DataFrame. Unpickling untrusted data can run
arbitrary code, so this loader only reconstructs the handful of pandas / numpy classes a plain
DataFrame needs and refuses everything else.
"""

from __future__ import annotations

import pickle
from typing import Any, BinaryIO

_ALLOWED_PREFIXES = ("pandas.", "numpy.")
_ALLOWED_EXACT = {
    ("builtins", "slice"),
    ("builtins", "object"),
    ("builtins", "tuple"),
    ("builtins", "list"),
    ("builtins", "dict"),
    ("builtins", "set"),
    ("builtins", "frozenset"),
    ("collections", "OrderedDict"),
    ("_codecs", "encode"),
    ("pyarrow.lib", "_restore_array"),  # pandas 3 string columns; builds an array from buffers
}
# Names that sit inside the allowed packages but can read files, run code or import modules.
_DENIED_NAMES = {
    "eval", "exec", "system", "popen", "load", "loads", "read_pickle", "read_csv", "to_pickle",
    "fromfile", "memmap", "save", "savez", "open", "testing",
}


class UnsafePickleError(pickle.UnpicklingError):
    pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        full = f"{module}.{name}"
        if (module, name) in _ALLOWED_EXACT:
            return super().find_class(module, name)
        if full.startswith(_ALLOWED_PREFIXES) and name not in _DENIED_NAMES and ".testing" not in full:
            return super().find_class(module, name)
        raise UnsafePickleError(f"blocked global during unpickling: {full}")


def safe_load(fh: BinaryIO) -> Any:
    return _SafeUnpickler(fh).load()
