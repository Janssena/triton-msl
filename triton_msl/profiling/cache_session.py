"""Owned temporary directories shared by test and benchmark entry points.

Pure standard library: safe before Torch/Triton collection or child-process imports.
Caller paths are placement parents, never directories to empty or reuse. Nothing
is deleted, including on partial allocation failure; callers retain the evidence.
"""

import os
from pathlib import Path
import tempfile


CACHE_KEYS = (
    "TRITON_CACHE_DIR",
    "TRITON_MSL_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "CLANG_MODULE_CACHE_PATH",
)


def private_directory(parent=None, *, prefix="triton-msl-session-"):
    """Allocate a new empty child; propagate invalid-path and permission errors."""
    options = {"prefix": prefix}
    if parent:
        parent = os.path.abspath(os.path.expanduser(os.fspath(parent)))
        os.makedirs(parent, exist_ok=True)
        options["dir"] = parent
    return Path(tempfile.mkdtemp(**options))


def fresh_cache_environment(environment=None, *, keys=CACHE_KEYS):
    """Return a copied environment with new private cache children for each key.

    Never mutate the input or os.environ; process/session owners apply the result.
    A missing or empty setting means the system temporary placement directory.
    """
    result = dict(os.environ if environment is None else environment)
    for key in keys:
        result[key] = str(private_directory(result.get(key), prefix=f"triton-msl-{key.lower()}-"))
    return result
