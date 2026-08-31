"""Compile `kernels.c` on demand and expose it through ctypes.

Why ctypes and not a build system? Because the topic is the kernel, not the
packaging. One `cc` invocation, cached by mtime, and a `.so` that lands in a
gitignored `build/` directory. If no compiler is present the module degrades to
`AVAILABLE = False` and the tests skip rather than fail — the same
"degrade, don't die" rule the AlphaDesk registry follows.

.NET analogy: this is `DllImport` with the P/Invoke signatures declared by
hand, plus a tiny MSBuild step you can read in one screen.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import sysconfig
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "kernels.c"
BUILD = HERE.parent.parent / "build"
LIBRARY = BUILD / "libmatmul.so"

#: Tried in order; the first that compiles the file wins.
FLAG_SETS: list[list[str]] = [
    ["-O3", "-march=native", "-fopenmp", "-funroll-loops"],
    ["-O3", "-march=native", "-funroll-loops"],
    ["-O3", "-funroll-loops"],
    ["-O2"],
]

_KERNEL_SIGNATURE = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_BLOCKED_EXTRA = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

PLAIN_KERNELS = ("matmul_naive_ijk", "matmul_ikj")
BLOCKED_KERNELS = ("matmul_blocked", "matmul_blocked_omp", "matmul_blocked_regtile")
ALL_KERNELS = PLAIN_KERNELS + BLOCKED_KERNELS


class CompilerUnavailable(RuntimeError):
    pass


def _compiler() -> str:
    for candidate in (os.environ.get("CC"), sysconfig.get_config_var("CC"), "cc", "gcc", "clang"):
        if not candidate:
            continue
        exe = candidate.split()[0]
        if shutil.which(exe):
            return exe
    raise CompilerUnavailable("no C compiler on PATH (set CC=... to choose one)")


def build(force: bool = False) -> tuple[Path, list[str]]:
    """Compile the shared library if it is missing or older than the source."""
    BUILD.mkdir(exist_ok=True)
    if not force and LIBRARY.exists() and LIBRARY.stat().st_mtime > SOURCE.stat().st_mtime:
        return LIBRARY, _read_flags()

    cc = _compiler()
    errors = []
    for flags in FLAG_SETS:
        cmd = [cc, *flags, "-shared", "-fPIC", str(SOURCE), "-o", str(LIBRARY)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            (BUILD / "flags.txt").write_text(" ".join(flags))
            return LIBRARY, flags
        errors.append(f"{' '.join(flags)}: {proc.stderr.strip().splitlines()[-1:]}")
    raise CompilerUnavailable("every flag set failed:\n" + "\n".join(errors))


def _read_flags() -> list[str]:
    f = BUILD / "flags.txt"
    return f.read_text().split() if f.exists() else []


class Kernels:
    """The loaded library, with one Python callable per C kernel."""

    def __init__(self, force_build: bool = False):
        path, self.flags = build(force=force_build)
        self._lib = ctypes.CDLL(str(path))
        for name in PLAIN_KERNELS:
            fn = getattr(self._lib, name)
            fn.argtypes = list(_KERNEL_SIGNATURE)
            fn.restype = None
        for name in BLOCKED_KERNELS:
            fn = getattr(self._lib, name)
            fn.argtypes = [*_KERNEL_SIGNATURE, *_BLOCKED_EXTRA]
            fn.restype = None
        self._lib.matmul_threads.restype = ctypes.c_int

    @property
    def threads(self) -> int:
        return int(self._lib.matmul_threads())

    @property
    def openmp(self) -> bool:
        return "-fopenmp" in self.flags

    def call(
        self,
        name: str,
        A: np.ndarray,
        B: np.ndarray,
        *,
        mc: int = 128,
        kc: int = 128,
        nc: int = 256,
    ) -> np.ndarray:
        if name not in ALL_KERNELS:
            raise KeyError(f"unknown kernel {name!r}; have {ALL_KERNELS}")
        A = np.ascontiguousarray(A, dtype=np.float64)
        B = np.ascontiguousarray(B, dtype=np.float64)
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"shape mismatch: {A.shape} @ {B.shape}")
        m, k = A.shape
        n = B.shape[1]
        C = np.empty((m, n), dtype=np.float64)
        fn = getattr(self._lib, name)
        if name in BLOCKED_KERNELS:
            fn(A, B, C, m, n, k, mc, kc, nc)
        else:
            fn(A, B, C, m, n, k)
        return C


_CACHED: Kernels | None = None
AVAILABLE = True
UNAVAILABLE_REASON = ""

try:  # probe once at import so tests can skip cleanly
    build()
except Exception as exc:  # noqa: BLE001 - degrade, don't die
    AVAILABLE = False
    UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def kernels() -> Kernels:
    """The process-wide loaded library."""
    global _CACHED
    if _CACHED is None:
        _CACHED = Kernels()
    return _CACHED


def environment() -> dict:
    info = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "compiler_available": AVAILABLE,
    }
    if AVAILABLE:
        k = kernels()
        info |= {"flags": " ".join(k.flags), "openmp": k.openmp, "threads": k.threads}
    else:
        info["reason"] = UNAVAILABLE_REASON
    return info
