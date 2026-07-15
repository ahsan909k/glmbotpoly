"""Cross-platform peak-RSS reporting — no third-party dependency (psutil is not pinned).

Used by the journal-reading stages to report the peak resident memory of the
streaming pass against the ``--max-rss-mb`` ceiling.
"""

from __future__ import annotations

import sys


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in **MB**, or ``None`` if it can't be
    determined on this platform.

    - Windows: ``GetProcessMemoryInfo().PeakWorkingSetSize`` via ctypes.
    - Linux/macOS: ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (KB on Linux, bytes
      on macOS).
    """
    if sys.platform == "win32":
        return _windows_peak_rss_mb()
    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024
    except Exception:  # pragma: no cover - platform-dependent
        return None


def report_peak_rss(stage: str, peak_mb: float | None, ceiling_mb: int | None) -> None:
    """Print the peak RSS for a stage and warn if it exceeded the advisory ceiling."""
    if peak_mb is None:
        print(f"[{stage}] peak RSS: unavailable on this platform")
        return
    msg = f"[{stage}] peak RSS: {peak_mb:,.0f} MB"
    if ceiling_mb:
        msg += f" (ceiling {ceiling_mb:,} MB)"
        if peak_mb > ceiling_mb:
            msg += " — OVER the ceiling; narrow --since/--until to bound memory"
    print(msg)


def _windows_peak_rss_mb() -> float | None:  # pragma: no cover - Windows-only
    try:
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        counters = _PMC()
        counters.cb = ctypes.sizeof(_PMC)
        # (HANDLE)-1 is the GetCurrentProcess pseudo-handle (avoids handle truncation).
        if fn(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        return None
    return None
