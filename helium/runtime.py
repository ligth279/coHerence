"""GPU lookup. diagnose() and ModalLLMClient do not branch on cold vs warm.

Flip helium.constants.HELIUM_RUNTIME (or env HELIUM_RUNTIME):
  deployed  — Cls.from_name; replica stays up (min_containers=1). Default.
  ephemeral — app.run() per call; container boots each time.

Never import helium.modal_app while a live `modal run` is already up.
That import is a second module/App and makes Modal mount PythonPackage:helium
on the GPU, which then pulls hydrogen. Hydrogen stays on CPU.
"""

from __future__ import annotations

import os
import sys
from typing import Any

from helium.constants import HELIUM_CLS_NAME, MODAL_APP_NAME

_VALID_RUNTIMES = frozenset({"ephemeral", "deployed"})


def current_runtime() -> str:
    from helium.constants import HELIUM_RUNTIME

    value = os.environ.get("HELIUM_RUNTIME", HELIUM_RUNTIME).strip().lower()
    if value not in _VALID_RUNTIMES:
        raise ValueError(
            f"Unknown HELIUM_RUNTIME={value!r}; use 'ephemeral' or 'deployed'"
        )
    return value


def complete_on_gpu(system: str, user: str) -> str:
    return invoke_gpu("complete", system, user)


def invoke_gpu(method: str, *args: Any) -> Any:
    """Call a HeliumGPU method. Nitrogen uses this too (same replica)."""
    if current_runtime() == "deployed":
        return _deployed(method, *args)
    return _ephemeral(method, *args)


def _invoke_timeout() -> float:
    from helium.constants import HELIUM_INVOKE_TIMEOUT_S

    raw = os.environ.get("HELIUM_INVOKE_TIMEOUT", str(HELIUM_INVOKE_TIMEOUT_S))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return float(HELIUM_INVOKE_TIMEOUT_S)


def _call(handle, *args: Any) -> Any:
    """Wait at most HELIUM_INVOKE_TIMEOUT_S so a rebooting replica cannot freeze Audit."""
    timeout = _invoke_timeout()
    spawn = getattr(handle, "spawn", None)
    if spawn is None:
        return handle.remote(*args)
    return spawn(*args).get(timeout=timeout)


def _deployed(method: str, *args: Any) -> Any:
    import modal

    cls = modal.Cls.from_name(MODAL_APP_NAME, HELIUM_CLS_NAME)
    return _call(getattr(cls(), method), *args)


def _live_gpu_cls() -> Any | None:
    """HeliumGPU from the already-running `modal run` app, if any.

    `modal run helium/modal_app.py` loads the file as one module. Importing
    `helium.modal_app` again creates a second App. Prefer the live one.
    """
    for mod in list(sys.modules.values()):
        gpu = getattr(mod, "HeliumGPU", None)
        app = getattr(mod, "app", None)
        if gpu is None or app is None:
            continue
        if getattr(app, "name", None) != MODAL_APP_NAME:
            continue
        if getattr(app, "app_id", None):
            return gpu
    return None


def _ephemeral(method: str, *args: Any) -> Any:
    """Cold-start path. Reuses the running app during `modal run`."""
    gpu = _live_gpu_cls()
    if gpu is not None:
        return _call(getattr(gpu(), method), *args)

    import modal

    from helium.modal_app import HeliumGPU, app

    def _run() -> Any:
        return _call(getattr(HeliumGPU(), method), *args)

    if app.app_id:
        return _run()
    with modal.enable_output():
        with app.run():
            return _run()
