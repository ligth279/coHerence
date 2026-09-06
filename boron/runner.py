"""Playwright execution harness. Sync API, one profile per session.

Two ways to drive a task:

  steps=[...]  scripted CSS selectors -- deterministic, stamped boron-v1
  goal="..."   vision-driven via a VLClient -- stamped boron-vl-v1

The constraint layer is identical either way. Only the source of the next
target changes.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from playwright.sync_api import sync_playwright

from boron.capture import (
    CANONICAL_SELECTOR_SCRIPT,
    capture_artifacts,
    element_at_point,
    write_preview,
)
from boron.models import (
    CAPTURE_POLICY,
    CAPTURE_POLICY_VL,
    RawSessionArtifacts,
    SessionArtifacts,
    Telemetry,
)
from boron.navigator import MAX_STEPS, navigate
from boron.profiles import get_profile

DATA_ROOT = "data/sessions"

DEFAULT_SEED = 1729
# Wikipedia and other ad-heavy pages never fire `load`. Capture must not hang.
GOTO_TIMEOUT_MS = 20_000
DEFAULT_TIMEOUT_MS = 20_000

NAV_TRACE_FILENAME = "nav_trace.json"

# Counts DOM changes so a click that achieved nothing can be called dead.
_OBSERVER_SCRIPT = """
window.__boron = { mutations: 0 };
new MutationObserver((records) => { window.__boron.mutations += records.length; })
  .observe(document, { subtree: true, childList: true, attributes: true });
"""


@contextmanager
def _driver():
    """Start Playwright under an event loop that can actually spawn its driver.

    Playwright's sync API runs the driver as a subprocess, and on Windows only a
    Proactor loop can start one. `import modal` -- which `nitrogen.ModalVLClient`
    triggers on its first call -- installs `WindowsSelectorEventLoopPolicy`
    process-wide, so in a suite the first profile would succeed and every profile
    after it would die in `new_event_loop()` with a bare `NotImplementedError`.

    The policy only matters while the loop is being constructed: Playwright calls
    `asyncio.new_event_loop()` inside `__enter__` and never consults the policy
    again. So pin it across construction and hand modal back the policy it chose.
    """
    proactor = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
    previous = asyncio.get_event_loop_policy()
    swap = (
        sys.platform == "win32"
        and proactor is not None
        and not isinstance(previous, proactor)
    )
    if swap:
        asyncio.set_event_loop_policy(proactor())

    manager = sync_playwright()
    try:
        playwright = manager.__enter__()
    finally:
        if swap:
            asyncio.set_event_loop_policy(previous)
    try:
        yield playwright
    finally:
        manager.__exit__(None, None, None)


def run_session(
    url: str,
    profile_id: str,
    session_id: str,
    success_selector: str,
    steps: list[str] | None = None,
    goal: str | None = None,
    vl_client=None,
    max_steps: int = MAX_STEPS,
    out_root: str = DATA_ROOT,
    seed: int = DEFAULT_SEED,
    on_progress=None,
) -> RawSessionArtifacts:
    """Drive one profile through one task. Writes artifacts, returns Contract 1."""
    if (steps is None) == (goal is None):
        raise ValueError("pass exactly one of steps=[...] or goal='...'")
    if goal is not None and vl_client is None:
        raise ValueError("goal=... needs a vl_client")

    profile = get_profile(profile_id)
    out_dir = Path(out_root) / session_id

    with _driver() as p:
        browser = p.chromium.launch(
            args=[
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--use-gl=swiftshader",
            ]
        )
        context = browser.new_context(
            viewport={
                "width": profile.viewport_width,
                "height": profile.viewport_height,
            },
            has_touch=profile.has_touch,
            color_scheme="light",
        )
        page = context.new_page()
        page.set_default_timeout(DEFAULT_TIMEOUT_MS)
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))

        page.add_init_script(_OBSERVER_SCRIPT)
        page.goto(url, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)
        page.wait_for_timeout(400)
        if profile.zoom != 1.0:
            page.evaluate("(z) => { document.body.style.zoom = z; }", profile.zoom)

        rng = random.Random(seed)
        started = time.perf_counter()

        def emit(event: dict) -> None:
            if on_progress is None:
                return
            shot = write_preview(page, out_dir)
            on_progress(
                {
                    "profile_id": profile.id,
                    "session_id": session_id,
                    "screenshot": shot,
                    **event,
                }
            )

        emit({"stage": "page_ready"})
        if steps is not None:
            counters = _run_steps(page, profile, steps, errors, rng, emit=emit)
            completed = _is_visible(page, success_selector)
            vl_ms = 0
            nav = None
        else:
            counters = _new_counters()
            nav = navigate(
                page,
                profile,
                goal,
                success_selector,
                vl_client,
                press_point=_seeded_press_point(rng),
                is_visible=_is_visible,
                max_steps=max_steps,
                errors=errors,
                counters=counters,
                emit=emit,
            )
            completed = nav.completed
            vl_ms = nav.vl_ms
            if not completed and nav.steps_taken >= max_steps:
                errors.append(f"step budget of {max_steps} exhausted")
        # Model inference is not the user's think time. A 30B call is seconds
        # while read_delay_ms is hundreds of ms -- left in, completion_time_ms
        # would measure GPU queueing and the disparity signal would be gone.
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000) - vl_ms)

        artifacts = capture_artifacts(page, out_dir)
        context.close()
        browser.close()

    if nav is not None:
        _write_nav_trace(out_dir, nav, goal, mode="live", vl_ms=vl_ms)

    return RawSessionArtifacts(
        session_id=session_id,
        profile_id=profile.id,
        url=url,
        artifacts=SessionArtifacts(**artifacts),
        telemetry=Telemetry(
            completion_time_ms=elapsed_ms,
            task_completed=completed,
            total_clicks=counters["total_clicks"],
            dead_clicks=counters["dead_clicks"],
            missed_clicks=counters["missed_clicks"],
            keyboard_nav_steps=counters["keyboard_nav_steps"],
            error_count=len(errors),
            failed_selectors=counters["failed_selectors"],
        ),
        capture_policy=CAPTURE_POLICY if steps is not None else CAPTURE_POLICY_VL,
    )


def _new_counters() -> dict:
    return {
        "total_clicks": 0,
        "dead_clicks": 0,
        "missed_clicks": 0,
        "keyboard_nav_steps": 0,
        "failed_selectors": [],
    }


def _write_nav_trace(out_dir: Path, nav, goal, mode: str, vl_ms: int) -> None:
    """Sidecar, deliberately not declared in SessionArtifacts.

    Contract 1 has four artifact paths and adding a fifth needs Dev 2's ack.
    Carbon ignores files it does not know about, so the trace rides along for
    debugging and the frontend's execution tracker without a contract change.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / NAV_TRACE_FILENAME).write_text(
        json.dumps(
            {
                "goal": goal,
                "mode": mode,
                "completed": nav.completed,
                "steps_taken": nav.steps_taken,
                "vl_ms": vl_ms,
                "activated_selectors": nav.activated_selectors,
                "trace": nav.trace,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_steps(page, profile, steps, errors, rng, emit=None) -> dict[str, int]:
    counters = _new_counters()
    for selector in steps:
        try:
            if profile.read_delay_ms:
                page.wait_for_timeout(profile.read_delay_ms)
            _activate(page, profile, selector, counters, rng)
            if emit is not None:
                emit({"stage": "step", "selector": selector})
        except Exception as exc:  # a step that cannot be reached is a real failure
            errors.append(f"{selector}: {exc}")
            _record_failure(counters, selector, page)
            if emit is not None:
                emit({"stage": "step", "selector": selector, "failed": True})
            break
    return counters


def _record_failure(counters, selector: str, page=None) -> None:
    """Store the elements.json form of the selector so Dev 2's join matches."""
    if page is not None:
        try:
            selector = page.evaluate(CANONICAL_SELECTOR_SCRIPT, selector)
        except Exception:
            pass
    if selector not in counters["failed_selectors"]:
        counters["failed_selectors"].append(selector)


def _mutations(page):
    """The DOM-change counter, or None when the execution context is gone."""
    try:
        return page.evaluate("() => window.__boron.mutations")
    except Exception:
        return None


def _changed(page, before, before_url) -> bool:
    """True if the press mutated the DOM or navigated. Navigation drops the counter."""
    try:
        if page.url != before_url:
            return True
    except Exception:
        return True
    after = _mutations(page)
    return after is None or before is None or after != before


def _activate(page, profile, selector, counters, rng) -> None:
    """Reach the target and act on it. Retries a slip the way a person would."""
    if profile.keyboard_only:
        if profile.ax_tree_only and not _has_accessible_name(page, selector):
            raise RuntimeError("no accessible name exposed")
        before, before_url = _mutations(page), page.url
        counters["keyboard_nav_steps"] += _tab_to(page, selector)
        page.keyboard.press("Enter")
        page.wait_for_timeout(50)
        if not _changed(page, before, before_url):
            counters["dead_clicks"] += 1
        return

    for _ in range(profile.max_attempts):
        before, before_url = _mutations(page), page.url
        _press(page, profile, selector, rng, counters)
        counters["total_clicks"] += 1
        page.wait_for_timeout(50)
        if _changed(page, before, before_url):
            return
        counters["dead_clicks"] += 1
        _record_failure(counters, selector, page)
    raise RuntimeError(f"{profile.max_attempts} attempts produced no change")


def _press(page, profile, selector, rng, counters=None) -> None:
    """One pointer press at the centre of a selector's box."""
    box = page.locator(selector).bounding_box()
    if box is None:
        raise RuntimeError("no bounding box")
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    _press_point(page, profile, x, y, rng, counters, box=box)


def _press_point(page, profile, x, y, rng, counters=None, box=None, aimed=None) -> None:
    """One pointer press at a point, displaced by the profile tremor.

    A miss is judged against `box` when the caller had a target box (scripted
    path) and against the element under the aimed point otherwise (VL path,
    where a coordinate click has no box at all).
    """
    if profile.tremor_px:
        x += rng.gauss(0.0, profile.tremor_px)
        y += rng.gauss(0.0, profile.tremor_px)
        if counters is not None and _missed(page, x, y, box, aimed):
            counters["missed_clicks"] += 1
    page.mouse.move(x, y)
    page.mouse.down()
    if profile.dwell_ms:
        page.wait_for_timeout(profile.dwell_ms)
    page.mouse.up()


def _missed(page, x, y, box, aimed) -> bool:
    if box is not None:
        return not (
            box["x"] <= x <= box["x"] + box["width"]
            and box["y"] <= y <= box["y"] + box["height"]
        )
    return element_at_point(page, x, y) != aimed


def _seeded_press_point(rng):
    """Bind the run's RNG so the navigator needs no seed of its own.

    Wraps the click in the same attempt/dead-click policy the scripted path
    uses, so a VL click is measured identically to a scripted one.
    """

    def press(page, profile, x, y, counters, aimed):
        for _ in range(profile.max_attempts):
            before, before_url = _mutations(page), page.url
            _press_point(page, profile, x, y, rng, counters, aimed=aimed)
            if counters is not None:
                counters["total_clicks"] += 1
            page.wait_for_timeout(50)
            if _changed(page, before, before_url):
                return
            if counters is not None:
                counters["dead_clicks"] += 1
                if aimed:
                    _record_failure(counters, aimed)
        raise RuntimeError(f"{profile.max_attempts} attempts produced no change")

    return press


def _has_accessible_name(page, selector: str) -> bool:
    """An assistive-tech user can only reach what the a11y tree actually names."""
    return page.evaluate(
        """(sel) => {
            const el = document.querySelector(sel);
            if (!el) return false;
            const name = el.getAttribute('aria-label')
              || el.getAttribute('alt')
              || el.getAttribute('title')
              || (el.textContent || '').trim();
            return Boolean(name) && el.tabIndex >= 0;
        }""",
        selector,
    )


def _tab_to(page, selector, limit: int = 60) -> int:
    """Tab until the target holds focus. Raises if it is never reachable."""
    for pressed in range(1, limit + 1):
        page.keyboard.press("Tab")
        if page.evaluate(
            "(sel) => document.activeElement === document.querySelector(sel)", selector
        ):
            return pressed
    raise RuntimeError(f"not keyboard reachable within {limit} tab stops")


def _is_visible(page, selector: str) -> bool:
    if not selector:
        return False
    return page.locator(selector).is_visible()


def run_suite(
    url: str,
    profile_ids: list[str],
    session_id_prefix: str,
    success_selector: str,
    steps: list[str] | None = None,
    goal: str | None = None,
    vl_client=None,
    plan_once: bool = False,
    max_steps: int = MAX_STEPS,
    runs: int = 1,
    out_root: str = DATA_ROOT,
    seed: int = DEFAULT_SEED,
    on_progress=None,
) -> list[RawSessionArtifacts]:
    """Run every profile through the same task. One record per (profile, run).

    Profiles run sequentially on purpose: completion_time_ms drives the fairness
    score, and concurrent browsers contend for CPU and skew it.

    `plan_once` runs the VL loop on the first profile only and replays the
    selectors it discovered through the rest. One B300 is shared with Helium and
    holds a single generate lock, so a live loop across 11 profiles serialises
    ~90 30B calls behind it.
    """
    if plan_once and goal is None:
        raise ValueError("plan_once needs goal=...")

    records: list[RawSessionArtifacts] = []
    replay_steps: list[str] | None = None

    for position, profile_id in enumerate(profile_ids):
        for index in range(runs):
            suffix = "" if runs == 1 else f"_{index + 1}"
            session_id = f"{session_id_prefix}_{profile_id}{suffix}"
            use_goal, use_steps = goal, steps
            if plan_once and position > 0:
                use_goal, use_steps = None, replay_steps or []

            if on_progress is not None:
                on_progress(
                    {
                        "stage": "profile_start",
                        "profile_id": profile_id,
                        "session_id": session_id,
                    }
                )
            record = run_session(
                url=url,
                profile_id=profile_id,
                session_id=session_id,
                success_selector=success_selector,
                steps=use_steps,
                goal=use_goal,
                vl_client=vl_client,
                max_steps=max_steps,
                out_root=out_root,
                # Each run needs its own seed or repeats are byte-identical.
                seed=seed + index,
                on_progress=on_progress,
            )
            if plan_once and position > 0:
                # The path came from the model even though this run replayed it.
                record = record.model_copy(
                    update={"capture_policy": CAPTURE_POLICY_VL}
                )
                _write_replay_trace(Path(out_root) / session_id, goal, use_steps)
            records.append(record)
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "profile_done",
                        "profile_id": record.profile_id,
                        "session_id": record.session_id,
                        "screenshot": record.artifacts.screenshot_path,
                        "task_completed": record.telemetry.task_completed,
                        "error_count": record.telemetry.error_count,
                    }
                )
            if plan_once and position == 0 and replay_steps is None:
                try:
                    replay_steps = _plan_from(Path(out_root) / session_id, record)
                except PlanFailed as exc:
                    raise PlanFailed(str(exc), records=list(records)) from exc
    return records


class PlanFailed(RuntimeError):
    """The planning run produced no usable path, so there is nothing to replay."""

    def __init__(self, message: str, records=None):
        super().__init__(message)
        self.records = list(records or [])


def _plan_from(out_dir: Path, record: RawSessionArtifacts) -> list[str]:
    """Selectors the planner walked. Raises PlanFailed if the path is empty or unproven."""
    trace = json.loads((out_dir / NAV_TRACE_FILENAME).read_text(encoding="utf-8"))
    seen: list[str] = []
    for selector in trace.get("activated_selectors", []):
        if selector and selector not in seen:
            seen.append(selector)

    if not seen:
        raise PlanFailed(
            f"the planning run ({record.profile_id}) activated nothing, so there is "
            f"no path to replay. Either the goal is unreachable or the model could "
            f"not ground its coordinates -- check {out_dir / NAV_TRACE_FILENAME}."
        )
    if not record.telemetry.task_completed:
        raise PlanFailed(
            f"the planning run ({record.profile_id}) did not reach the success "
            f"selector, so its path is unproven; replaying it would fail every "
            f"profile identically. Walked {seen}. "
            f"Check {out_dir / NAV_TRACE_FILENAME}."
        )
    return seen


def _write_replay_trace(out_dir: Path, goal, steps) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / NAV_TRACE_FILENAME).write_text(
        json.dumps(
            {"goal": goal, "mode": "replayed", "activated_selectors": list(steps or [])},
            indent=2,
        ),
        encoding="utf-8",
    )
