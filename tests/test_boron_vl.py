"""VL navigation, driven entirely offline.

Nitrogen is never imported here. The client is injected, so Dev 1's tests need
no GPU and no dependency on Dev 3's folder.
"""

import json
import time
from pathlib import Path

import pytest

from boron import get_profile
from boron.navigator import _point, build_prompt, parse_action

pytest.importorskip("playwright", reason="run: playwright install chromium")

from boron import run_session, run_suite  # noqa: E402
from boron.manual import from_screenshots  # noqa: E402
from boron.runner import PlanFailed  # noqa: E402

FIXTURE_URL = (Path(__file__).parent / "fixtures" / "test_page.html").resolve().as_uri()
SUCCESS = "#order-confirmed"
GOAL = "Place the order"

# Actions carry 0-1000 grid coordinates, the space the prompt specifies and
# Qwen3-VL answers in. Real centres in the 1280x800 fixture viewport:
#   div#fake-button  "Place order"  (83, 346) px -> (65, 433) on the grid
#   button#submit-order "Pay"       (36, 385) px -> (28, 481)
# Both are stable across the gating click -- enabling #submit-order does not reflow.
FAKE_BUTTON = {"action": "click", "x": 65, "y": 433, "target": "Place order"}
SUBMIT = {"action": "click", "x": 28, "y": 481, "target": "Pay"}
# An h1: present in elements.json, but pressing it mutates nothing.
DEAD_SPOT = {"action": "click", "x": 500, "y": 60, "target": "the heading"}


class ScriptedVLClient:
    """Canned action sequence. `latency_s` fakes 30B inference time."""

    def __init__(self, actions, latency_s: float = 0.0):
        self.replies = [a if isinstance(a, str) else json.dumps(a) for a in actions]
        self.latency_s = latency_s
        self.calls = []

    def complete(self, system, user, image_b64=None):
        self.calls.append({"system": system, "user": user, "image_b64": image_b64})
        if self.latency_s:
            time.sleep(self.latency_s)
        if not self.replies:
            return json.dumps({"action": "give_up", "reason": "out of script"})
        return self.replies.pop(0)


def _run(tmp_path, profile_id, actions, latency_s=0.0, session_id=None, max_steps=12):
    client = ScriptedVLClient(actions, latency_s=latency_s)
    record = run_session(
        url=FIXTURE_URL,
        profile_id=profile_id,
        session_id=session_id or f"vl_{profile_id}",
        success_selector=SUCCESS,
        goal=GOAL,
        vl_client=client,
        max_steps=max_steps,
        out_root=str(tmp_path),
    )
    return record, client


def test_goal_only_completes_when_a_click_opens_a_new_path(tmp_path):
    """DEFAULT_GOAL is 'open it'. A link that changes the document path has."""
    dest = tmp_path / "article.html"
    dest.write_text(
        "<!doctype html><html><body><h1>Article</h1></body></html>",
        encoding="utf-8",
    )
    src = tmp_path / "home.html"
    src.write_text(
        """<!doctype html><html><body style="margin:0">
<a id="go" href="article.html" style="display:block;padding:40px 80px">Open article</a>
</body></html>""",
        encoding="utf-8",
    )
    client = ScriptedVLClient(
        [{"action": "click", "x": 90, "y": 70, "target": "Open article"}]
    )
    record = run_session(
        url=src.resolve().as_uri(),
        profile_id="baseline_default",
        session_id="open_path",
        success_selector="",
        goal=GOAL,
        vl_client=client,
        max_steps=3,
        out_root=str(tmp_path),
    )
    assert record.telemetry.task_completed is True
    assert len(client.calls) == 1


def test_goal_without_selector_accepts_model_done(tmp_path):
    client = ScriptedVLClient([{"action": "done", "reason": "the page is readable"}])
    record = run_session(
        url=FIXTURE_URL,
        profile_id="baseline_default",
        session_id="goal_only",
        success_selector="",
        goal=GOAL,
        vl_client=client,
        max_steps=3,
        out_root=str(tmp_path),
    )
    assert record.telemetry.task_completed is True
    assert len(client.calls) == 1


# --- mode selection -------------------------------------------------------


def test_steps_and_goal_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError):
        run_session(
            url=FIXTURE_URL, profile_id="baseline_default", session_id="x",
            success_selector=SUCCESS, steps=["#a"], goal=GOAL,
            vl_client=ScriptedVLClient([]), out_root=str(tmp_path),
        )
    with pytest.raises(ValueError):
        run_session(
            url=FIXTURE_URL, profile_id="baseline_default", session_id="x",
            success_selector=SUCCESS, out_root=str(tmp_path),
        )


def test_goal_without_a_client_is_refused(tmp_path):
    with pytest.raises(ValueError):
        run_session(
            url=FIXTURE_URL, profile_id="baseline_default", session_id="x",
            success_selector=SUCCESS, goal=GOAL, out_root=str(tmp_path),
        )


# --- the loop -------------------------------------------------------------


def test_vision_actions_complete_the_gated_task(tmp_path):
    record, client = _run(tmp_path, "baseline_default", [FAKE_BUTTON, SUBMIT])
    assert record.telemetry.task_completed is True
    assert record.telemetry.total_clicks >= 2
    assert len(client.calls) == 2
    assert all(c["image_b64"] for c in client.calls), "sighted profile saw no image"


def test_capture_policy_marks_the_run_as_vl_driven(tmp_path):
    record, _ = _run(tmp_path, "baseline_default", [FAKE_BUTTON, SUBMIT])
    assert record.capture_policy == "boron-vl-v1"


def test_all_four_artifacts_still_land(tmp_path):
    record, _ = _run(tmp_path, "baseline_default", [FAKE_BUTTON, SUBMIT])
    for path in record.artifacts.model_dump().values():
        assert Path(path).stat().st_size > 0


def test_nav_trace_is_a_sidecar_not_a_contract1_artifact(tmp_path):
    record, _ = _run(tmp_path, "baseline_default", [FAKE_BUTTON, SUBMIT])
    trace_path = Path(tmp_path) / record.session_id / "nav_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["mode"] == "live"
    assert trace["activated_selectors"]
    # Contract 1 declares exactly four artifacts; the trace is not one of them.
    assert str(trace_path) not in record.artifacts.model_dump().values()
    assert set(record.artifacts.model_dump()) == {
        "html_path", "screenshot_path", "a11y_tree_path", "elements_path",
    }


def test_a_garbled_reply_is_a_failed_step_not_a_crash(tmp_path):
    record, _ = _run(
        tmp_path, "baseline_default", ["not json at all", FAKE_BUTTON, SUBMIT]
    )
    assert record.telemetry.error_count >= 1
    assert record.telemetry.task_completed is True


def test_give_up_ends_the_run(tmp_path):
    record, client = _run(
        tmp_path, "baseline_default",
        [{"action": "give_up", "reason": "no path"}, FAKE_BUTTON],
    )
    assert record.telemetry.task_completed is False
    assert len(client.calls) == 1


def test_done_is_verified_against_the_page(tmp_path):
    """A model claiming success does not make the task complete."""
    record, _ = _run(
        tmp_path, "baseline_default", [{"action": "done", "reason": "I ordered it"}]
    )
    assert record.telemetry.task_completed is False


def test_an_unverified_done_does_not_end_the_run(tmp_path):
    """Observed live: Qwen3-VL clicked the gate then claimed the order was
    placed. Stopping on that claim throws away the remaining budget on a task
    the model only thinks it finished."""
    record, client = _run(
        tmp_path,
        "baseline_default",
        [FAKE_BUTTON, {"action": "done", "reason": "placed it"}, SUBMIT],
    )
    assert len(client.calls) == 3, "the run stopped at the unverified claim"
    assert record.telemetry.task_completed is True


def test_repeated_unverified_done_ends_the_run(tmp_path):
    """Observed live: given only "not visible yet", Qwen3-VL repeated done for
    the whole budget. One correction is worth giving; a second identical claim
    means it will not look again."""
    done = {"action": "done", "reason": "placed it"}
    record, client = _run(tmp_path, "baseline_default", [done] * 8, max_steps=8)
    assert len(client.calls) == 2, "the run argued with the model past the cap"
    assert record.telemetry.task_completed is False
    assert record.telemetry.error_count >= 1


def test_a_verified_done_still_ends_the_run(tmp_path):
    record, client = _run(
        tmp_path,
        "baseline_default",
        [FAKE_BUTTON, SUBMIT, {"action": "done", "reason": "placed it"}],
    )
    assert record.telemetry.task_completed is True
    # The loop sees the success selector before asking a third time.
    assert len(client.calls) == 2


def test_step_budget_exhaustion_is_recorded(tmp_path):
    scroll = {"action": "scroll", "dy": 10}
    record, client = _run(
        tmp_path, "baseline_default", [scroll] * 5, max_steps=3
    )
    assert len(client.calls) == 3
    assert record.telemetry.task_completed is False
    assert record.telemetry.error_count >= 1


# --- timing ---------------------------------------------------------------


def test_completion_time_excludes_model_inference(tmp_path):
    """Otherwise completion_time_ms measures how busy the B300 was."""
    record, client = _run(
        tmp_path, "baseline_default", [FAKE_BUTTON, SUBMIT], latency_s=0.4
    )
    assert record.telemetry.task_completed is True
    assert len(client.calls) == 2
    # 2 x 400ms of fake inference must not appear in the user's time.
    assert record.telemetry.completion_time_ms < 800


# --- constraints are what the model is given, not what it is told ---------


def test_the_prompt_never_names_the_profile(tmp_path):
    _, client = _run(tmp_path, "elderly", [FAKE_BUTTON, SUBMIT])
    blob = " ".join(c["system"] + c["user"] for c in client.calls).lower()
    for word in ("elderly", "impaired", "disabled", "tremor", "pretend", "persona"):
        assert word not in blob, f"prompt leaked a persona hint: {word}"


def test_keyboard_only_gets_no_click_verb():
    system, _ = build_prompt(
        get_profile("keyboard_only"), GOAL, "img", "", (1280, 800), []
    )
    assert '"click"' not in system
    assert '"press"' in system


def test_pointer_profile_gets_the_click_verb():
    system, _ = build_prompt(
        get_profile("baseline_default"), GOAL, "img", "", (1280, 800), []
    )
    assert '"click"' in system


def test_screen_reader_gets_the_a11y_tree_and_no_image(tmp_path):
    record, client = _run(
        tmp_path, "screen_reader_users",
        [{"action": "press", "key": "Tab"}, {"action": "press", "key": "Enter"}],
    )
    assert client.calls, "the loop never called the model"
    assert all(c["image_b64"] is None for c in client.calls), "a11y user was shown a screenshot"
    assert "cannot see the page" in client.calls[0]["user"]
    # The fixture's gate is an unnamed div, so it is absent from the a11y list.
    assert "fake-button" not in client.calls[0]["user"]
    assert record.telemetry.task_completed is False


def test_a_click_from_a_keyboard_only_profile_is_refused(tmp_path):
    record, _ = _run(tmp_path, "keyboard_only", [FAKE_BUTTON, SUBMIT])
    assert record.telemetry.task_completed is False
    assert record.telemetry.error_count >= 1


def test_screenshot_is_the_viewport_not_the_full_page(tmp_path):
    """The model answers in the space of the image it saw; mouse works in the viewport."""
    import base64
    import io

    pytest.importorskip("PIL")
    from PIL import Image

    _, client = _run(tmp_path, "baseline_default", [FAKE_BUTTON, SUBMIT])
    png = base64.b64decode(client.calls[0]["image_b64"])
    assert Image.open(io.BytesIO(png)).size == (1280, 800)
    assert "1280x800 pixels" in client.calls[0]["user"]


def test_image_dimensions_come_from_the_png_not_the_viewport():
    """They diverge at device_scale_factor != 1, and the model answers in the
    numbers it is told -- reporting the viewport there would halve every
    coordinate on a HiDPI capture."""
    import base64

    from boron.capture import _png_size, viewport_screenshot_b64
    from boron.runner import _driver

    # _driver, not sync_playwright: a bare sync_playwright() breaks once modal
    # has been imported by an earlier test in the same process.
    with _driver() as p:
        browser = p.chromium.launch()
        try:
            for dpr, expected in ((1, (1280, 800)), (2, (2560, 1600))):
                context = browser.new_context(
                    viewport={"width": 1280, "height": 800}, device_scale_factor=dpr
                )
                page = context.new_page()
                page.goto(FIXTURE_URL)
                b64, width, height = viewport_screenshot_b64(page)
                assert (width, height) == expected
                assert _png_size(base64.b64decode(b64)) == (width, height)
                context.close()
        finally:
            browser.close()


# --- the carbon join key --------------------------------------------------


def test_failed_selectors_from_a_coordinate_click_match_elements_json(tmp_path):
    """A VL click has no selector; elementFromPoint has to supply the join key."""
    record, _ = _run(
        tmp_path, "baseline_default", [DEAD_SPOT, FAKE_BUTTON, SUBMIT],
        session_id="vl_join",
    )
    elements = json.loads(
        Path(record.artifacts.elements_path).read_text(encoding="utf-8")
    )
    known = {e["element_selector"] for e in elements}
    assert record.telemetry.failed_selectors
    assert set(record.telemetry.failed_selectors) <= known


def test_tremor_still_costs_a_vl_driven_profile(tmp_path):
    record, _ = _run(tmp_path, "tremor_users", [FAKE_BUTTON, SUBMIT])
    assert record.telemetry.total_clicks >= 2
    assert record.telemetry.dead_clicks + record.telemetry.missed_clicks > 0


class _ModalImportingVLClient(ScriptedVLClient):
    """Imports modal on first use, exactly as nitrogen.ModalVLClient does."""

    def complete(self, system, user, image_b64=None):
        import modal  # noqa: F401

        return super().complete(system, user, image_b64)


def test_a_suite_survives_the_modal_import(tmp_path):
    """`import modal` installs WindowsSelectorEventLoopPolicy process-wide, and
    on Windows that loop cannot spawn Playwright's driver subprocess. Without
    pinning the policy the first profile succeeds and every later one dies in
    new_event_loop() with a bare NotImplementedError."""
    pytest.importorskip("modal", reason="modal not installed")
    client = _ModalImportingVLClient([FAKE_BUTTON, SUBMIT, FAKE_BUTTON, SUBMIT])
    records = run_suite(
        url=FIXTURE_URL,
        profile_ids=["baseline_default", "motor_impaired"],
        session_id_prefix="modal",
        success_selector=SUCCESS,
        goal=GOAL,
        vl_client=client,
        out_root=str(tmp_path),
    )
    assert len(records) == 2, "the second profile never launched a browser"
    assert all(r.telemetry.total_clicks > 0 for r in records)


# --- plan_once ------------------------------------------------------------


def test_plan_once_runs_one_vl_loop_and_replays_it(tmp_path):
    client = ScriptedVLClient([FAKE_BUTTON, SUBMIT])
    profiles = ["baseline_default", "motor_impaired", "keyboard_only"]
    records = run_suite(
        url=FIXTURE_URL,
        profile_ids=profiles,
        session_id_prefix="plan",
        success_selector=SUCCESS,
        goal=GOAL,
        vl_client=client,
        plan_once=True,
        out_root=str(tmp_path),
    )
    assert [r.profile_id for r in records] == profiles
    # Two calls total: the baseline's loop. The other profiles replay it.
    assert len(client.calls) == 2
    assert all(r.capture_policy == "boron-vl-v1" for r in records)

    trace = json.loads(
        (Path(tmp_path) / "plan_motor_impaired" / "nav_trace.json").read_text()
    )
    assert trace["mode"] == "replayed"
    assert trace["activated_selectors"] == ["div#fake-button", "button#submit-order"]


def test_plan_once_reproduces_the_scripted_disparity_shape(tmp_path):
    client = ScriptedVLClient([FAKE_BUTTON, SUBMIT])
    records = run_suite(
        url=FIXTURE_URL,
        profile_ids=["baseline_default", "keyboard_only"],
        session_id_prefix="shape",
        success_selector=SUCCESS,
        goal=GOAL,
        vl_client=client,
        plan_once=True,
        out_root=str(tmp_path),
    )
    by_id = {r.profile_id: r for r in records}
    assert by_id["baseline_default"].telemetry.task_completed is True
    assert by_id["keyboard_only"].telemetry.task_completed is False


class _DudVLClient:
    """The model never finds a control -- what bad grounding looks like."""

    def complete(self, system, user, image_b64=None):
        return json.dumps({"action": "give_up", "reason": "cannot locate"})


class _PartialVLClient:
    """Opens the gate, then claims success without reaching it."""

    def __init__(self):
        self.n = 0

    def complete(self, system, user, image_b64=None):
        self.n += 1
        if self.n == 1:
            return json.dumps(FAKE_BUTTON)
        return json.dumps({"action": "done", "reason": "I think it worked"})


def test_plan_once_refuses_to_replay_an_empty_path(tmp_path):
    """Otherwise every profile runs zero steps and reports zero errors -- a
    suite where nobody completes, which reads downstream as 'this site fails
    everyone' when the model simply could not drive it."""
    with pytest.raises(PlanFailed, match="activated nothing") as caught:
        run_suite(
            url=FIXTURE_URL, profile_ids=["baseline_default", "motor_impaired"],
            session_id_prefix="dud", success_selector=SUCCESS, goal=GOAL,
            vl_client=_DudVLClient(), plan_once=True, out_root=str(tmp_path),
        )
    assert [r.profile_id for r in caught.value.records] == ["baseline_default"]


def test_plan_once_refuses_to_replay_an_unproven_path(tmp_path):
    """A planning run that never reached the goal has not proven its path."""
    with pytest.raises(PlanFailed, match="unproven") as caught:
        run_suite(
            url=FIXTURE_URL, profile_ids=["baseline_default", "motor_impaired"],
            session_id_prefix="partial", success_selector=SUCCESS, goal=GOAL,
            vl_client=_PartialVLClient(), plan_once=True, out_root=str(tmp_path),
        )
    assert [r.profile_id for r in caught.value.records] == ["baseline_default"]


def test_dead_heading_does_not_end_the_run(tmp_path):
    record, client = _run(
        tmp_path,
        "baseline_default",
        [DEAD_SPOT, DEAD_SPOT, FAKE_BUTTON, SUBMIT],
        max_steps=12,
    )
    assert record.telemetry.task_completed is True
    assert len(client.calls) == 4
    trace = json.loads(
        (Path(tmp_path) / "vl_baseline_default" / "nav_trace.json").read_text()
    )
    assert any("fake-button" in (step.get("aimed_selector") or "") for step in trace["trace"])


def test_failed_click_is_not_echoed_as_copyable_json(tmp_path):
    """Observed live: history that contained the failed action JSON made the
    model paste x=[288,575] for the rest of the budget."""
    _, client = _run(
        tmp_path, "baseline_default", [DEAD_SPOT, FAKE_BUTTON, SUBMIT]
    )
    follow_up = client.calls[1]["user"]
    assert '"x": 500' not in follow_up
    assert "NO change" in follow_up


def test_clicking_text_inside_a_link_activates_the_link(tmp_path):
    """Wikipedia's language box is `a > strong`. Refusing the inner tag as
    static text throws away the click that leaves the portal."""
    html = tmp_path / "inner_link.html"
    html.write_text(
        """<!doctype html><html><body style="margin:0">
<a id="go" href="#opened" style="display:block;padding:40px 80px">
  <strong id="label">Open article</strong>
</a>
<p id="opened" style="display:none">yes</p>
<script>
  document.getElementById('go').addEventListener('click', (e) => {
    e.preventDefault();
    document.getElementById('opened').style.display = 'block';
  });
</script>
</body></html>""",
        encoding="utf-8",
    )
    # 1280x800 viewport. The <strong> sits near (80+?, 40+?). A point inside
    # the padded <a> hits the <strong>, not the <a> itself.
    inner = {"action": "click", "x": 90, "y": 70, "target": "Open article"}
    client = ScriptedVLClient([inner])
    record = run_session(
        url=html.resolve().as_uri(),
        profile_id="baseline_default",
        session_id="inner_link",
        success_selector="#opened",
        goal=GOAL,
        vl_client=client,
        max_steps=3,
        out_root=str(tmp_path),
    )
    assert record.telemetry.task_completed is True


def test_plan_once_needs_a_goal(tmp_path):
    with pytest.raises(ValueError):
        run_suite(
            url=FIXTURE_URL, profile_ids=["baseline_default"],
            session_id_prefix="x", success_selector=SUCCESS,
            steps=["#fake-button"], plan_once=True, out_root=str(tmp_path),
        )


# --- manual capture -------------------------------------------------------


def test_from_screenshots_yields_contract1_records(tmp_path):
    src = tmp_path / "shot.png"
    src.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # not a real PNG; nothing decodes it here
    )
    records = from_screenshots(
        [str(src)], url="https://example.com", session_id="man",
        out_root=str(tmp_path / "out"),
    )
    assert len(records) == 1
    record = records[0]
    assert record.capture_policy == "boron-manual-png-v1"
    assert record.session_id == "man_1"
    # elements.json exists but is empty: carbon's geometry rules find nothing,
    # which is the honest result for an image with no DOM behind it.
    assert json.loads(Path(record.artifacts.elements_path).read_text()) == []
    assert Path(record.artifacts.screenshot_path).stat().st_size > 0


def test_from_screenshots_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        from_screenshots(
            ["nope.png"], url="https://example.com", session_id="m",
            out_root=str(tmp_path),
        )


@pytest.mark.skip(reason="needs a human at the keyboard; see boron/sub-arch.md")
def test_run_manual_captures_a_state_per_hotkey():
    """Manual repro:

        python -c "import boron; boron.run_manual('https://example.com','sess_manual')"

    Chromium opens. F8 on each page, Escape to finish. Expect one directory per
    F8 under data/sessions/sess_manual_{n}/ with all four artifacts.
    """


@pytest.mark.parametrize(
    "action,expected",
    [
        ({"x": 64, "y": 431}, (64.0, 431.0)),
        # Observed from a live Qwen3-VL run: the pair packed into x.
        ({"x": [64, 431], "y": 431}, (64.0, 431.0)),
        ({"x": "64", "y": "431"}, (64.0, 431.0)),
        ({"point": [12, 34]}, (12.0, 34.0)),
        ({"point_2d": [[12, 34]]}, (12.0, 34.0)),
        ({"coordinate": [5, 6]}, (5.0, 6.0)),
        # A box means aim at its centre.
        ({"bbox_2d": [10, 20, 30, 40]}, (20.0, 30.0)),
        ({"x": [10, 20, 30, 40]}, (20.0, 30.0)),
    ],
)
def test_point_accepts_the_shapes_a_grounding_model_emits(action, expected):
    """Qwen-VL grounding output is not one fixed shape. Refusing anything but
    two scalars turns a correctly located target into a crashed step."""
    assert _point(action) == expected


def test_point_rejects_an_action_with_no_coordinates():
    with pytest.raises(RuntimeError, match="no usable coordinates"):
        _point({"action": "click", "target": "the button"})


def test_a_packed_coordinate_pair_still_clicks(tmp_path):
    """End to end: the live failure shape must now drive the page."""
    packed_gate = {"action": "click", "x": [65, 433], "y": 433, "target": "gate"}
    packed_submit = {"action": "click", "x": [28, 481], "y": 481, "target": "order"}
    record, _ = _run(tmp_path, "baseline_default", [packed_gate, packed_submit])
    assert record.telemetry.task_completed is True


# --- parsing --------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"action": "done"}', "done"),
        ('```json\n{"action": "click", "x": 1, "y": 2}\n```', "click"),
        ('Sure, I will click. {"action": "click", "x": 1, "y": 2}', "click"),
        ('<think>hmm</think>{"action": "press", "key": "Tab"}', "press"),
        (
            '{"action": "click", "x": [241, 360], "target": "featured article"}\n'
            '{"action": "done", "reason": "The main article about President',
            "click",
        ),
    ],
)
def test_parse_action_accepts_the_shapes_a_model_actually_emits(raw, expected):
    assert parse_action(raw)["action"] == expected


def test_parse_action_keeps_the_click_when_done_is_concatenated():
    """Live Wikipedia: two objects in one reply. The first was the article link."""
    parsed = parse_action(
        '{"action": "click", "x": [241, 360], "target": "the featured article"}\n'
        '{"action": "done", "reason": "The main article about President William McKinley has '
    )
    assert parsed is not None
    assert parsed["action"] == "click"
    assert parsed["x"] == [241, 360]


@pytest.mark.parametrize("raw", ["", "no json", "{broken", '{"reason": "no action"}', "[1,2]"])
def test_parse_action_rejects_junk(raw):
    assert parse_action(raw) is None
