"""Lithium HTTP + create_report. Mock LLM. No live GPU."""

import pytest
from fastapi.testclient import TestClient

from helium.client import MockLLMClient
from helium.example import example_bundle
from hydrogen.models import ScoreStatus
from lithium.app import app, get_llm_client
from lithium.jobs import clear as clear_jobs
from lithium.reports import create_report


@pytest.fixture
def client():
    clear_jobs()
    return TestClient(app)


def test_create_report_without_diagnose_locks_hydrogen_score():
    bundle = example_bundle()
    report = create_report(
        bundle.model_dump(mode="json"), "rep_1", diagnose=False
    )
    assert report.report_id == "rep_1"
    assert report.overall_fairness_score == 72
    assert report.analyst == "hydrogen"
    assert report.diagnosis == ""
    assert report.findings[0].diagnosis == ""


def test_create_report_diagnose_does_not_change_score():
    report = create_report(
        example_bundle(),
        "rep_2",
        diagnose=True,
        llm_client=MockLLMClient(),
    )
    assert report.analyst == "helium"
    assert report.overall_fairness_score == 72
    assert report.score_status is ScoreStatus.VALID
    assert report.scoring_policy == "hydrogen-v1"
    assert report.diagnosis
    assert report.findings[0].diagnosis == ""


def test_empty_bundle_stays_null_not_one_hundred():
    report = create_report(
        {
            "evidence": [],
            "disparities": [],
            "target_url": "https://example.com",
            "profiles_tested": ["baseline_default"],
        },
        "rep_empty",
        diagnose=False,
    )
    assert report.overall_fairness_score is None
    assert report.score_status is ScoreStatus.INSUFFICIENT_EVIDENCE


def test_post_reports_diagnose_false(client):
    body = example_bundle().model_dump(mode="json")
    body["report_id"] = "rep_http"
    body["diagnose"] = False
    response = client.post("/reports", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["report_id"] == "rep_http"
    assert payload["overall_fairness_score"] == 72
    assert payload["analyst"] == "hydrogen"
    assert payload["diagnosis"] == ""


def test_post_reports_null_score_is_json_null(client):
    response = client.post(
        "/reports",
        json={
            "report_id": "rep_null",
            "diagnose": False,
            "evidence": [],
            "disparities": [],
            "target_url": "https://example.com",
            "profiles_tested": [],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["overall_fairness_score"] is None
    assert payload["score_status"] == "INSUFFICIENT_EVIDENCE"


def test_post_reports_diagnose_uses_override(client):
    app.dependency_overrides[get_llm_client] = lambda: MockLLMClient()
    try:
        body = example_bundle().model_dump(mode="json")
        body["report_id"] = "rep_llm"
        body["diagnose"] = True
        response = client.post("/reports", json=body)
        assert response.status_code == 200
        payload = response.json()
        assert payload["analyst"] == "helium"
        assert payload["overall_fairness_score"] == 72
        assert payload["diagnosis"]
    finally:
        app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_demo_checkout_is_served(client):
    response = client.get("/demo/checkout.html")
    assert response.status_code == 200
    assert b"fake-button" in response.content


def test_cancel_marks_running_job(client, monkeypatch):
    from helium.example import example_report
    from lithium.jobs import Job, JobStatus, put as put_job, running_id

    monkeypatch.setattr("lithium.app.run_pipeline", lambda *_a, **_k: example_report())
    put_job(Job(job_id="job_stuck", url="https://example.com", status=JobStatus.running))
    assert running_id() == "job_stuck"
    blocked = client.post(
        "/jobs",
        json={
            "job_id": "job_blocked",
            "url": "https://example.com/checkout",
            "success_selector": "#done",
            "steps": ["#a"],
            "diagnose": False,
        },
    )
    assert blocked.status_code == 409
    cancelled = client.post("/jobs/job_stuck/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["error"] == "cancelled"
    nxt = client.post(
        "/jobs",
        json={
            "job_id": "job_after",
            "url": "https://example.com/checkout",
            "success_selector": "#done",
            "steps": ["#a"],
            "diagnose": False,
        },
    )
    assert nxt.status_code == 202


def test_cancelled_worker_keeps_the_slot(client, monkeypatch):
    from helium.example import example_report
    from lithium.jobs import Job, JobStatus, mark_worker_done, put as put_job, running_id

    monkeypatch.setattr("lithium.app.run_pipeline", lambda *_a, **_k: example_report())
    put_job(
        Job(
            job_id="job_live",
            url="https://example.com",
            status=JobStatus.error,
            error="cancelled",
            worker_alive=True,
        )
    )
    assert running_id() == "job_live"
    blocked = client.post(
        "/jobs",
        json={
            "job_id": "job_blocked2",
            "url": "https://example.com/checkout",
            "success_selector": "#done",
            "steps": ["#a"],
            "diagnose": False,
        },
    )
    assert blocked.status_code == 409
    mark_worker_done("job_live")
    nxt = client.post(
        "/jobs",
        json={
            "job_id": "job_after_live",
            "url": "https://example.com/checkout",
            "success_selector": "#done",
            "steps": ["#a"],
            "diagnose": False,
        },
    )
    assert nxt.status_code == 202


def test_execute_does_not_resurrect_a_cancelled_job(monkeypatch):
    from helium.example import example_report
    from lithium.app import CreateJobBody, _execute_job
    from lithium.jobs import Job, JobStatus, get as get_job, put as put_job

    monkeypatch.setattr("lithium.app.run_pipeline", lambda *_a, **_k: example_report())
    put_job(Job(job_id="job_q", url="https://example.com", worker_alive=True))
    from lithium.jobs import cancel as cancel_job

    cancel_job("job_q")
    _execute_job(
        "job_q",
        CreateJobBody(
            url="https://example.com",
            success_selector="#done",
            steps=["#a"],
            diagnose=False,
        ),
        None,
    )
    live = get_job("job_q")
    assert live is not None
    assert live.error == "cancelled"
    assert live.status is JobStatus.error
    assert live.report is None
    assert live.worker_alive is False


def test_screenshot_diagnose_does_not_clear_cancel(monkeypatch):
    from helium.example import example_report
    from lithium.app import _execute_screenshot_job
    from lithium.jobs import Job, JobStatus, cancel as cancel_job, get as get_job, put as put_job

    monkeypatch.setattr(
        "beryllium.pipeline.run_screenshot_pipeline",
        lambda *_a, **_k: example_report(),
    )

    def fake_diag(report, client=None):
        cancel_job("man_x")
        return report

    monkeypatch.setattr("helium.engine.diagnose", fake_diag)
    put_job(Job(job_id="man_x", url="https://example.com", worker_alive=True))
    _execute_screenshot_job("man_x", "https://example.com", [_png_bytes()], True, None)
    live = get_job("man_x")
    assert live is not None
    assert live.error == "cancelled"
    assert live.status is JobStatus.error
    assert live.worker_alive is False


def test_jobs_run_pipeline_in_background(client, monkeypatch):
    from helium.example import example_report

    def fake_pipeline(job_id, **kwargs):
        assert kwargs["n_trials"] == 2
        assert kwargs["url"] == "https://example.com/checkout"
        assert kwargs["diagnose"] is False
        return example_report()

    monkeypatch.setattr("lithium.app.run_pipeline", fake_pipeline)
    response = client.post(
        "/jobs",
        json={
            "job_id": "job_bg",
            "url": "https://example.com/checkout",
            "n_trials": 2,
            "success_selector": "#done",
            "steps": ["#a"],
            "diagnose": False,
        },
    )
    assert response.status_code == 202
    assert response.json()["status"] in {"queued", "running", "done"}
    done = client.get("/jobs/job_bg")
    assert done.status_code == 200
    assert done.json()["status"] == "done"
    assert done.json()["n_trials"] == 2
    report = client.get("/jobs/job_bg/report")
    assert report.status_code == 200
    assert report.json()["overall_fairness_score"] == 72


def test_jobs_require_steps_or_goal(client):
    response = client.post(
        "/jobs",
        json={
            "url": "https://example.com",
            "success_selector": "#x",
            "n_trials": 1,
        },
    )
    assert response.status_code == 400


def test_jobs_reject_n_trials_zero(client):
    response = client.post(
        "/jobs",
        json={
            "url": "https://example.com",
            "n_trials": 0,
            "success_selector": "#x",
            "steps": ["#a"],
        },
    )
    assert response.status_code == 400


def test_unknown_job_is_404(client):
    assert client.get("/jobs/missing").status_code == 404


def test_jobs_expand_wikipedia_alias(client, monkeypatch):
    from helium.example import example_report

    seen = {}

    def fake_pipeline(job_id, **kwargs):
        seen["url"] = kwargs["url"]
        return example_report()

    monkeypatch.setattr("lithium.app.run_pipeline", fake_pipeline)
    for raw in ("https://wiki/", "https://wikipedia/", "wikipedia"):
        clear_jobs()
        seen.clear()
        response = client.post(
            "/jobs",
            json={
                "job_id": "job_alias",
                "url": raw,
                "success_selector": "#done",
                "steps": ["#a"],
                "diagnose": False,
            },
        )
        assert response.status_code == 202, raw
        assert response.json()["url"] == "https://en.wikipedia.org"
        assert seen["url"] == "https://en.wikipedia.org"


def test_jobs_reject_unknown_bare_hostname(client):
    response = client.post(
        "/jobs",
        json={
            "url": "https://notasite/",
            "success_selector": "#x",
            "steps": ["#a"],
        },
    )
    assert response.status_code == 400
    assert "not a DNS name" in response.json()["detail"]


def test_jobs_reject_file_url(client):
    response = client.post(
        "/jobs",
        json={
            "url": "file:///etc/passwd",
            "success_selector": "#x",
            "steps": ["#a"],
        },
    )
    assert response.status_code == 400


def test_jobs_reject_metadata_ip(client):
    response = client.post(
        "/jobs",
        json={
            "url": "http://169.254.169.254/latest/meta-data",
            "success_selector": "#x",
            "steps": ["#a"],
        },
    )
    assert response.status_code == 400


def test_jobs_reject_path_job_id(client):
    response = client.post(
        "/jobs",
        json={
            "job_id": "../etc",
            "url": "https://example.com",
            "success_selector": "#x",
            "steps": ["#a"],
        },
    )
    assert response.status_code == 400


def test_jobs_ignore_out_root(client, monkeypatch):
    from helium.example import example_report

    seen = {}

    def fake_pipeline(job_id, **kwargs):
        seen.update(kwargs)
        return example_report()

    monkeypatch.setattr("lithium.app.run_pipeline", fake_pipeline)
    response = client.post(
        "/jobs",
        json={
            "job_id": "job_root",
            "url": "https://example.com/checkout",
            "success_selector": "#done",
            "steps": ["#a"],
            "diagnose": False,
            "out_root": "/tmp/evil",
        },
    )
    assert response.status_code == 202
    assert seen["out_root"].endswith("data/sessions") or "data/sessions" in seen["out_root"]
    assert seen.get("out_root") != "/tmp/evil"


def test_job_failure_surfaces_exception(client, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("Page.goto: net::ERR_NAME_NOT_RESOLVED at https://wiki/")

    monkeypatch.setattr("lithium.app.run_pipeline", boom)
    response = client.post(
        "/jobs",
        json={
            "job_id": "job_dns",
            "url": "https://wiki/",
            "success_selector": "#done",
            "goal": "browse the page",
            "diagnose": False,
        },
    )
    assert response.status_code == 202
    done = client.get("/jobs/job_dns").json()
    assert done["status"] == "error"
    assert "ERR_NAME_NOT_RESOLVED" in done["error"]


def test_fill_from_score_writes_hydrogen_math():
    from helium.example import example_report
    from lithium.reports import fill_from_score

    filled = fill_from_score(example_report())
    assert filled.analyst == "hydrogen"
    assert filled.overall_fairness_score == 72
    assert "Fairness score 72/100" in filled.diagnosis
    assert "Bottleneck" in filled.diagnosis
    assert filled.remediation


def test_diagnose_failure_keeps_score(client, monkeypatch):
    from helium.example import example_report

    monkeypatch.setattr(
        "lithium.app.run_pipeline", lambda *a, **k: example_report()
    )

    def boom(*_a, **_k):
        raise RuntimeError("gpu down")

    monkeypatch.setattr("helium.engine.diagnose", boom)
    response = client.post(
        "/jobs",
        json={
            "job_id": "job_diag",
            "url": "https://example.com/checkout",
            "success_selector": "#done",
            "steps": ["#a"],
            "diagnose": True,
        },
    )
    assert response.status_code == 202
    done = client.get("/jobs/job_diag").json()
    assert done["status"] == "done"
    assert done["report"]["overall_fairness_score"] == 72
    assert done["warning"] is None
    assert "Fairness score" in done["report"]["diagnosis"]
    assert done["report"]["analyst"] == "hydrogen"


def test_diagnose_timeout_keeps_score(client, monkeypatch):
    from helium.example import example_report

    monkeypatch.setattr(
        "lithium.app.run_pipeline", lambda *a, **k: example_report()
    )

    def boom(*_a, **_k):
        raise TimeoutError("FunctionCall timed out")

    monkeypatch.setattr("helium.engine.diagnose", boom)
    response = client.post(
        "/jobs",
        json={
            "job_id": "job_diag_to",
            "url": "https://example.com/checkout",
            "success_selector": "#done",
            "steps": ["#a"],
            "diagnose": True,
        },
    )
    assert response.status_code == 202
    done = client.get("/jobs/job_diag_to").json()
    assert done["status"] == "done"
    assert done["report"]["overall_fairness_score"] == 72
    assert done["warning"] is None
    assert "Fairness score" in done["report"]["diagnosis"]
    assert done["report"]["analyst"] == "hydrogen"


# --- the success selector is what makes task_completed a measurement --------


@pytest.mark.parametrize("selector", ["body", "HTML", ":root", "*"])
def test_a_selector_that_cannot_measure_completion_is_refused(client, selector):
    response = client.post(
        "/jobs",
        json={
            "url": "https://example.com/checkout",
            "success_selector": selector,
            "goal": "Place the order",
        },
    )
    assert response.status_code == 400
    assert "success_selector" in response.json()["detail"]


def test_goal_without_selector_is_accepted(client, monkeypatch):
    from helium.example import example_report

    seen = {}

    def fake_pipeline(job_id, **kwargs):
        seen.update(kwargs)
        return example_report()

    monkeypatch.setattr("lithium.app.run_pipeline", fake_pipeline)
    response = client.post(
        "/jobs",
        json={
            "job_id": "job_goal_only",
            "url": "https://en.wikipedia.org",
            "goal": "Open the English article from the main page",
            "diagnose": False,
        },
    )
    assert response.status_code == 202
    assert seen["goal"] == "Open the English article from the main page"
    assert seen["success_selector"] == ""


# --- manual capture: screenshots a human took ------------------------------


def _png_bytes(color: str = "white") -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _b64(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


def test_screenshot_job_scores_without_a_browser(client, tmp_path, monkeypatch):
    from nitrogen import MockVLClient
    from lithium.app import app as lithium_app, get_vision_client

    monkeypatch.setattr("lithium.app._SESSION_ROOT", tmp_path)
    lithium_app.dependency_overrides[get_vision_client] = lambda: MockVLClient(
        "The primary button sits below the fold."
    )
    try:
        response = client.post(
            "/jobs/screenshots",
            json={
                "job_id": "man_ok",
                "url": "https://example.com/checkout",
                "images": [_b64(_png_bytes()), _b64(_png_bytes("gray"))],
                "diagnose": False,
            },
        )
        assert response.status_code == 202
        done = client.get("/jobs/man_ok").json()
    finally:
        lithium_app.dependency_overrides.pop(get_vision_client, None)

    assert done["status"] == "done"
    report = done["report"]
    # One profile, no constrained pair, so there is no disparity to score. A
    # null score with INSUFFICIENT_EVIDENCE is the honest answer, not a 100.
    assert report["overall_fairness_score"] is None
    assert report["score_status"] == ScoreStatus.INSUFFICIENT_EVIDENCE.value
    assert report["profiles_tested"] == ["baseline_default"]
    assert (tmp_path / "man_ok_1" / "screenshot.png").is_file()
    assert (tmp_path / "man_ok_2" / "screenshot.png").is_file()


def test_screenshot_job_rejects_unreadable_uploads(client):
    assert client.post(
        "/jobs/screenshots", json={"url": "https://example.com", "images": []}
    ).status_code == 400
    assert client.post(
        "/jobs/screenshots",
        json={"url": "https://example.com", "images": [_b64(b"not an image")]},
    ).status_code == 400
    assert client.post(
        "/jobs/screenshots",
        json={"url": "https://example.com", "images": [_b64(_png_bytes())] * 13},
    ).status_code == 400


def test_jobs_expand_gmail_alias(client, monkeypatch):
    """gmail.com serves marketing when signed out; the product is the mail host."""
    from helium.example import example_report

    seen = {}

    def fake_pipeline(job_id, **kwargs):
        seen["url"] = kwargs["url"]
        return example_report()

    monkeypatch.setattr("lithium.app.run_pipeline", fake_pipeline)
    for raw in ("https://gmail/", "gmail.com", "https://www.gmail.com"):
        clear_jobs()
        seen.clear()
        response = client.post(
            "/jobs",
            json={
                "job_id": "job_gmail",
                "url": raw,
                "success_selector": "#gb",
                "goal": "Open the first message in the inbox",
                "diagnose": False,
            },
        )
        assert response.status_code == 202, raw
        assert response.json()["url"] == "https://mail.google.com"
        assert seen["url"] == "https://mail.google.com"
