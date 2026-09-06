"""FastAPI surface for lithium.create_report and beryllium jobs."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from urllib.parse import urlparse

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from beryllium import run_pipeline
from lithium.jobs import (
    Job,
    JobStatus,
    append_event,
    cancel as cancel_job,
    get as get_job,
    mark_worker_done,
    put as put_job,
    running_id,
    snapshot,
    snapshot_job,
    update,
)
from lithium.reports import create_report, fill_from_score
from lithium.safety import check_job_url, job_id_ok

log = logging.getLogger("lithium")
app = FastAPI(title="CoHERence Lithium", version="v1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SESSION_ROOT = Path("data/sessions").resolve()
_DEMO_PAGE = (
    Path(__file__).resolve().parent.parent / "client" / "public" / "demo" / "checkout.html"
)
_DEMO_PATH = "/demo/checkout.html"
# Manual capture upload caps. Images arrive base64 in the JSON body, so the
# whole batch sits in memory while it is decoded.
MAX_SCREENSHOTS = 12
MAX_SCREENSHOT_BYTES = 6 * 1024**2


class CreateReportBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    report_id: str = ""
    diagnose: bool = True
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    disparities: list[dict[str, Any]] = Field(default_factory=list)
    target_url: str = ""
    profiles_tested: list[str] = Field(default_factory=list)


class CreateJobBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str
    n_trials: int = 1
    success_selector: str = ""
    steps: list[str] | None = None
    goal: str | None = None
    profile_ids: list[str] | None = None
    diagnose: bool = True
    plan_once: bool = False
    job_id: str = ""
    seed: int | None = None


class CreateScreenshotJobBody(BaseModel):
    """Screenshots a human captured by hand. No browser, no task, no selector."""

    model_config = ConfigDict(extra="ignore")

    url: str
    images: list[str] = Field(default_factory=list)
    diagnose: bool = True
    job_id: str = ""


def get_llm_client():
    """Live Helium client is resolved inside helium.diagnose when this is None."""
    return None


def get_vision_client():
    """Live Fluorine client is constructed in the job when this is None."""
    return None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/demo/checkout.html")
def demo_checkout():
    """Fixture page for capture. Playwright hits Lithium, not the Vite SPA."""
    if not _DEMO_PAGE.is_file():
        raise HTTPException(status_code=404, detail="demo missing")
    return FileResponse(_DEMO_PAGE, media_type="text/html")


@app.get("/profiles")
def profiles() -> dict:
    import boron

    return {"profiles": boron.list_profiles()}


@app.post("/reports")
def post_report(
    body: CreateReportBody, llm_client=Depends(get_llm_client)
) -> dict:
    report_id = body.report_id or _new_id("rep")
    payload = body.model_dump(
        exclude={"report_id", "diagnose"},
        mode="json",
    )
    report = create_report(
        payload,
        report_id,
        diagnose=body.diagnose,
        llm_client=llm_client if body.diagnose else None,
    )
    return report.model_dump(mode="json")


def _cancelled(job_id: str) -> bool:
    live = get_job(job_id)
    return live is not None and live.error == "cancelled"


def _execute_job(job_id: str, body: CreateJobBody, llm_client) -> None:
    def on_progress(event: dict) -> None:
        if _cancelled(job_id):
            raise RuntimeError("cancelled")
        append_event(job_id, event)

    try:
        if _cancelled(job_id):
            return
        update(job_id, status=JobStatus.running, stage="capture")
        vl_client = None
        text_client = None
        vision_client = None
        if body.goal:
            from nitrogen import ModalVLClient

            vl_client = ModalVLClient()
            try:
                from oxygen.client import GpuTextClient

                text_client = GpuTextClient()
            except Exception:
                text_client = None
            try:
                from fluorine import ModalVLClient as FluorineVLClient

                vision_client = FluorineVLClient()
            except Exception:
                vision_client = None
        report = run_pipeline(
            job_id,
            url=body.url,
            n_trials=body.n_trials,
            profile_ids=body.profile_ids,
            success_selector=body.success_selector,
            steps=body.steps,
            goal=body.goal,
            vl_client=vl_client,
            plan_once=body.plan_once,
            out_root=str(_SESSION_ROOT),
            seed=body.seed,
            diagnose=False,
            on_progress=on_progress,
            text_client=text_client,
            vision_client=vision_client,
        )
        if _cancelled(job_id):
            return
        if body.diagnose:
            update(job_id, stage="diagnose")
            try:
                from helium import diagnose as helium_diagnose

                report = helium_diagnose(
                    report, client=llm_client if llm_client else None
                )
            except Exception:
                report = fill_from_score(report)
        if _cancelled(job_id):
            return
        update(
            job_id,
            status=JobStatus.done,
            stage="done",
            report=report,
            error=None,
        )
    except Exception as exc:
        if _cancelled(job_id):
            return
        log.exception("job %s failed", job_id)
        detail = next((line.strip() for line in str(exc).splitlines() if line.strip()), "job failed")
        update(job_id, status=JobStatus.error, stage="error", error=detail[:500])
    finally:
        mark_worker_done(job_id)


def _capture_url(url: str, api_base: str) -> str:
    path = urlparse(url).path
    if path.endswith(_DEMO_PATH):
        return api_base.rstrip("/") + _DEMO_PATH
    return url


@app.post("/jobs", status_code=202)
def post_job(
    body: CreateJobBody,
    background: BackgroundTasks,
    request: Request,
    llm_client=Depends(get_llm_client),
) -> dict:
    if body.n_trials < 1:
        raise HTTPException(status_code=400, detail="n_trials must be >= 1")
    if bool(body.steps) == bool(body.goal):
        raise HTTPException(
            status_code=400, detail="pass steps or goal, not both or neither"
        )
    selector = (body.success_selector or "").strip()
    if selector.lower() in {"body", "html", ":root", "*"}:
        raise HTTPException(
            status_code=400,
            detail=f"success_selector {selector!r} is visible before the task starts",
        )
    if body.steps and not selector:
        raise HTTPException(
            status_code=400,
            detail="success_selector is required when using steps",
        )
    try:
        url = check_job_url(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    body = body.model_copy(update={"url": url, "success_selector": selector})
    job_id = body.job_id or _new_id("job")
    if not job_id_ok(job_id):
        raise HTTPException(status_code=400, detail="job_id not allowed")
    busy = running_id()
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"capture already running ({busy}); cancel it first",
        )
    try:
        job = put_job(
            Job(
                job_id=job_id,
                n_trials=body.n_trials,
                url=body.url,
                worker_alive=True,
            )
        )
    except KeyError:
        raise HTTPException(status_code=409, detail="job_id already exists") from None
    capture = body.model_copy(
        update={"url": _capture_url(body.url, str(request.base_url))}
    )
    background.add_task(_execute_job, job.job_id, capture, llm_client)
    return snapshot_job(job)


def _decode_images(images: list[str]) -> list[bytes]:
    """base64 (or a data: URL) -> PNG bytes. Anything Pillow cannot open is a 400."""
    import base64
    import binascii
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    if not images:
        raise HTTPException(status_code=400, detail="at least one screenshot is required")
    if len(images) > MAX_SCREENSHOTS:
        raise HTTPException(
            status_code=400, detail=f"at most {MAX_SCREENSHOTS} screenshots per job"
        )
    out: list[bytes] = []
    for index, item in enumerate(images, start=1):
        payload = item.split(",", 1)[-1] if item.startswith("data:") else item
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(
                status_code=400, detail=f"screenshot {index} is not valid base64"
            ) from None
        if len(raw) > MAX_SCREENSHOT_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"screenshot {index} is larger than {MAX_SCREENSHOT_BYTES // 1024**2} MiB",
            )
        # Normalise to PNG: `from_screenshots` writes every image to
        # screenshot.png, and a JPEG under that name would be served back as
        # image/png by /preview.
        try:
            image = Image.open(BytesIO(raw))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(
                status_code=400, detail=f"screenshot {index} is not a readable image"
            ) from None
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        out.append(buffer.getvalue())
    return out


def _execute_screenshot_job(
    job_id: str,
    url: str,
    images: list[bytes],
    diagnose: bool,
    llm_client,
    vision_client=None,
) -> None:
    import tempfile

    def on_progress(event: dict) -> None:
        if _cancelled(job_id):
            raise RuntimeError("cancelled")
        append_event(job_id, event)

    try:
        if _cancelled(job_id):
            return
        update(job_id, status=JobStatus.running, stage="describe")
        if vision_client is None:
            try:
                from fluorine import ModalVLClient as FluorineVLClient

                vision_client = FluorineVLClient()
            except Exception:
                vision_client = None
        with tempfile.TemporaryDirectory() as staging:
            paths = []
            for index, raw in enumerate(images, start=1):
                path = Path(staging) / f"{index:03d}.png"
                path.write_bytes(raw)
                paths.append(str(path))
            from beryllium import run_screenshot_pipeline

            report = run_screenshot_pipeline(
                job_id,
                paths,
                url=url,
                out_root=str(_SESSION_ROOT),
                vision_client=vision_client,
                diagnose=False,
                on_progress=on_progress,
            )
        if _cancelled(job_id):
            return
        if diagnose:
            update(job_id, stage="diagnose")
            try:
                from helium import diagnose as helium_diagnose

                report = helium_diagnose(
                    report, client=llm_client if llm_client else None
                )
            except Exception:
                report = fill_from_score(report)
        if _cancelled(job_id):
            return
        update(
            job_id,
            status=JobStatus.done,
            stage="done",
            report=report,
            error=None,
        )
    except Exception as exc:
        if _cancelled(job_id):
            return
        log.exception("screenshot job %s failed", job_id)
        detail = next(
            (line.strip() for line in str(exc).splitlines() if line.strip()),
            "job failed",
        )
        update(job_id, status=JobStatus.error, stage="error", error=detail[:500])
    finally:
        mark_worker_done(job_id)


@app.post("/jobs/screenshots", status_code=202)
def post_screenshot_job(
    body: CreateScreenshotJobBody,
    background: BackgroundTasks,
    llm_client=Depends(get_llm_client),
    vision_client=Depends(get_vision_client),
) -> dict:
    """Manual capture: the human visited the pages, these are the views.

    Vision only. Nothing here is scored against a baseline, so the report comes
    back INSUFFICIENT_EVIDENCE with a null score by design -- see
    `beryllium.run_screenshot_pipeline`.
    """
    try:
        url = check_job_url(body.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    images = _decode_images(body.images)
    job_id = body.job_id or _new_id("man")
    if not job_id_ok(job_id):
        raise HTTPException(status_code=400, detail="job_id not allowed")
    busy = running_id()
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"capture already running ({busy}); cancel it first",
        )
    try:
        job = put_job(Job(job_id=job_id, n_trials=1, url=url, worker_alive=True))
    except KeyError:
        raise HTTPException(status_code=409, detail="job_id already exists") from None
    background.add_task(
        _execute_screenshot_job,
        job.job_id,
        url,
        images,
        body.diagnose,
        llm_client,
        vision_client,
    )
    return snapshot_job(job)


@app.post("/jobs/{job_id}/cancel")
def post_cancel(job_id: str) -> dict:
    job = cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    payload = snapshot(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return payload


@app.get("/jobs/{job_id}")
def read_job(job_id: str) -> dict:
    payload = snapshot(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return payload


@app.get("/jobs/{job_id}/report")
def read_job_report(job_id: str) -> dict:
    payload = snapshot(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    if payload["status"] == JobStatus.error.value:
        raise HTTPException(status_code=409, detail=payload["error"] or "job failed")
    if payload["status"] != JobStatus.done.value or payload["report"] is None:
        raise HTTPException(status_code=409, detail="report not ready")
    return payload["report"]


@app.get("/jobs/{job_id}/preview")
def read_job_preview(job_id: str):
    payload = snapshot(job_id) or {}
    raw = payload.get("preview")
    path = Path(raw).resolve() if raw else None
    if path is None or not path.is_file():
        matches = sorted(
            _SESSION_ROOT.glob(f"{job_id}_*/screenshot.png"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        path = matches[0].resolve() if matches else None
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="preview not ready")
    try:
        path.relative_to(_SESSION_ROOT)
    except ValueError:
        raise HTTPException(status_code=404, detail="preview not ready") from None
    if path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="preview not ready")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
