"""In-process job store. Lithium v1; not a durable queue."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

from hydrogen.models import HydrogenReport


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


@dataclass
class Job:
    job_id: str
    n_trials: int = 1
    url: str = ""
    status: JobStatus = JobStatus.queued
    report: HydrogenReport | None = None
    error: str | None = None
    warning: str | None = None
    stage: str = ""
    current_profile: str = ""
    preview: str | None = None
    events: list[dict] = field(default_factory=list)
    # Stays true until the background worker returns, including after cancel.
    worker_alive: bool = False


_lock = threading.Lock()
_JOBS: dict[str, Job] = {}


def running_id() -> str | None:
    with _lock:
        for job in _JOBS.values():
            if job.worker_alive or job.status in (JobStatus.queued, JobStatus.running):
                return job.job_id
        return None


def put(job: Job) -> Job:
    with _lock:
        if job.job_id in _JOBS:
            raise KeyError(job.job_id)
        _JOBS[job.job_id] = job
        return job


def get(job_id: str) -> Job | None:
    with _lock:
        return _JOBS.get(job_id)


def update(job_id: str, **fields) -> Job:
    with _lock:
        job = _JOBS[job_id]
        if job.error == "cancelled":
            fields = {key: value for key, value in fields.items() if key == "worker_alive"}
        for key, value in fields.items():
            setattr(job, key, value)
        return job


def append_event(job_id: str, event: dict) -> Job:
    with _lock:
        job = _JOBS[job_id]
        job.events.append(event)
        stage = event.get("stage")
        if stage:
            job.stage = str(stage)
        profile = event.get("profile_id")
        if profile:
            job.current_profile = str(profile)
        shot = event.get("screenshot")
        if shot:
            job.preview = str(shot)
        if stage == "plan_failed" and event.get("error"):
            job.warning = str(event["error"])
        return job


def _dump(job: Job) -> dict:
    payload = {
        "job_id": job.job_id,
        "status": job.status.value,
        "n_trials": job.n_trials,
        "url": job.url,
        "error": job.error,
        "warning": job.warning,
        "stage": job.stage,
        "current_profile": job.current_profile,
        "preview": job.preview,
        "events": list(job.events),
        "report": None,
    }
    if job.report is not None:
        payload["report"] = job.report.model_dump(mode="json")
    return payload


def snapshot(job_id: str) -> dict | None:
    with _lock:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        return _dump(job)


def snapshot_job(job: Job) -> dict:
    with _lock:
        live = _JOBS.get(job.job_id, job)
        return _dump(live)


def cancel(job_id: str) -> Job | None:
    with _lock:
        job = _JOBS.get(job_id)
        if job is None:
            return None
        if job.status in (JobStatus.queued, JobStatus.running):
            job.status = JobStatus.error
            job.stage = "error"
            job.error = "cancelled"
        return job


def mark_worker_done(job_id: str) -> None:
    with _lock:
        job = _JOBS.get(job_id)
        if job is not None:
            job.worker_alive = False


def clear() -> None:
    with _lock:
        _JOBS.clear()
