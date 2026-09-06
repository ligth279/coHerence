"""beryllium.run_pipeline — capture, aggregate, score. Hydrogen never sees N."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import hydrogen

BASELINE_PROFILE_ID = "baseline_default"
_COMPLETION_METRIC = "task_completion_rate"
_FRICTION_METRIC = "composite_friction_score"
_VIEW_FRICTION_RULE = "FLUORINE_VIEW_FRICTION"
_WARNING_RATIO = 1.5
_CRITICAL_RATIO = 2.5
_FRICTION_DELTA = 20.0


def run_pipeline(
    job_id: str,
    url: str | None = None,
    n_trials: int = 1,
    *,
    contract2_path: str | Path | None = None,
    profile_ids: list[str] | None = None,
    success_selector: str | None = None,
    steps: list[str] | None = None,
    goal: str | None = None,
    vl_client=None,
    plan_once: bool = False,
    max_steps: int | None = None,
    out_root: str | None = None,
    seed: int | None = None,
    diagnose: bool = False,
    llm_client=None,
    text_client=None,
    vision_client=None,
    baseline_profile_id: str = BASELINE_PROFILE_ID,
    on_progress=None,
):
    """Run Dev 1 → Dev 2 → hydrogen.evaluate. Optionally helium.diagnose.

    `n_trials` is `boron.run_suite(..., runs=N)`. Hydrogen is not passed N.
    Pass `contract2_path` to skip capture (already-aggregated Contract 2).
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if contract2_path and url:
        raise ValueError("pass url or contract2_path, not both")
    if contract2_path:
        payload = _load_contract2(contract2_path)
        tested = list(payload.get("profiles_tested") or [])
    elif url:
        profile_ids = _profiles(profile_ids, baseline_profile_id)
        try:
            records = _capture(
                url=url,
                job_id=job_id,
                n_trials=n_trials,
                profile_ids=profile_ids,
                success_selector=success_selector,
                steps=steps,
                goal=goal,
                vl_client=vl_client,
                plan_once=plan_once,
                max_steps=max_steps,
                out_root=out_root,
                seed=seed,
                on_progress=on_progress,
            )
        except Exception as exc:
            from boron.runner import PlanFailed

            if not isinstance(exc, PlanFailed) or not getattr(exc, "records", None):
                raise
            records = list(exc.records)
            if on_progress is not None:
                on_progress(
                    {
                        "stage": "plan_failed",
                        "error": (
                            "Nitrogen could not finish the goal on "
                            f"{records[0].profile_id}, so other profiles were not "
                            "replayed. Scoring the capture we have."
                        ),
                    }
                )
        folded, rates = _aggregate_sessions(records)
        frictions = _trial_frictions(records)
        if on_progress is not None:
            on_progress({"stage": "rules"})
        captured_ids: list[str] = []
        for record in records:
            if record.profile_id not in captured_ids:
                captured_ids.append(record.profile_id)
        payload = _build_contract2(
            folded,
            rates,
            frictions,
            url=url,
            profile_ids=captured_ids or profile_ids,
            baseline_profile_id=baseline_profile_id,
            session_ids=[r.session_id for r in records],
        )
        tested = list(captured_ids or profile_ids)
        extra = _host_evidence(
            folded, baseline_profile_id, text_client, vision_client
        )
        if extra:
            payload["evidence"] = list(payload.get("evidence") or []) + extra
    else:
        raise ValueError("url or contract2_path is required")

    if tested and not payload.get("profiles_tested"):
        payload["profiles_tested"] = tested

    if on_progress is not None:
        on_progress({"stage": "score"})
    report = hydrogen.evaluate(hydrogen.parse_contract2(payload), job_id)
    if diagnose:
        from helium import diagnose as helium_diagnose

        report = helium_diagnose(report, client=llm_client)
    return report


def _load_contract2(path: str | Path) -> dict:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _profiles(profile_ids: list[str] | None, baseline_profile_id: str) -> list[str]:
    if profile_ids is None:
        import boron

        ids = list(boron.list_profiles())
    else:
        ids = list(profile_ids)
    if baseline_profile_id not in ids:
        ids.insert(0, baseline_profile_id)
    return ids


def _capture(
    *,
    url: str,
    job_id: str,
    n_trials: int,
    profile_ids: list[str],
    success_selector: str | None,
    steps: list[str] | None,
    goal: str | None,
    vl_client,
    plan_once: bool,
    max_steps: int | None,
    out_root: str | None,
    seed: int | None,
    on_progress=None,
):
    if not success_selector and not goal:
        raise ValueError("success_selector is required when capturing without a goal")
    success_selector = success_selector or ""
    kwargs = {
        "url": url,
        "profile_ids": profile_ids,
        "session_id_prefix": job_id,
        "success_selector": success_selector,
        "steps": steps,
        "goal": goal,
        "vl_client": vl_client,
        "plan_once": plan_once,
        "runs": n_trials,
        "on_progress": on_progress,
    }
    if out_root is not None:
        kwargs["out_root"] = out_root
    if seed is not None:
        kwargs["seed"] = seed
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return _run_suite(**kwargs)


def _run_suite(**kwargs):
    import boron

    return boron.run_suite(**kwargs)


def _mean_int(values) -> int:
    values = list(values)
    return int(round(sum(values) / len(values)))


def _aggregate_sessions(records):
    """One session per profile. Completion rate is successes / N, not a bool."""
    if not records:
        raise ValueError("boron.run_suite returned no sessions")
    groups: dict[str, list] = defaultdict(list)
    order: list[str] = []
    for record in records:
        if record.profile_id not in groups:
            order.append(record.profile_id)
        groups[record.profile_id].append(record)

    folded = []
    rates: dict[str, float] = {}
    for profile_id in order:
        runs = groups[profile_id]
        tels = [r.telemetry for r in runs]
        n = len(runs)
        successes = sum(1 for t in tels if t.task_completed)
        rates[profile_id] = successes / n
        failed: list[str] = []
        for t in tels:
            for selector in getattr(t, "failed_selectors", None) or []:
                if selector not in failed:
                    failed.append(selector)
        data = runs[0].model_dump()
        data["telemetry"] = {
            **data["telemetry"],
            "completion_time_ms": _mean_int(t.completion_time_ms for t in tels),
            "task_completed": successes == n,
            "total_clicks": _mean_int(t.total_clicks for t in tels),
            "dead_clicks": _mean_int(t.dead_clicks for t in tels),
            "keyboard_nav_steps": _mean_int(t.keyboard_nav_steps for t in tels),
            "missed_clicks": _mean_int(getattr(t, "missed_clicks", 0) for t in tels),
            "error_count": _mean_int(t.error_count for t in tels),
            "failed_selectors": failed,
        }
        folded.append(type(runs[0]).model_validate(data))
    return folded, rates


def _host_evidence(folded, baseline_id: str, text_client, vision_client) -> list[dict]:
    """Oxygen (text) and Fluorine (vision) notes. Callers own prompts. Optional."""
    if not folded or (text_client is None and vision_client is None):
        return []
    from carbon.schemas.contracts import EvidenceItem, Severity

    base = next((row for row in folded if row.profile_id == baseline_id), folded[0])
    arts = base.artifacts
    if isinstance(arts, dict):
        html_path = arts.get("html_path")
        shot_path = arts.get("screenshot_path")
    else:
        html_path = getattr(arts, "html_path", None)
        shot_path = getattr(arts, "screenshot_path", None)
    items: list[dict] = []
    if text_client is not None and html_path:
        try:
            html = Path(html_path).read_text(encoding="utf-8", errors="ignore")[:3500]
            note = text_client.complete(
                "You flag inclusive-design friction in page text. Two short sentences. Constraints only, no personas.",
                html,
            ).strip()[:400]
            if note:
                items.append(
                    EvidenceItem(
                        element_selector="body",
                        rule_id="OXYGEN_PAGE_TEXT",
                        severity=Severity.INFO,
                        metric_value=note,
                    ).model_dump(mode="json")
                )
        except Exception:
            pass
    if vision_client is not None and shot_path:
        note = _vision_note(vision_client, shot_path)
        if note:
            items.append(
                EvidenceItem(
                    element_selector="screenshot",
                    rule_id=_VIEW_FRICTION_RULE,
                    severity=Severity.INFO,
                    metric_value=note,
                ).model_dump(mode="json")
            )
    return items


def _vision_note(vision_client, shot_path) -> str:
    """One friction note for one screenshot. Never raises: a note is optional."""
    import base64

    try:
        image_b64 = base64.b64encode(Path(shot_path).read_bytes()).decode("ascii")
        return vision_client.complete(
            "You flag visible UI friction in a screenshot. Two short sentences. Name controls if you can.",
            "What on this page is hard to use?",
            image_b64,
        ).strip()[:400]
    except Exception:
        return ""


def _trial_frictions(records) -> dict[str, float]:
    """Mean per-trial friction. Do not score the folded all-or-nothing bool."""
    from carbon.disparity.metrics import compute_friction_score
    from carbon.schemas.contracts import TelemetryData

    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        tel = record.telemetry
        payload = tel.model_dump() if hasattr(tel, "model_dump") else dict(tel)
        groups[record.profile_id].append(
            compute_friction_score(TelemetryData.model_validate(payload))
        )
    return {
        profile_id: round(sum(scores) / len(scores), 2)
        for profile_id, scores in groups.items()
    }


def _build_contract2(
    folded,
    rates: dict[str, float],
    frictions: dict[str, float],
    *,
    url: str,
    profile_ids: list[str],
    baseline_profile_id: str,
    session_ids: list[str],
) -> dict:
    import carbon
    from carbon.disparity.metrics import compute_disparity_ratio
    from carbon.schemas.contracts import DisparityItem, Severity

    sessions = [
        carbon.RawSessionArtifacts.model_validate(record.model_dump())
        for record in folded
    ]
    by_id = {session.profile_id: session for session in sessions}
    if baseline_profile_id not in by_id:
        raise ValueError(f"no {baseline_profile_id} session to compare against")
    baseline = by_id[baseline_profile_id]
    constrained = [
        session for session in sessions if session.profile_id != baseline_profile_id
    ]

    engine = carbon.RuleEngine()
    evidence = engine.evaluate_session_artifacts(baseline)
    evidence = engine.attribute_from_failures(evidence, sessions)
    disparities = carbon.DisparityEngine().analyze_sessions(
        baseline, constrained, evidence
    )

    base_rate = rates.get(baseline_profile_id, 0.0)
    completion = []
    for profile_id, rate in rates.items():
        if profile_id == baseline_profile_id:
            continue
        ratio = compute_disparity_ratio(base_rate, rate, higher_is_better=True)
        if ratio >= _WARNING_RATIO or base_rate > rate:
            completion.append(
                DisparityItem(
                    metric=_COMPLETION_METRIC,
                    baseline_value=round(base_rate, 2),
                    constrained_value=round(rate, 2),
                    disparity_ratio=ratio,
                    disadvantaged_group=profile_id,
                    delta_absolute=round(rate - base_rate, 2),
                    severity=(
                        Severity.CRITICAL
                        if ratio >= _CRITICAL_RATIO
                        else Severity.WARNING
                    ),
                    interpretation=(
                        f"{profile_id} experienced a {ratio:.1f}x lower "
                        "completion rate compared to baseline."
                    ),
                )
            )
    rest = [
        row
        for row in disparities
        if row.metric not in {_COMPLETION_METRIC, _FRICTION_METRIC}
    ]
    base_friction = frictions.get(baseline_profile_id, 0.0)
    friction_rows = []
    for profile_id, score in frictions.items():
        if profile_id == baseline_profile_id:
            continue
        ratio = compute_disparity_ratio(
            max(5.0, base_friction), max(5.0, score), higher_is_better=False
        )
        if ratio >= _WARNING_RATIO or (score - base_friction) >= _FRICTION_DELTA:
            friction_rows.append(
                DisparityItem(
                    metric=_FRICTION_METRIC,
                    baseline_value=round(base_friction, 2),
                    constrained_value=round(score, 2),
                    disparity_ratio=ratio,
                    disadvantaged_group=profile_id,
                    delta_absolute=round(score - base_friction, 1),
                    severity=(
                        Severity.CRITICAL
                        if ratio >= _CRITICAL_RATIO
                        else Severity.WARNING
                    ),
                    interpretation=(
                        f"{profile_id} encountered an aggregate friction score "
                        f"of {score} vs baseline {base_friction}."
                    ),
                )
            )
    contract = carbon.EvidenceRecord(
        evidence=evidence,
        disparities=completion + friction_rows + rest,
        target_url=url,
        profiles_tested=list(profile_ids),
        session_ids=list(session_ids),
    )
    return contract.model_dump(mode="json")


def run_screenshot_pipeline(
    job_id: str,
    image_paths: list[str],
    *,
    url: str,
    out_root: str | None = None,
    vision_client=None,
    diagnose: bool = False,
    llm_client=None,
    on_progress=None,
):
    """Score screenshots a human captured by hand. No browser, vision only.

    `boron.from_screenshots` writes one Contract 1 record per image, stamped
    `boron-manual-png-v1`. There is no DOM, no computed style and no a11y tree
    behind a PNG, so carbon's geometry, contrast and WCAG rules are not run --
    they would find nothing and reporting that as a clean page would be a lie.
    One profile means no baseline-vs-constrained pair either, so `disparities`
    is empty by construction and hydrogen returns INSUFFICIENT_EVIDENCE with a
    null score. What this path does produce is one vision note per view, plus
    helium's diagnosis over them when `diagnose` is on.
    """
    import boron
    from carbon.schemas.contracts import EvidenceItem, Severity

    paths = list(image_paths)
    if not paths:
        raise ValueError("at least one screenshot is required")

    kwargs = {} if out_root is None else {"out_root": out_root}
    records = boron.from_screenshots(
        paths, url=url, session_id=job_id, **kwargs
    )

    evidence = []
    for record in records:
        if on_progress is not None:
            on_progress(
                {
                    "stage": "describe",
                    "profile_id": record.profile_id,
                    "session_id": record.session_id,
                    "screenshot": record.artifacts.screenshot_path,
                }
            )
        if vision_client is None:
            continue
        note = _vision_note(vision_client, record.artifacts.screenshot_path)
        if note:
            evidence.append(
                EvidenceItem(
                    element_selector=f"screenshot:{record.session_id}",
                    rule_id=_VIEW_FRICTION_RULE,
                    severity=Severity.INFO,
                    metric_value=note,
                )
            )

    import carbon

    payload = carbon.EvidenceRecord(
        evidence=evidence,
        disparities=[],
        target_url=url,
        profiles_tested=[BASELINE_PROFILE_ID],
        session_ids=[record.session_id for record in records],
    ).model_dump(mode="json")

    if on_progress is not None:
        on_progress({"stage": "score"})
    report = hydrogen.evaluate(hydrogen.parse_contract2(payload), job_id)
    if diagnose:
        from helium import diagnose as helium_diagnose

        report = helium_diagnose(report, client=llm_client)
    return report
