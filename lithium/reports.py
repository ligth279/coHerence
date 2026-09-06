"""lithium.create_report — Hydrogen score, optional Helium text. No rescoring."""

from __future__ import annotations

import hydrogen
from hydrogen.models import EvidenceBundle, HydrogenReport


def create_report(
    payload: dict | EvidenceBundle,
    report_id: str,
    *,
    diagnose: bool = True,
    llm_client=None,
) -> HydrogenReport:
    """Contract 2 JSON (or EvidenceBundle) → Contract 3 HydrogenReport.

    Always `hydrogen.evaluate`. `helium.diagnose` only when `diagnose=True`.
    The integer score is never rewritten here.
    """
    if not isinstance(payload, dict):
        payload = payload.model_dump(mode="json")
    report = hydrogen.evaluate(hydrogen.parse_contract2(payload), report_id)
    if not diagnose:
        return report
    from helium import diagnose as helium_diagnose

    try:
        return helium_diagnose(report, client=llm_client)
    except Exception:
        return fill_from_score(report)


def fill_from_score(report: HydrogenReport) -> HydrogenReport:
    """Turn Hydrogen's numbers into the report body when Helium does not run."""
    if (report.diagnosis or "").strip():
        return report
    breakdown = report.breakdown
    score = report.overall_fairness_score
    status = getattr(report.score_status, "value", report.score_status)
    parts: list[str] = []
    if score is None:
        parts.append(f"Fairness score is null ({status}, {report.scoring_policy}).")
    else:
        parts.append(f"Fairness score {score}/100 ({status}, {report.scoring_policy}).")
    if breakdown and breakdown.bottleneck_group:
        parts.append(
            "Bottleneck "
            f"{breakdown.bottleneck_group} on {breakdown.bottleneck_metric}: "
            f"baseline {breakdown.bottleneck_baseline}, "
            f"constrained {breakdown.bottleneck_constrained}, "
            f"absolute gap {breakdown.bottleneck_abs_gap}, "
            f"ratio {breakdown.max_disparity_ratio}."
        )
    if report.profiles_tested:
        parts.append("Profiles tested: " + ", ".join(report.profiles_tested) + ".")
    for row in (report.disparities or [])[:8]:
        parts.append(
            f"{row.metric} for {row.disadvantaged_group}: "
            f"{row.baseline_value} vs {row.constrained_value} "
            f"(ratio {row.disparity_ratio:.2f})."
        )
    rem: list[str] = []
    for item in (report.findings or [])[:8]:
        sev = getattr(item.severity, "value", item.severity)
        where = f" at {item.element_selector}" if item.element_selector else ""
        parts.append(f"{sev} {item.rule_id}{where}: {item.title}.")
        if item.element_selector:
            rem.append(f"Resolve {item.rule_id} on {item.element_selector}.")
        else:
            rem.append(f"Resolve {item.rule_id}.")
    diagnosis = " ".join(parts) if parts else "Hydrogen produced a report with no extra narrative."
    remediation = " ".join(rem) if rem else "No ranked element fixes."
    return report.model_copy(
        update={
            "diagnosis": diagnosis,
            "remediation": remediation,
            "analyst": "hydrogen",
        }
    )
