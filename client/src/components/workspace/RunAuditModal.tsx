import { useEffect, useMemo, useState } from "react";
import { Images, Sparkles, X } from "lucide-react";
import { toast } from "sonner";
import {
  DEFAULT_GOAL,
  DEMO_GOAL,
  DEMO_PROFILES,
  DEMO_STEPS,
  DEMO_SUCCESS,
  GOAL_PLACEHOLDER,
  SUCCESS_PLACEHOLDER,
  cancelJob,
  isDemoCheckout,
  isUsableSuccessSelector,
  readImagesAsDataUrls,
  startJob,
  startScreenshotJob,
} from "@/lib/coherenceApi";
import { cancelAudit, useAuditStore, watchJob } from "@/stores/auditStore";

type Props = {
  isOpen: boolean;
  onClose: () => void;
  url: string;
};

type Mode = "auto" | "screenshots";

export default function RunAuditModal({ isOpen, onClose, url }: Props) {
  const demo = useMemo(() => isDemoCheckout(url), [url]);
  const [mode, setMode] = useState<Mode>("auto");
  const [goal, setGoal] = useState(demo ? DEMO_GOAL : DEFAULT_GOAL);
  const [success, setSuccess] = useState(demo ? DEMO_SUCCESS : "");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const running = useAuditStore((s) => s.running);
  const jobId = useAuditStore((s) => s.jobId);

  useEffect(() => {
    if (!isOpen) return;
    setMode("auto");
    setGoal(demo ? DEMO_GOAL : DEFAULT_GOAL);
    setSuccess(demo ? DEMO_SUCCESS : "");
    setFiles([]);
    setError("");
    setBusy(false);
  }, [isOpen, demo, url]);

  if (!isOpen) return null;

  const selectorState = isUsableSuccessSelector(success);
  const typedGoal = goal.trim() || (!selectorState.ok ? success.trim() : "");
  const typedSelector = selectorState.ok ? success.trim() : "";
  const ready =
    mode === "screenshots"
      ? files.length > 0
      : demo || typedGoal.length > 0;

  const startCapture = async () => {
    const body = demo
      ? {
          url,
          success_selector: DEMO_SUCCESS,
          steps: DEMO_STEPS,
          profile_ids: DEMO_PROFILES,
          n_trials: 1,
          diagnose: true,
        }
      : {
          url,
          success_selector: typedSelector,
          goal: typedGoal || DEFAULT_GOAL,
          plan_once: true,
          profile_ids: DEMO_PROFILES,
          n_trials: 1,
          diagnose: true,
        };
    for (let attempt = 0; attempt < 20; attempt += 1) {
      try {
        return await startJob(body);
      } catch (first) {
        const message = first instanceof Error ? first.message : "";
        const stuck = message.match(/already running \(([^)]+)\)/);
        if (!stuck) throw first;
        await cancelJob(stuck[1]);
        await new Promise((resolve) => window.setTimeout(resolve, 200));
      }
    }
    throw new Error("capture is still shutting down; try again");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      if (running && jobId) {
        await cancelAudit();
      }
      const snap =
        mode === "screenshots"
          ? await startScreenshotJob({
              url,
              images: await readImagesAsDataUrls(files),
              diagnose: true,
            })
          : await startCapture();
      watchJob(snap.job_id);
      toast.success(
        mode === "screenshots"
          ? `Reading ${files.length} screenshot${files.length === 1 ? "" : "s"}`
          : demo
            ? "Playwright is capturing the demo checkout"
            : "Playwright is opening this site",
      );
      onClose();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not start job";
      setError(message);
      toast.error(message);
      setBusy(false);
    }
  };

  return (
    <div className="ws-modal-backdrop" onClick={onClose}>
      <div
        className="ws-modal-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-labelledby="audit-title"
      >
        <div className="ws-modal-header">
          <div className="ws-modal-title-wrap">
            <span className="ws-modal-icon-badge">
              <Sparkles size={18} />
            </span>
            <div>
              <h2 id="audit-title" className="ws-modal-title">
                Run fairness audit
              </h2>
              <p className="ws-modal-subtitle">
                Playwright drives each profile. Hydrogen scores. Helium
                (Qwen 27B) writes the diagnosis.
              </p>
            </div>
          </div>
          <button
            type="button"
            className="ws-modal-close-btn"
            onClick={onClose}
            aria-label="Close"
          >
            <X size={18} />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="ws-modal-body">
          <div className="ws-modal-field">
            <label className="ws-modal-label">Target</label>
            <input className="ws-modal-input plain" value={url} readOnly />
          </div>

          <div className="ws-modal-field">
            <label className="ws-modal-label">Capture</label>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <button
                type="button"
                className={`ws-modal-preset-chip ${mode === "auto" ? "is-active" : ""}`}
                onClick={() => setMode("auto")}
              >
                Automatic
              </button>
              <button
                type="button"
                className={`ws-modal-preset-chip ${mode === "screenshots" ? "is-active" : ""}`}
                onClick={() => setMode("screenshots")}
              >
                My screenshots
              </button>
            </div>
          </div>

          {mode === "screenshots" ? (
            <>
              <p className="ws-modal-hint">
                Visit the pages yourself, screenshot each one, and drop them
                here in order. No browser runs, so there is no DOM behind these
                images: contrast, touch-target and WCAG rules are skipped and
                the score comes back INSUFFICIENT_EVIDENCE. You get the vision
                read of each view plus Helium&apos;s diagnosis.
              </p>
              <div className="ws-modal-field">
                <label className="ws-modal-label">
                  Screenshots (PNG or JPEG, up to 12)
                </label>
                <input
                  className="ws-modal-input plain"
                  type="file"
                  accept="image/png,image/jpeg"
                  multiple
                  onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
                />
              </div>
              {files.length ? (
                <p className="ws-modal-hint">
                  {files.map((file) => file.name).join(", ")}
                </p>
              ) : null}
            </>
          ) : demo ? (
            <p className="ws-modal-hint">
              Playwright clicks Place order → Pay now (you will see each
              frame). Helium writes the report after. Wikipedia-style sites
              wait on Nitrogen for each click.
            </p>
          ) : (
            <>
              <p className="ws-modal-hint">
                A goal is already filled in. Click Start capture. Change the
                sentence only if you want a different task. CSS selector is
                optional.
              </p>
              <div className="ws-modal-field">
                <label className="ws-modal-label">Task goal</label>
                <input
                  className="ws-modal-input plain"
                  placeholder={GOAL_PLACEHOLDER}
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                />
              </div>
              <div className="ws-modal-field">
                <label className="ws-modal-label">
                  Success selector (optional)
                </label>
                <input
                  className="ws-modal-input plain"
                  placeholder={SUCCESS_PLACEHOLDER}
                  value={success}
                  onChange={(e) => setSuccess(e.target.value)}
                />
              </div>
            </>
          )}

          {error ? (
            <p className="ws-modal-hint" style={{ color: "#E8A598" }}>
              {error}
            </p>
          ) : null}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button
              type="submit"
              className="ws-toolbar-audit-btn"
              disabled={busy || (mode === "screenshots" && !ready)}
            >
              {mode === "screenshots" ? (
                <Images size={13} />
              ) : (
                <Sparkles size={13} />
              )}
              <span>
                {busy
                  ? "Starting…"
                  : mode === "screenshots"
                    ? "Read screenshots"
                    : running
                      ? "Restart capture"
                      : "Start capture"}
              </span>
            </button>
            {running ? (
              <button
                type="button"
                className="ws-toolbar-audit-btn"
                onClick={() => void cancelAudit()}
              >
                <span>Cancel</span>
              </button>
            ) : null}
          </div>
        </form>
      </div>
    </div>
  );
}
