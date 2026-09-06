import { useSyncExternalStore } from "react";
import {
  cancelJob,
  getHealth,
  getJob,
  type HydrogenReport,
  type JobSnapshot,
} from "@/lib/coherenceApi";

type AuditState = {
  jobId: string | null;
  url: string;
  status: JobSnapshot["status"] | "idle";
  stage: string;
  currentProfile: string;
  preview: string | null;
  previewBust: number;
  events: JobSnapshot["events"];
  report: HydrogenReport | null;
  warning: string | null;
  error: string | null;
  running: boolean;
  apiOnline: boolean | null;
};

const listeners = new Set<() => void>();
let state: AuditState = {
  jobId: null,
  url: "",
  status: "idle",
  stage: "",
  currentProfile: "",
  preview: null,
  previewBust: 0,
  events: [],
  report: null,
  warning: null,
  error: null,
  running: false,
  apiOnline: null,
};
let pollTimer: number | null = null;

function emit() {
  listeners.forEach((l) => l());
}

function setState(partial: Partial<AuditState>) {
  state = { ...state, ...partial };
  emit();
}

function applySnapshot(snap: JobSnapshot) {
  setState({
    jobId: snap.job_id,
    url: snap.url,
    status: snap.status,
    stage: snap.stage,
    currentProfile: snap.current_profile,
    preview: snap.preview,
    previewBust: snap.preview ? snap.events.length : 0,
    events: snap.events,
    report: snap.report,
    warning: snap.warning,
    error: snap.error,
    running: snap.status === "queued" || snap.status === "running",
  });
}

async function tick(jobId: string) {
  try {
    const snap = await getJob(jobId);
    applySnapshot(snap);
    if (snap.status === "done" || snap.status === "error") {
      stopPolling();
    }
  } catch {
    stopPolling();
    setState({ running: false, status: "error", error: "job not found" });
  }
}

function stopPolling() {
  if (pollTimer !== null) {
    window.clearInterval(pollTimer);
    pollTimer = null;
  }
}

export async function cancelAudit() {
  const jobId = state.jobId;
  if (!jobId) {
    setState({ running: false, status: "idle", error: null });
    stopPolling();
    return;
  }
  try {
    const snap = await cancelJob(jobId);
    if (snap.status === "done") {
      applySnapshot(snap);
    } else {
      applySnapshot({
        ...snap,
        status: "error",
        error: snap.error || "cancelled",
      });
    }
  } catch {
    setState({ running: false, status: "error", error: "cancelled" });
  }
  stopPolling();
  setState({ running: false });
}

export function watchJob(jobId: string) {
  stopPolling();
  setState({ jobId, running: true, status: "queued", error: null, report: null });
  void tick(jobId);
  pollTimer = window.setInterval(() => {
    void tick(jobId);
  }, 800);
}

export function useAuditStore<T = AuditState>(selector?: (s: AuditState) => T): T {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => (selector ? selector(state) : (state as unknown as T)),
    () => (selector ? selector(state) : (state as unknown as T)),
  );
}

useAuditStore.getState = () => state;

if (typeof window !== "undefined") {
  const flagged = window as Window & { __coherenceHealthPoll?: boolean };
  if (!flagged.__coherenceHealthPoll) {
    flagged.__coherenceHealthPoll = true;
    void (async function pollHealth() {
      setState({ apiOnline: await getHealth() });
      window.setInterval(async () => {
        setState({ apiOnline: await getHealth() });
      }, 4000);
    })();
  }
}
