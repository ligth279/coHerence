export type JobSnapshot = {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  n_trials: number;
  url: string;
  error: string | null;
  warning: string | null;
  stage: string;
  current_profile: string;
  preview: string | null;
  events: Array<Record<string, unknown>>;
  report: HydrogenReport | null;
};

export type HydrogenReport = {
  report_id: string;
  target_url: string;
  overall_fairness_score: number | null;
  score_status: string;
  scoring_policy: string;
  profiles_tested: string[];
  diagnosis: string;
  remediation: string;
  analyst: string;
  findings: Array<{
    id: string;
    title: string;
    severity: string;
    element_selector: string;
    rule_id: string;
  }>;
  breakdown?: {
    bottleneck_group?: string;
    bottleneck_metric?: string;
  };
};

export type StartJobBody = {
  url: string;
  n_trials?: number;
  success_selector?: string;
  steps?: string[];
  goal?: string;
  profile_ids?: string[];
  diagnose?: boolean;
  plan_once?: boolean;
};

const jsonHeaders = { "Content-Type": "application/json" };

const HOST_ALIASES: Record<string, string> = {
  wiki: "https://en.wikipedia.org",
  wikipedia: "https://en.wikipedia.org",
  gmail: "https://mail.google.com",
  "gmail.com": "https://mail.google.com",
  "www.gmail.com": "https://mail.google.com",
  github: "https://github.com",
  youtube: "https://www.youtube.com",
  netflix: "https://www.netflix.com/browse",
  mdn: "https://developer.mozilla.org",
};

export function canonicalizeSiteUrl(raw: string): string {
  const trimmed = (raw || "").trim();
  if (!trimmed) return trimmed;
  if (trimmed.startsWith("/")) {
    return new URL(trimmed, window.location.origin).href;
  }
  const candidate = /^https?:\/\//i.test(trimmed)
    ? trimmed
    : `https://${trimmed}`;
  try {
    const parsed = new URL(candidate);
    const alias = HOST_ALIASES[parsed.hostname.toLowerCase()];
    return alias || parsed.href;
  } catch {
    return candidate;
  }
}

function errorDetail(text: string, status: number): string {
  try {
    const payload = JSON.parse(text) as { detail?: unknown };
    if (typeof payload.detail === "string") return payload.detail;
  } catch {
    /* not JSON */
  }
  return text || `job failed (${status})`;
}

export async function startJob(body: StartJobBody): Promise<JobSnapshot> {
  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ ...body, url: canonicalizeSiteUrl(body.url) }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(errorDetail(detail, response.status));
  }
  return response.json();
}

export type StartScreenshotJobBody = {
  url: string;
  images: string[];
  diagnose?: boolean;
};

export async function startScreenshotJob(
  body: StartScreenshotJobBody,
): Promise<JobSnapshot> {
  const response = await fetch("/api/jobs/screenshots", {
    method: "POST",
    headers: jsonHeaders,
    body: JSON.stringify({ ...body, url: canonicalizeSiteUrl(body.url) }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(errorDetail(detail, response.status));
  }
  return response.json();
}

export function readImagesAsDataUrls(files: File[]): Promise<string[]> {
  return Promise.all(
    files.map(
      (file) =>
        new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onerror = () => reject(new Error(`could not read ${file.name}`));
          reader.onload = () => resolve(String(reader.result));
          reader.readAsDataURL(file);
        }),
    ),
  );
}

/** A success selector has to be able to be false before the task and true after. */
export function isUsableSuccessSelector(raw: string): {
  ok: boolean;
  reason: string;
} {
  const selector = (raw || "").trim();
  if (!selector) {
    return { ok: false, reason: "A success selector is required." };
  }
  if (/^(body|html|:root|\*)$/i.test(selector)) {
    return {
      ok: false,
      reason: `"${selector}" is visible before the task starts, so every profile would be scored as complete.`,
    };
  }
  try {
    document.createDocumentFragment().querySelector(selector);
  } catch {
    return { ok: false, reason: "That is not a valid CSS selector." };
  }
  return { ok: true, reason: "" };
}

export async function getHealth(): Promise<boolean> {
  try {
    const response = await fetch("/api/health", {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return false;
    const payload = (await response.json()) as { ok?: boolean };
    return payload.ok === true;
  } catch {
    return false;
  }
}

export async function cancelJob(jobId: string): Promise<JobSnapshot> {
  const response = await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" });
  if (!response.ok) {
    throw new Error("could not cancel");
  }
  return response.json();
}

export async function getJob(jobId: string): Promise<JobSnapshot> {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error("job not found");
  }
  return response.json();
}

export function previewUrl(jobId: string, bust: number): string {
  return `/api/jobs/${jobId}/preview?t=${bust}`;
}

export const DEMO_CHECKOUT_PATH = "/demo/checkout.html";
export const DEMO_STEPS = ["#fake-button", "#submit-order"];
export const DEMO_SUCCESS = "#order-confirmed";
export const DEMO_GOAL = "Place the order";
export const DEFAULT_GOAL =
  "Find the main action or article on this page and open it";
export const GOAL_PLACEHOLDER = DEFAULT_GOAL;
export const SUCCESS_PLACEHOLDER = "#order-confirmed";
export const DEMO_PROFILES = [
  "baseline_default",
  "motor_impaired",
  "keyboard_only",
];

export function isDemoCheckout(url: string): boolean {
  try {
    return new URL(url, window.location.origin).pathname.endsWith(
      DEMO_CHECKOUT_PATH,
    );
  } catch {
    return url.includes("demo/checkout");
  }
}
