import { useRef, useState, useCallback, type PointerEvent } from "react";
import {
  useWorkspaceStore,
  type WorkspaceElement,
} from "@/stores/workspaceStore";
import {
  ImagePlus,
  Globe,
  RotateCw,
  ExternalLink,
  MousePointerClick,
  Hand,
  ShieldCheck,
  ShieldAlert,
  Tv,
  LayoutGrid,
  Play,
  Film,
  Check,
} from "lucide-react";
import { resolveEmbed, type EmbedInfo } from "@/lib/embedHelper";
import { canonicalizeSiteUrl, previewUrl } from "@/lib/coherenceApi";
import { useAuditStore } from "@/stores/auditStore";

function sameSite(a: string, b: string): boolean {
  const origin =
    typeof window !== "undefined" ? window.location.origin : "http://localhost";
  try {
    const left = new URL(canonicalizeSiteUrl(a) || a, origin);
    const right = new URL(canonicalizeSiteUrl(b) || b, origin);
    const path = (value: string) => value.replace(/\/+$/, "") || "/";
    return left.origin === right.origin && path(left.pathname) === path(right.pathname);
  } catch {
    return a.replace(/\/+$/, "") === b.replace(/\/+$/, "");
  }
}

export default function ElementRenderer({
  element,
  zIndex,
}: {
  element: WorkspaceElement;
  zIndex?: number;
}) {
  const {
    selectedIds,
    hoveredId,
    activeTool,
    canvas,
    elements,
    selectElement,
    setHoveredId,
    updateElement,
    updateElements,
    pushHistory,
  } = useWorkspaceStore();

  const isSelected = selectedIds.includes(element.id);
  const isHovered = hoveredId === element.id;
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{
    startX: number;
    startY: number;
    initialPositions: { id: string; x: number; y: number }[];
  } | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const textRef = useRef<HTMLDivElement>(null);

  const handlePointerDown = useCallback(
    (e: PointerEvent) => {
      if (activeTool !== "select" || element.locked) return;
      e.stopPropagation();
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);

      const isAlreadySelected = selectedIds.includes(element.id);
      if (!isAlreadySelected) {
        selectElement(element.id, e.shiftKey);
      }
      pushHistory();

      const targets = isAlreadySelected && selectedIds.length > 1 && !e.shiftKey
        ? elements.filter((el) => selectedIds.includes(el.id) && !el.locked)
        : [element];

      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        initialPositions: targets.map((t) => ({ id: t.id, x: t.x, y: t.y })),
      };
      setIsDragging(true);
    },
    [activeTool, element, selectedIds, elements, selectElement, pushHistory],
  );

  const handlePointerMove = useCallback(
    (e: PointerEvent) => {
      if (!isDragging || !dragRef.current) return;
      const dx = (e.clientX - dragRef.current.startX) / canvas.zoom;
      const dy = (e.clientY - dragRef.current.startY) / canvas.zoom;
      if (dragRef.current.initialPositions.length > 1) {
        updateElements(
          dragRef.current.initialPositions.map((p) => ({
            id: p.id,
            changes: { x: Math.round(p.x + dx), y: Math.round(p.y + dy) },
          })),
        );
      } else if (dragRef.current.initialPositions.length === 1) {
        const p = dragRef.current.initialPositions[0];
        updateElement(p.id, {
          x: Math.round(p.x + dx),
          y: Math.round(p.y + dy),
        });
      }
    },
    [isDragging, canvas.zoom, updateElement, updateElements],
  );

  const handlePointerUp = useCallback(() => {
    setIsDragging(false);
    dragRef.current = null;
  }, []);

  const handleDoubleClick = useCallback(() => {
    if (element.type === "text") {
      setIsEditing(true);
      requestAnimationFrame(() => {
        if (textRef.current) {
          textRef.current.focus();
          const sel = window.getSelection();
          const range = document.createRange();
          range.selectNodeContents(textRef.current);
          sel?.removeAllRanges();
          sel?.addRange(range);
        }
      });
    }
  }, [element.type]);

  const handleTextBlur = useCallback(() => {
    setIsEditing(false);
    if (textRef.current) {
      updateElement(element.id, { text: textRef.current.textContent ?? "" });
    }
  }, [element.id, updateElement]);

  const children = elements.filter(
    (e) => e.parentId === element.id && e.visible,
  );

  const selClass = isSelected ? " ws-element--selected" : "";
  const hovClass = isHovered && !isSelected ? " ws-element--hovered" : "";
  const lockClass = element.locked ? " ws-element--locked" : "";

  const wrapperStyle: React.CSSProperties = {
    left: element.x,
    top: element.y,
    width: element.width,
    height: element.height,
    transform: element.rotation ? `rotate(${element.rotation}deg)` : undefined,
    opacity: element.opacity,
    zIndex: zIndex ?? undefined,
  };

  const renderContent = () => {
    switch (element.type) {
      case "rectangle":
        return (
          <div
            style={{
              width: "100%",
              height: "100%",
              background: element.fill,
              opacity: element.fillOpacity,
              borderRadius: element.cornerRadius,
              border:
                element.strokeWidth > 0
                  ? `${element.strokeWidth}px solid ${element.stroke}`
                  : undefined,
            }}
          />
        );

      case "ellipse":
        return (
          <div
            style={{
              width: "100%",
              height: "100%",
              background: element.fill,
              opacity: element.fillOpacity,
              borderRadius: "50%",
              border:
                element.strokeWidth > 0
                  ? `${element.strokeWidth}px solid ${element.stroke}`
                  : undefined,
            }}
          />
        );

      case "text":
        return (
          <div
            ref={textRef}
            className="ws-text-content"
            contentEditable={isEditing}
            suppressContentEditableWarning
            onBlur={handleTextBlur}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                setIsEditing(false);
                (e.target as HTMLElement).blur();
              }
            }}
            style={{
              fontSize: element.fontSize,
              fontFamily: element.fontFamily,
              fontWeight: element.fontWeight,
              textAlign: element.textAlign,
              lineHeight: element.lineHeight,
              letterSpacing: element.letterSpacing
                ? `${element.letterSpacing}px`
                : undefined,
              color: element.textColor ?? "#173B36",
              cursor: isEditing ? "text" : undefined,
            }}
          >
            {element.text}
          </div>
        );

      case "image":
        return element.src ? (
          <img
            src={element.src}
            alt={element.name}
            draggable={false}
            style={{
              width: "100%",
              height: "100%",
              objectFit: element.objectFit ?? "cover",
              borderRadius: element.cornerRadius,
            }}
          />
        ) : (
          <div className="ws-image-placeholder">
            <ImagePlus size={32} />
          </div>
        );

      case "website":
        return <WebsiteElementView element={element} />;

      case "frame":
        return (
          <>
            <span className="ws-frame-label">{element.name}</span>
            <div
              style={{
                width: "100%",
                height: "100%",
                background: element.fill,
                opacity: element.fillOpacity,
                borderRadius: element.cornerRadius,
                position: "relative",
                overflow: "hidden",
              }}
            >
              {children.map((child, childIdx) => (
                <ElementRenderer
                  key={child.id}
                  element={child}
                  zIndex={childIdx + 1}
                />
              ))}
            </div>
          </>
        );

      case "line":
        return (
          <svg
            width={element.width || 1}
            height={Math.max(element.height, 2)}
            style={{ overflow: "visible", position: "absolute", top: 0, left: 0 }}
          >
            {element.endArrow && (
              <defs>
                <marker
                  id={`arrow-${element.id}`}
                  markerWidth="8"
                  markerHeight="6"
                  refX="8"
                  refY="3"
                  orient="auto"
                >
                  <path d="M0,0 L8,3 L0,6 Z" fill={element.stroke} />
                </marker>
              </defs>
            )}
            <line
              x1={0}
              y1={element.height / 2 || 1}
              x2={element.x2 ?? element.width}
              y2={(element.y2 ?? 0) + (element.height / 2 || 1)}
              stroke={element.stroke}
              strokeWidth={element.strokeWidth || 2}
              markerEnd={
                element.endArrow ? `url(#arrow-${element.id})` : undefined
              }
            />
          </svg>
        );
    }
  };

  return (
    <div
      className={`ws-element${selClass}${hovClass}${lockClass}`}
      style={wrapperStyle}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerEnter={() => setHoveredId(element.id)}
      onPointerLeave={() => setHoveredId(null)}
      onDoubleClick={handleDoubleClick}
      data-element-id={element.id}
    >
      {renderContent()}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Website Smart Card (For Netflix, protected media, and card mode)
// ---------------------------------------------------------------------------
function WebsiteSmartCard({
  element,
  embedInfo,
  onSwitchToLive,
}: {
  element: WorkspaceElement;
  embedInfo: EmbedInfo;
  onSwitchToLive: () => void;
}) {
  const isNetflix = embedInfo.platform === "netflix";
  const title =
    element.name && element.name !== "Website"
      ? element.name
      : embedInfo.title;

  return (
    <div className={`ws-website-smart-card ${embedInfo.platform}`}>
      <div className="ws-smart-card-content">
        <div className="ws-smart-card-badge-row">
          {isNetflix ? (
            <div className="ws-netflix-pill">
              <span className="ws-netflix-n-logo">N</span>
              <span className="ws-netflix-label">NETFLIX</span>
            </div>
          ) : (
            <div className="ws-generic-pill">
              {embedInfo.faviconUrl && (
                <img
                  src={embedInfo.faviconUrl}
                  alt=""
                  className="ws-smart-card-icon"
                  onError={(e) => {
                    (e.currentTarget as HTMLElement).style.display = "none";
                  }}
                />
              )}
              <span>{embedInfo.platformName}</span>
            </div>
          )}
          <span className="ws-smart-card-tag">
            {isNetflix ? "STREAMING MEDIA" : "WEB RESOURCE"}
          </span>
        </div>

        <div className="ws-smart-card-hero">
          <div className={`ws-smart-card-media-icon ${embedInfo.platform}`}>
            {isNetflix ? (
              <Film size={28} className="ws-smart-card-icon-svg" />
            ) : (
              <Globe size={28} className="ws-smart-card-icon-svg" />
            )}
          </div>
          <h3 className="ws-smart-card-title">{title}</h3>
          <p className="ws-smart-card-url" title={element.url}>
            {element.url}
          </p>
          <p className="ws-smart-card-desc">
            {isNetflix
              ? "Netflix streaming is protected by Widevine DRM and session security. Open directly in your browser with your active Netflix account."
              : "This resource is embedded on your canvas. Click below to launch in your browser or switch to live view."}
          </p>
        </div>

        <div className="ws-smart-card-actions">
          {element.url && (
            <a
              href={element.url}
              target="_blank"
              rel="noopener noreferrer"
              className={`ws-smart-card-primary-btn ${embedInfo.platform}`}
              onClick={(e) => e.stopPropagation()}
            >
              {isNetflix ? (
                <>
                  <Play size={13} fill="currentColor" /> Watch on Netflix
                </>
              ) : (
                <>
                  <ExternalLink size={13} /> Open in New Window
                </>
              )}
            </a>
          )}
          <button
            type="button"
            className="ws-smart-card-secondary-btn"
            onClick={(e) => {
              e.stopPropagation();
              onSwitchToLive();
            }}
            title="Preview page with CORS and header bypass"
          >
            <Tv size={12} /> Try Live Browser View
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Website Element View (Live Browser or Smart Card)
// ---------------------------------------------------------------------------
function WebsiteElementView({ element }: { element: WorkspaceElement }) {
  const { updateElement, deleteElements } = useWorkspaceStore();
  const audit = useAuditStore();
  const [isEditingUrl, setIsEditingUrl] = useState(false);
  const [urlDraft, setUrlDraft] = useState(element.url || "");
  const [reloadKey, setReloadKey] = useState(0);
  const jobForThis =
    !!audit.jobId &&
    !!element.url &&
    !!audit.url &&
    sameSite(audit.url, element.url);
  const lastEvent = audit.events.at(-1);
  const waitingOnVl =
    jobForThis &&
    audit.status === "running" &&
    (audit.stage === "vl_wait" ||
      (typeof lastEvent?.stage === "string" && lastEvent.stage === "vl_wait"));
  const captureFailed = jobForThis && audit.status === "error";
  const agentView =
    jobForThis && !!audit.preview && audit.status !== "error";

  const isInteractive = !!element.isInteractive;
  const useProxy = element.useProxy !== false;
  const embedInfo = resolveEmbed(element.url, useProxy);
  const isProtected = embedInfo.isStreamingProtected;
  const viewMode = element.viewMode || (isProtected ? "card" : "live");

  const handleUrlSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = urlDraft.trim();
    const finalUrl = canonicalizeSiteUrl(trimmed);
    if (finalUrl && finalUrl !== element.url) {
      const newInfo = resolveEmbed(finalUrl, useProxy);
      updateElement(element.id, {
        url: finalUrl,
        name: newInfo.title || element.name,
        viewMode: newInfo.isStreamingProtected ? "card" : "live",
      });
    }
    setIsEditingUrl(false);
  };

  return (
    <div
      className="ws-website-frame"
      style={{
        width: "100%",
        height: "100%",
        borderRadius: element.cornerRadius ?? 8,
        border:
          element.strokeWidth > 0
            ? `${element.strokeWidth}px solid ${element.stroke}`
            : "1px solid rgba(255,255,255,0.12)",
      }}
    >
      <div
        className="ws-website-header"
        onPointerDown={(e) => e.stopPropagation()}
      >
        <div className="ws-website-traffic-lights">
          <span
            className="ws-traffic-dot close"
            title="Delete website window"
            onClick={(e) => {
              e.stopPropagation();
              deleteElements([element.id]);
            }}
          />
          <span className="ws-traffic-dot min" />
          <span
            className="ws-traffic-dot max"
            title="Reset window size (680x440)"
            onClick={(e) => {
              e.stopPropagation();
              updateElement(element.id, { width: 680, height: 440 });
            }}
          />
        </div>

        <div className="ws-website-url-bar">
          {embedInfo.faviconUrl ? (
            <img
              src={embedInfo.faviconUrl}
              alt=""
              className="ws-website-favicon"
              onError={(e) => {
                (e.currentTarget as HTMLElement).style.display = "none";
              }}
            />
          ) : (
            <Globe size={11} className="ws-website-globe-icon" />
          )}

          {isEditingUrl ? (
            <form onSubmit={handleUrlSubmit} className="ws-website-url-form">
              <input
                type="text"
                autoFocus
                className="ws-website-url-input"
                value={urlDraft}
                onChange={(e) => setUrlDraft(e.target.value)}
                onBlur={handleUrlSubmit}
                onKeyDown={(e) => {
                  if (e.key === "Escape") {
                    setUrlDraft(element.url || "");
                    setIsEditingUrl(false);
                  }
                }}
              />
              <button
                type="submit"
                className="ws-website-url-submit-btn"
                title="Save URL"
              >
                <Check size={11} />
              </button>
            </form>
          ) : (
            <div
              className="ws-website-url-display"
              title="Click to edit URL"
              onClick={() => {
                setUrlDraft(element.url || "");
                setIsEditingUrl(true);
              }}
            >
              <span className="ws-website-url-text">
                {element.url || "https://..."}
              </span>
              {embedInfo.platformName && embedInfo.platformName !== "Website" && (
                <span className={`ws-platform-badge ${embedInfo.platform}`}>
                  {embedInfo.platformName}
                </span>
              )}
            </div>
          )}
        </div>

        <div className="ws-website-actions">
          {/* Card / Live Toggle */}
          <button
            type="button"
            className={`ws-website-action-btn ${viewMode === "card" ? "is-active" : ""}`}
            title={
              viewMode === "card"
                ? "Card view active (click for live browser iframe)"
                : "Live browser view active (click for smart card view)"
            }
            onClick={(e) => {
              e.stopPropagation();
              updateElement(element.id, {
                viewMode: viewMode === "card" ? "live" : "card",
              });
            }}
          >
            {viewMode === "card" ? <Tv size={12} /> : <LayoutGrid size={12} />}
          </button>

          {/* Proxy Toggle for general web */}
          {!embedInfo.canDirectIframe && (
            <button
              type="button"
              className={`ws-website-action-btn ${useProxy ? "is-proxy-on" : "is-proxy-off"}`}
              title={
                useProxy
                  ? "Bypass Proxy ACTIVE (removes X-Frame-Options/CSP restrictions). Click to switch to direct URL."
                  : "Direct mode (Proxy OFF). Click to enable proxy bypass."
              }
              onClick={(e) => {
                e.stopPropagation();
                updateElement(element.id, { useProxy: !useProxy });
              }}
            >
              {useProxy ? <ShieldCheck size={12} /> : <ShieldAlert size={12} />}
            </button>
          )}

          {/* Interactive Mode Toggle */}
          <button
            type="button"
            className={`ws-website-action-btn ${isInteractive ? "is-active" : ""}`}
            title={
              isInteractive
                ? "Interactive mode ON (click/scroll page). Click to enable dragging."
                : "Interactive mode OFF (drag canvas enabled). Click to interact with website."
            }
            onClick={(e) => {
              e.stopPropagation();
              updateElement(element.id, { isInteractive: !isInteractive });
            }}
          >
            {isInteractive ? <MousePointerClick size={12} /> : <Hand size={12} />}
          </button>

          {/* Reload Button */}
          <button
            type="button"
            className="ws-website-action-btn"
            title="Reload website"
            onClick={(e) => {
              e.stopPropagation();
              setReloadKey((k) => k + 1);
            }}
          >
            <RotateCw size={12} />
          </button>

          {/* Open in New Window */}
          {element.url && (
            <a
              href={element.url}
              target="_blank"
              rel="noopener noreferrer"
              className="ws-website-action-btn"
              title="Open site in new browser tab"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLink size={12} />
            </a>
          )}
        </div>
      </div>

      <div className="ws-website-viewport">
        {viewMode === "card" ? (
          <WebsiteSmartCard
            element={element}
            embedInfo={embedInfo}
            onSwitchToLive={() =>
              updateElement(element.id, { viewMode: "live" })
            }
          />
        ) : element.url ? (
          <>
            {agentView ? (
              <img
                className="ws-website-iframe"
                alt={`${audit.currentProfile || "agent"} view`}
                src={previewUrl(audit.jobId as string, audit.previewBust)}
              />
            ) : (
              <iframe
                key={reloadKey}
                id={`iframe-${element.id}`}
                src={embedInfo.embedUrl}
                title={element.name || "Embedded website"}
                className="ws-website-iframe"
                sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-presentation"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                style={{
                  pointerEvents: isInteractive ? "auto" : "none",
                }}
              />
            )}
            {agentView ? (
              <div className="ws-website-agent-label">
                Playwright
                {audit.currentProfile
                  ? ` · ${audit.currentProfile.replaceAll("_", " ")}`
                  : ""}
                {waitingOnVl
                  ? " · waiting on Nitrogen"
                  : typeof lastEvent?.selector === "string"
                    ? ` → ${String(lastEvent.selector)}`
                    : audit.stage === "diagnose"
                      ? " · Helium"
                      : ""}
              </div>
            ) : null}
            {waitingOnVl ? (
              <div className="ws-website-agent-banner">
                Chromium is paused. Nitrogen (Qwen VL) is choosing the next
                click on the B300 — frames update after it returns.
              </div>
            ) : null}
            {captureFailed ? (
              <div className="ws-website-agent-banner is-error">
                Playwright did not capture this URL.
                {audit.error ? ` ${audit.error}` : ""}
              </div>
            ) : null}
            {jobForThis && !captureFailed && audit.warning ? (
              <div className="ws-website-agent-banner">{audit.warning}</div>
            ) : null}
            {jobForThis && audit.status === "running" && !audit.preview ? (
              <div className="ws-website-agent-banner">
                Playwright is opening Chromium. The live iframe is not the
                capture — clicks and screenshots appear here once the first
                frame lands.
              </div>
            ) : null}
            {isProtected && !agentView && (
              <div
                className="ws-website-drm-helper"
                onPointerDown={(e) => e.stopPropagation()}
              >
                <span>Netflix requires browser player</span>
                <a
                  href={element.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ws-website-drm-btn"
                  onClick={(e) => e.stopPropagation()}
                >
                  <Play size={10} fill="currentColor" /> Watch
                </a>
                <button
                  type="button"
                  className="ws-website-drm-toggle"
                  onClick={(e) => {
                    e.stopPropagation();
                    updateElement(element.id, { viewMode: "card" });
                  }}
                >
                  Card view
                </button>
              </div>
            )}
            {!isInteractive && !agentView && (
              <div
                className="ws-website-drag-overlay"
                title="Click the Hand/Pointer button in toolbar to interact with page"
              />
            )}
          </>
        ) : (
          <div className="ws-website-placeholder">
            <Globe size={28} />
            <span>No URL provided</span>
          </div>
        )}
      </div>
    </div>
  );
}
