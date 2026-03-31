import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
} from "react";
import {
  getPredictions,
  predict,
  type PredictionHistoryItem,
  type PredictionResponse,
} from "./api";

const APP_NAME = "SkinSight Triage";

const NOTE_TEMPLATES = [
  "55-year-old female with lesion on anterior torso. Slow change in pigmentation over the last 2 months. No fever. Mild itching reported.",
  "42-year-old male with irregular pigmented lesion on upper back. Recent increase in size. No bleeding. Family history of skin cancer.",
  "29-year-old female with small lesion on forearm. Stable appearance. No pain, ulceration, or discharge. Monitoring requested.",
];

const CLINICAL_HINTS = [
  "Where the lesion is located",
  "Whether it recently changed in size or color",
  "Pain, itching, bleeding, or discharge",
  "Relevant patient history or family history",
];

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function getRiskMeta(prediction?: string): {
  toneClass: string;
  badgeClass: string;
  title: string;
  summary: string;
} {
  const normalized = (prediction || "").toLowerCase();

  if (normalized.includes("high")) {
    return {
      toneClass: "tone-high",
      badgeClass: "badge-high",
      title: "Needs faster follow-up",
      summary:
        "The model sees stronger risk signals in this case. Use this as a prioritization aid and pair it with clinical judgment.",
    };
  }

  if (normalized.includes("medium")) {
    return {
      toneClass: "tone-medium",
      badgeClass: "badge-medium",
      title: "Worth a closer review",
      summary:
        "This case sits in the middle range and may benefit from additional review, better notes, or follow-up imaging.",
    };
  }

  return {
    toneClass: "tone-low",
    badgeClass: "badge-low",
    title: "Lower-priority triage signal",
    summary:
      "The current inputs suggest a lower-risk pattern, but this should still be interpreted in context by a clinician.",
  };
}

function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [image, setImage] = useState<File | null>(null);
  const [notes, setNotes] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [error, setError] = useState<string>("");
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [previewUrl, setPreviewUrl] = useState<string>("");

  async function loadHistory(): Promise<void> {
    try {
      const data = await getPredictions();
      setHistory(data.items || []);
    } catch (err) {
      console.error(err);
    }
  }

  useEffect(() => {
    document.title = APP_NAME;
    void loadHistory();
  }, []);

  useEffect(() => {
    if (!image) {
      setPreviewUrl("");
      return;
    }

    const nextUrl = URL.createObjectURL(image);
    setPreviewUrl(nextUrl);

    return () => {
      URL.revokeObjectURL(nextUrl);
    };
  }, [image]);

  const sortedProbabilities = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  }, [result]);

  const riskMeta = getRiskMeta(result?.predicted_class);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();

    if (!image) {
      setError("Please upload a lesion image before running triage.");
      return;
    }

    setError("");
    setLoading(true);

    try {
      const data = await predict(image, notes);
      setResult(data);
      await loadHistory();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Prediction failed.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  function handleFileSelection(file: File | null): void {
    if (!file) {
      setImage(null);
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Only image files are supported.");
      return;
    }

    setError("");
    setImage(file);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>): void {
    const file = event.target.files?.[0] || null;
    handleFileSelection(file);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>): void {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0] || null;
    handleFileSelection(file);
  }

  function resetCase(): void {
    setImage(null);
    setNotes("");
    setError("");
    setResult(null);
    setDragActive(false);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function reuseHistoryItem(item: PredictionHistoryItem): void {
    setNotes(item.notes || "");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="app-shell">
      <div className="animated-bg">
        <span className="orb orb-one" />
        <span className="orb orb-two" />
        <span className="orb orb-three" />
        <span className="grid-glow" />
      </div>

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">✚</div>
          <div>
            <p className="brand-subtitle">
              Smarter lesion review with image and clinical context
            </p>
            <h1>{APP_NAME}</h1>
          </div>
        </div>
      </header>

      <section className="hero-panel">
        <div>
          <p className="eyebrow">Image + notes + triage summary</p>
          <h2>A cleaner, more human way to review dermatology-style cases.</h2>
          <p className="hero-copy">
            Upload a lesion image, add a short clinical note, and get an
            easy-to-read triage summary with confidence, supporting signals,
            and recent case history.
          </p>
        </div>

        <div className="hero-stats">
          <div className="mini-stat">
            <span className="mini-stat-label">What you upload</span>
            <strong>Image + case notes</strong>
            <small>A lesion image with optional clinical context</small>
          </div>
          <div className="mini-stat">
            <span className="mini-stat-label">What you get</span>
            <strong>Clear triage summary</strong>
            <small>Readable risk level, confidence, and class breakdown</small>
          </div>
          <div className="mini-stat">
            <span className="mini-stat-label">Best for</span>
            <strong>Fast case review</strong>
            <small>Useful for demos, prototyping, and workflow storytelling</small>
          </div>
        </div>
      </section>

      <main className="dashboard-grid">
        <section className="panel intake-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Step 1</p>
              <h3>Case Intake</h3>
            </div>
            <button type="button" className="text-button" onClick={resetCase}>
              Reset case
            </button>
          </div>

          <form onSubmit={handleSubmit} className="intake-form">
            <label
              className={`upload-zone ${dragActive ? "drag-active" : ""} ${
                previewUrl ? "has-preview" : ""
              }`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                hidden
              />

              {previewUrl ? (
                <div className="upload-preview">
                  <img src={previewUrl} alt="Selected lesion preview" />
                  <div className="upload-copy">
                    <strong>{image?.name}</strong>
                    <span>Click to replace or drag in a new image</span>
                  </div>
                </div>
              ) : (
                <div className="upload-empty">
                  <div className="upload-icon">⬆</div>
                  <strong>Drop lesion image here</strong>
                  <span>or click to browse from your device</span>
                </div>
              )}
            </label>

            <div className="template-bar">
              <span>Quick note templates</span>
              <div className="template-chips">
                {NOTE_TEMPLATES.map((template, index) => (
                  <button
                    key={index}
                    type="button"
                    className="chip"
                    onClick={() => setNotes(template)}
                  >
                    Template {index + 1}
                  </button>
                ))}
              </div>
            </div>

            <div className="notes-block">
              <label htmlFor="notes">Clinical notes</label>
              <textarea
                id="notes"
                placeholder="Enter age, lesion location, symptoms, recent changes, and relevant history..."
                value={notes}
                onChange={(event) => setNotes(event.target.value)}
                rows={7}
              />
            </div>

            <div className="action-row">
              <button type="submit" className="primary-button" disabled={loading}>
                {loading ? "Analyzing case..." : "Analyze Case"}
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => setNotes("")}
              >
                Clear notes
              </button>
            </div>
          </form>

          {error ? <div className="alert error-alert">{error}</div> : null}

          <div className="support-card">
            <p className="panel-kicker">Helpful inputs</p>
            <h4>What makes a stronger request?</h4>
            <ul className="hint-list">
              {CLINICAL_HINTS.map((hint) => (
                <li key={hint}>{hint}</li>
              ))}
            </ul>
          </div>
        </section>

        <section className="panel result-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Step 2</p>
              <h3>Triage Summary</h3>
            </div>
          </div>

          {result ? (
            <div className="result-stack">
              <div className={`result-hero ${riskMeta.toneClass}`}>
                <div>
                  <span className={`risk-badge ${riskMeta.badgeClass}`}>
                    {result.predicted_class}
                  </span>
                  <h4>{riskMeta.title}</h4>
                  <p>{riskMeta.summary}</p>
                </div>

                <div className="confidence-ring">
                  <span>{formatPercent(result.confidence)}</span>
                  <small>confidence</small>
                </div>
              </div>

              <div className="metric-grid">
                <div className="metric-card">
                  <span>Latency</span>
                  <strong>{result.duration_ms} ms</strong>
                </div>
                <div className="metric-card">
                  <span>Model version</span>
                  <strong>{result.model_version}</strong>
                </div>
                <div className="metric-card">
                  <span>Request ID</span>
                  <strong className="mono">{result.request_id.slice(0, 8)}</strong>
                </div>
              </div>

              <div className="probability-card">
                <div className="subsection-header">
                  <h4>Risk distribution</h4>
                  <span>Class probabilities</span>
                </div>

                <div className="probability-list">
                  {sortedProbabilities.map(([label, value]) => (
                    <div key={label} className="probability-row">
                      <div className="probability-meta">
                        <span>{label}</span>
                        <span>{formatPercent(value)}</span>
                      </div>
                      <div className="probability-track">
                        <div
                          className="probability-fill"
                          style={{ width: `${Math.max(value * 100, 4)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="two-column-card">
                <div>
                  <div className="subsection-header">
                    <h4>Warnings</h4>
                    <span>Model output notes</span>
                  </div>

                  <ul className="warning-list">
                    {result.warnings.length > 0 ? (
                      result.warnings.map((warning, index) => (
                        <li key={index}>{warning}</li>
                      ))
                    ) : (
                      <li>No warning flags returned for this case.</li>
                    )}
                  </ul>
                </div>

                <div>
                  <div className="subsection-header">
                    <h4>Assets</h4>
                    <span>Traceable output</span>
                  </div>

                  <div className="asset-list">
                    {result.image_url ? (
                      <a
                        href={result.image_url}
                        target="_blank"
                        rel="noreferrer"
                        className="asset-link"
                      >
                        Open uploaded image
                      </a>
                    ) : (
                      <span className="muted">No image link returned</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-icon">◎</div>
              <h4>No prediction yet</h4>
              <p>
                Submit an image and supporting notes to generate a triage
                summary, confidence score, and probability breakdown.
              </p>

              <div className="empty-feature-grid">
                <div className="empty-feature">
                  <strong>Readable output</strong>
                  <span>Clear risk status with human-friendly explanation</span>
                </div>
                <div className="empty-feature">
                  <strong>Explainability</strong>
                  <span>Probability bars instead of raw JSON blocks</span>
                </div>
                <div className="empty-feature">
                  <strong>Reusable history</strong>
                  <span>Quick access to recent cases and saved notes</span>
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="panel workflow-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Step 3</p>
              <h3>How it works</h3>
            </div>
          </div>

          <div className="workflow-list">
            <div className="workflow-item">
              <span>01</span>
              <div>
                <strong>Upload a case image</strong>
                <p>Drag and drop an image or browse from your device.</p>
              </div>
            </div>
            <div className="workflow-item">
              <span>02</span>
              <div>
                <strong>Add short notes</strong>
                <p>Include location, symptoms, history, or recent change.</p>
              </div>
            </div>
            <div className="workflow-item">
              <span>03</span>
              <div>
                <strong>Review the triage summary</strong>
                <p>See the risk class, confidence, and supporting breakdown.</p>
              </div>
            </div>
          </div>
        </section>

        <section className="panel history-panel">
          <div className="panel-header">
            <div>
              <p className="panel-kicker">Recent activity</p>
              <h3>Prediction History</h3>
            </div>
            <button
              type="button"
              className="text-button"
              onClick={() => void loadHistory()}
            >
              Refresh
            </button>
          </div>

          {history.length > 0 ? (
            <div className="history-grid">
              {history.map((item) => (
                <article key={item.request_id} className="history-card">
                  <div className="history-top">
                    <span
                      className={`risk-badge ${
                        getRiskMeta(item.predicted_class).badgeClass
                      }`}
                    >
                      {item.predicted_class}
                    </span>
                    <span className="history-time">
                      {formatTimestamp(item.timestamp)}
                    </span>
                  </div>

                  <p className="history-confidence">
                    Confidence: {formatPercent(item.confidence)}
                  </p>

                  <p className="history-notes">
                    {item.notes?.trim()
                      ? item.notes
                      : "No clinical notes saved for this case."}
                  </p>

                  <div className="history-actions">
                    <button
                      type="button"
                      className="ghost-chip"
                      onClick={() => reuseHistoryItem(item)}
                    >
                      Reuse notes
                    </button>

                    {item.image_url ? (
                      <a
                        href={item.image_url}
                        target="_blank"
                        rel="noreferrer"
                        className="ghost-chip link-chip"
                      >
                        Image
                      </a>
                    ) : null}
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <div className="empty-history">
              <p>No history yet. Run your first case to populate the activity feed.</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;