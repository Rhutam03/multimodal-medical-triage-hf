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

const APP_NAME = "Multimodal Dermatology Triage";

const NOTE_TEMPLATES: string[] = [
  "55-year-old female with lesion on anterior torso. Slow change in pigmentation over the last 2 months. Mild itching reported. No fever.",
  "42-year-old male with irregular pigmented lesion on upper back. Recent increase in size. No bleeding. Family history of skin cancer.",
  "29-year-old female with small lesion on forearm. Stable appearance. No pain, ulceration, or discharge. Monitoring requested.",
];

const CLINICAL_HINTS: string[] = [
  "Where the lesion is located",
  "Whether it recently changed in size or color",
  "Pain, itching, bleeding, or discharge",
  "Relevant patient history or family history",
];

const RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"] as const;

function formatPercent(value: number | undefined, capAt = 99.9): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "0.0%";

  const percent = value * 100;
  if (percent >= 100) return `${capAt.toFixed(1)}%`;
  return `${percent.toFixed(1)}%`;
}

function formatRawPercent(value: number | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "0.0%";
  return `${(value * 100).toFixed(1)}%`;
}

function buildProbabilityEntries(
  probabilities: Record<string, number> | undefined,
): Array<[string, number]> {
  const safe = probabilities ?? {};
  return RISK_ORDER.map((label) => [label, safe[label] ?? 0]) as Array<[string, number]>;
}

function riskToneClass(risk: string | undefined): string {
  if (!risk) return "tone-neutral";
  const normalized = risk.toLowerCase();

  if (normalized.includes("high")) return "tone-high";
  if (normalized.includes("medium")) return "tone-medium";
  if (normalized.includes("low")) return "tone-low";

  return "tone-neutral";
}

function getHeadline(risk: string | undefined): string {
  if (!risk) return "No prediction yet";
  if (risk === "High Risk") return "Higher-priority triage signal";
  if (risk === "Medium Risk") return "Moderate-priority triage signal";
  return "Lower-priority triage signal";
}

function getSummaryCopy(risk: string | undefined): string {
  if (!risk) return "Submit an image and supporting notes to generate a triage summary.";

  if (risk === "High Risk") {
    return "The current inputs suggest a higher-risk pattern and should be escalated for prompt clinical review.";
  }

  if (risk === "Medium Risk") {
    return "The current inputs suggest an intermediate-risk pattern that still warrants clinical evaluation.";
  }

  return "The current inputs suggest a lower-risk pattern, but this should still be interpreted in context by a clinician.";
}

function buildLocalHistoryItem(
  response: PredictionResponse,
  fileName: string,
  noteText: string,
): PredictionHistoryItem {
  return {
    id: response.request_id ?? `local-${Date.now()}`,
    created_at: new Date().toISOString(),
    image_name: fileName,
    note_text: noteText,
    triage_level: response.triage_level,
    confidence: response.confidence,
    probabilities: response.probabilities,
    predicted_index: response.predicted_index,
  };
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [noteText, setNoteText] = useState<string>("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState<boolean>(true);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadHistory() {
      setIsLoadingHistory(true);

      try {
        const items = await getPredictions();
        if (!cancelled) {
          setHistory(items);
        }
      } catch {
        if (!cancelled) {
          setHistory([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingHistory(false);
        }
      }
    }

    void loadHistory();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const probabilityEntries = useMemo(
    () => buildProbabilityEntries(prediction?.probabilities),
    [prediction],
  );

  const selectedHistoryItem = useMemo(() => {
    if (!selectedHistoryId) return null;
    return history.find((item) => item.id === selectedHistoryId) ?? null;
  }, [history, selectedHistoryId]);

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function applyFile(file: File | null) {
    if (!file) return;
    setSelectedFile(file);
    setError("");
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    applyFile(file);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0] ?? null;
    applyFile(file);
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
  }

  function handleTemplateClick(template: string) {
    setNoteText(template);
    setError("");
  }

  function clearNotes() {
    setNoteText("");
    setError("");
  }

  function resetCase() {
    setSelectedFile(null);
    setPreviewUrl("");
    setNoteText("");
    setPrediction(null);
    setError("");
    setSelectedHistoryId(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function refreshHistory() {
    try {
      const items = await getPredictions();
      setHistory(items);
    } catch {
      // keep current history if refresh fails
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Please upload an image before analyzing the case.");
      return;
    }

    setIsSubmitting(true);
    setError("");

    try {
      const response = await predict({
        file: selectedFile,
        noteText,
      });

      setPrediction(response);

      try {
        await refreshHistory();
      } catch {
        const localItem = buildLocalHistoryItem(response, selectedFile.name, noteText);
        setHistory((prev) => [localItem, ...prev].slice(0, 25));
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Unable to analyze this case right now.";
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  function loadHistoryItem(item: PredictionHistoryItem) {
    setSelectedHistoryId(item.id);
    setPrediction({
      predicted_index: item.predicted_index ?? 0,
      triage_level: item.triage_level,
      confidence: item.confidence,
      probabilities: item.probabilities,
      request_id: item.id,
    });
    setNoteText(item.note_text ?? "");
    setError("");
  }

  return (
    <div className="app-shell">
      <div className="app-header">
        <div>
          <div className="eyebrow">Multimodal dermatology triage</div>
          <h1>{APP_NAME}</h1>
          <p className="subtle">
            Upload a lesion image, add any helpful clinical context, and review a
            three-level triage summary with probability breakdown.
          </p>
        </div>
      </div>

      <div className="layout-grid">
        <section className="panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Step 1</div>
              <h2>Case Intake</h2>
            </div>
            <button type="button" className="ghost-button" onClick={resetCase}>
              Reset case
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={handleFileChange}
            />

            <div
              className={`upload-card ${isDragging ? "is-dragging" : ""}`}
              onClick={openFilePicker}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  openFilePicker();
                }
              }}
            >
              {previewUrl ? (
                <img src={previewUrl} alt="Selected lesion" className="upload-preview" />
              ) : (
                <div className="upload-placeholder">Select image</div>
              )}

              <div className="upload-copy">
                <strong>{selectedFile?.name ?? "No image selected"}</strong>
                <span>
                  {selectedFile
                    ? "Click to replace or drag in a new image"
                    : "Click to upload or drag and drop an image"}
                </span>
              </div>
            </div>

            <div className="template-row">
              <span className="section-label">Quick note templates</span>
              <div className="chip-row">
                {NOTE_TEMPLATES.map((template, index) => (
                  <button
                    key={`template-${index}`}
                    type="button"
                    className="chip"
                    onClick={() => handleTemplateClick(template)}
                  >
                    Template {index + 1}
                  </button>
                ))}
              </div>
            </div>

            <label className="section-label" htmlFor="clinical-notes">
              Clinical notes
            </label>
            <textarea
              id="clinical-notes"
              className="notes-box"
              value={noteText}
              onChange={(event: ChangeEvent<HTMLTextAreaElement>) => setNoteText(event.target.value)}
              placeholder="Enter age, lesion location, symptoms, recent changes, and relevant history..."
              rows={8}
            />

            <div className="action-row">
              <button type="submit" className="primary-button" disabled={isSubmitting}>
                {isSubmitting ? "Analyzing..." : "Analyze Case"}
              </button>
              <button type="button" className="secondary-button" onClick={clearNotes}>
                Clear notes
              </button>
            </div>

            {error ? <div className="error-banner">{error}</div> : null}

            <div className="hint-card">
              <div className="eyebrow">Helpful inputs</div>
              <h3>What makes a stronger request?</h3>
              <ul>
                {CLINICAL_HINTS.map((hint) => (
                  <li key={hint}>{hint}</li>
                ))}
              </ul>
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <div className="eyebrow">Step 2</div>
              <h2>Triage Summary</h2>
            </div>
          </div>

          {!prediction ? (
            <div className="empty-state">
              <div className="empty-icon" />
              <h3>No prediction yet</h3>
              <p>
                Submit an image and supporting notes to generate a triage summary,
                confidence score, and probability breakdown.
              </p>

              <div className="feature-grid">
                <div className="feature-card">
                  <h4>Readable output</h4>
                  <p>Clear risk status with natural, human-friendly explanation.</p>
                </div>
                <div className="feature-card">
                  <h4>Explainability</h4>
                  <p>Probability bars make the output easier to interpret quickly.</p>
                </div>
                <div className="feature-card">
                  <h4>Reusable history</h4>
                  <p>Quick access to recent cases and saved supporting notes.</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="summary-stack">
              <div className={`summary-card ${riskToneClass(prediction.triage_level)}`}>
                <div className="summary-header">
                  <span className="risk-pill">{prediction.triage_level}</span>
                  <div className="confidence-ring">
                    <strong>{formatPercent(prediction.confidence)}</strong>
                    <span>model confidence</span>
                  </div>
                </div>

                <h3>{getHeadline(prediction.triage_level)}</h3>
                <p>{getSummaryCopy(prediction.triage_level)}</p>
                <p className="subtle">
                  This is model confidence, not a clinical diagnosis or absolute certainty.
                </p>
              </div>

              <div className="stats-grid">
                <div className="stat-card">
                  <span className="stat-label">Request ID</span>
                  <strong>{prediction.request_id ?? "N/A"}</strong>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Model device</span>
                  <strong>{prediction.model_info?.device ?? "N/A"}</strong>
                </div>
                <div className="stat-card">
                  <span className="stat-label">Max token length</span>
                  <strong>{prediction.model_info?.max_len ?? "N/A"}</strong>
                </div>
              </div>

              <div className="probability-card">
                <div className="probability-header">
                  <h4>Risk distribution</h4>
                  <span>Class probabilities</span>
                </div>

                <div className="probability-list">
                  {probabilityEntries.map(([label, value]) => (
                    <div key={label} className="probability-item">
                      <div className="probability-row">
                        <span>{label}</span>
                        <strong>{formatRawPercent(value)}</strong>
                      </div>
                      <div className="progress-track">
                        <div
                          className={`progress-fill ${riskToneClass(label)}`}
                          style={{ width: `${Math.max(2, value * 100)}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="notes-card">
                <h4>Case text used by the model</h4>
                <p>
                  {prediction.text_used
                    ? prediction.text_used
                    : noteText || "No normalized note text returned by the backend."}
                </p>
              </div>
            </div>
          )}
        </section>
      </div>

      <section className="panel history-panel">
        <div className="panel-header">
          <div>
            <div className="eyebrow">Recent cases</div>
            <h2>Reusable history</h2>
          </div>
          <button
            type="button"
            className="ghost-button"
            onClick={() => {
              void refreshHistory();
            }}
          >
            Refresh
          </button>
        </div>

        {isLoadingHistory ? (
          <p className="subtle">Loading history...</p>
        ) : history.length === 0 ? (
          <p className="subtle">No saved predictions yet.</p>
        ) : (
          <div className="history-list">
            {history.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`history-item ${selectedHistoryItem?.id === item.id ? "is-active" : ""}`}
                onClick={() => loadHistoryItem(item)}
              >
                <div className="history-top">
                  <strong>{item.image_name}</strong>
                  <span className={`risk-pill small ${riskToneClass(item.triage_level)}`}>
                    {item.triage_level}
                  </span>
                </div>

                <div className="history-meta">
                  <span>{new Date(item.created_at).toLocaleString()}</span>
                  <span>{formatPercent(item.confidence)}</span>
                </div>

                <p className="history-note">{item.note_text || "No note text saved."}</p>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}