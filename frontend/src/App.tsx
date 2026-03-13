import { useEffect, useState } from "react";
import {
  getPredictions,
  predict,
  type PredictionHistoryItem,
  type PredictionResponse,
} from "./api";

function App() {
  const [image, setImage] = useState<File | null>(null);
  const [notes, setNotes] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [error, setError] = useState<string>("");

  async function loadHistory(): Promise<void> {
    try {
      const data = await getPredictions();
      setHistory(data.items || []);
    } catch (err) {
      console.error(err);
    }
  }

  useEffect(() => {
    void loadHistory();
  }, []);

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>): Promise<void> {
    e.preventDefault();

    if (!image) {
      setError("Please upload an image.");
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

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>): void {
    const file = e.target.files?.[0] || null;
    setImage(file);
  }

  return (
    <div className="page">
      <header className="hero">
        <h1>Multimodal Medical Triage</h1>
        <p>
          Upload a lesion image and optionally add clinical notes for triage
          prediction.
        </p>
      </header>

      <main className="grid">
        <section className="card">
          <h2>New Prediction</h2>

          <form onSubmit={handleSubmit} className="form">
            <input type="file" accept="image/*" onChange={handleFileChange} />

            <textarea
              placeholder="Enter clinical notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={5}
            />

            <button type="submit" disabled={loading}>
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </form>

          {error ? <p className="error">{error}</p> : null}
        </section>

        <section className="card">
          <h2>Latest Result</h2>

          {result ? (
            <div className="result">
              <p>
                <strong>Prediction:</strong> {result.predicted_class}
              </p>
              <p>
                <strong>Confidence:</strong>{" "}
                {(result.confidence * 100).toFixed(2)}%
              </p>
              <p>
                <strong>Latency:</strong> {result.duration_ms} ms
              </p>
              <p>
                <strong>Model version:</strong> {result.model_version}
              </p>

              {result.image_url ? (
                <p>
                  <a href={result.image_url} target="_blank" rel="noreferrer">
                    Uploaded image URL
                  </a>
                </p>
              ) : null}

              <div>
                <strong>Warnings</strong>
                <ul>
                  {result.warnings.length > 0 ? (
                    result.warnings.map((warning, index) => (
                      <li key={index}>{warning}</li>
                    ))
                  ) : (
                    <li>None</li>
                  )}
                </ul>
              </div>

              <div>
                <strong>Probabilities</strong>
                <pre>{JSON.stringify(result.probabilities, null, 2)}</pre>
              </div>
            </div>
          ) : (
            <p>No prediction yet.</p>
          )}
        </section>

        <section className="card full">
          <h2>Prediction History</h2>

          <div className="history">
            {history.length > 0 ? (
              history.map((item) => (
                <div key={item.request_id} className="history-item">
                  <p>
                    <strong>{item.predicted_class}</strong> —{" "}
                    {(item.confidence * 100).toFixed(2)}%
                  </p>
                  <p>{item.timestamp}</p>
                  <p>{item.notes}</p>
                </div>
              ))
            ) : (
              <p>No history yet.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;