const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export type PredictionResponse = {
  request_id: string;
  model_version: string;
  predicted_class: string;
  class_id: number;
  confidence: number;
  probabilities: Record<string, number>;
  warnings: string[];
  image_url?: string;
  duration_ms: number;
  saved: boolean;
};

export type PredictionHistoryItem = {
  request_id: string;
  timestamp: string;
  predicted_class: string;
  class_id: number;
  confidence: number;
  probabilities: Record<string, number>;
  notes: string;
  image_url?: string;
  model_version: string;
};

export type PredictionListResponse = {
  items: PredictionHistoryItem[];
};

export async function predict(
  image: File,
  text: string
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("text", text);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || "Prediction request failed");
  }

  return response.json();
}

export async function getPredictions(): Promise<PredictionListResponse> {
  const response = await fetch(`${API_BASE_URL}/predictions`);

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || "Failed to load prediction history");
  }

  return response.json();
}