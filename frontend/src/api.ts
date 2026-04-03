const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export type ClassProbabilities = Record<string, number>;

export type PredictionResponse = {
  predicted_index: number;
  triage_level: string;
  confidence: number;
  probabilities: ClassProbabilities;
  text_used?: string;
  request_id?: string;
  model_info?: {
    device?: string;
    weights_path?: string | null;
    vocab_path?: string | null;
    labels_csv_path?: string | null;
    max_len?: number;
  };
};

export type PredictionHistoryItem = {
  id: string;
  created_at: string;
  image_name: string;
  note_text: string;
  triage_level: string;
  confidence: number;
  probabilities: ClassProbabilities;
  predicted_index?: number;
};

export type PredictInput =
  | FormData
  | {
      file: File;
      noteText?: string;
      notes?: string;
      age?: string;
      sex?: string;
      site?: string;
    };

function buildFormData(input: PredictInput): FormData {
  if (input instanceof FormData) {
    return input;
  }

  const formData = new FormData();
  formData.append("file", input.file);
  formData.append("note_text", input.noteText ?? input.notes ?? "");
  formData.append("age", input.age ?? "");
  formData.append("sex", input.sex ?? "");
  formData.append("site", input.site ?? "");
  return formData;
}

async function parseResponse<T>(response: Response): Promise<T> {
  const rawText = await response.text();

  let data: unknown = null;
  try {
    data = rawText ? JSON.parse(rawText) : null;
  } catch {
    data = rawText;
  }

  if (!response.ok) {
    const detail =
      typeof data === "object" &&
      data !== null &&
      "detail" in data &&
      typeof (data as { detail?: unknown }).detail === "string"
        ? (data as { detail: string }).detail
        : typeof data === "string" && data
        ? data
        : `Request failed with status ${response.status}`;

    throw new Error(detail);
  }

  return data as T;
}

async function tryPostPrediction(formData: FormData): Promise<PredictionResponse> {
  const endpoints = ["/api/predict", "/predict", "/api/analyze", "/analyze"];

  let lastError: Error | null = null;

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });

      if (response.status === 404) {
        lastError = new Error(`Route not found: ${endpoint}`);
        continue;
      }

      return await parseResponse<PredictionResponse>(response);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error("Unknown request error");
      if (!/Route not found|404/.test(lastError.message)) {
        throw lastError;
      }
    }
  }

  throw lastError ?? new Error("Prediction request failed.");
}

export async function predict(input: PredictInput): Promise<PredictionResponse> {
  const formData = buildFormData(input);
  return tryPostPrediction(formData);
}

export async function getPredictions(): Promise<PredictionHistoryItem[]> {
  const endpoints = ["/api/predictions", "/predictions"];

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "GET",
      });

      if (response.status === 404) {
        continue;
      }

      return await parseResponse<PredictionHistoryItem[]>(response);
    } catch {
      continue;
    }
  }

  return [];
}