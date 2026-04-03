const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export type AnalyzeCaseInput = {
  file: File;
  noteText?: string;
  age?: string;
  sex?: string;
  site?: string;
};

export async function analyzeCase(input: AnalyzeCaseInput) {
  const formData = new FormData();
  formData.append("file", input.file);
  formData.append("note_text", input.noteText ?? "");
  formData.append("age", input.age ?? "");
  formData.append("sex", input.sex ?? "");
  formData.append("site", input.site ?? "");

  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    body: formData,
  });

  const rawText = await response.text();

  let data: any = null;
  try {
    data = rawText ? JSON.parse(rawText) : null;
  } catch {
    data = null;
  }

  if (!response.ok) {
    throw new Error(
      data?.detail || rawText || `Request failed with status ${response.status}`
    );
  }

  return data;
}