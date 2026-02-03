def preprocess_text(text: str) -> str:
    if not text or not text.strip():
        return "No clinical description provided."
    return text.strip()
