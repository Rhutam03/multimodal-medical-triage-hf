import torch

def preprocess_text(text, max_len=100):
    """
    text: str or None
    """
    if text is None or text.strip() == "":
        return None

    tokens = [ord(c) % 1000 for c in text][:max_len]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
