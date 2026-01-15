from transformers import AutoTokenizer
import torch

TOKENIZER_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def preprocess_text(text, max_len=128):
    """
    text: str | None
    returns: dict[str, Tensor] | None
    """
    if text is None or text.strip() == "":
        return None

    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
