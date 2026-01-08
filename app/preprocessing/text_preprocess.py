from transformers import DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_text(text):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
