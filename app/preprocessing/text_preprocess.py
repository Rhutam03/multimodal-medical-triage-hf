import torch
import torch.nn.functional as F

def preprocess_text(text: str, max_len: int = 128):
    """
    Always returns a tensor of shape [max_len]
    """

    if text is None:
        text = ""

    encoded = [ord(c) % 256 for c in text][:max_len]

    if len(encoded) == 0:
        encoded = [0]

    tensor = torch.tensor(encoded, dtype=torch.float32)

    if tensor.size(0) < max_len:
        tensor = F.pad(tensor, (0, max_len - tensor.size(0)))

    return tensor
