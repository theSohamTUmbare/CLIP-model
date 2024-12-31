from typing import List
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPTokenizer



# Load tokenizer
def_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

def the_tokenizer(text):
    """Tokenize the input text."""
    tokens = def_tokenizer(
        text, 
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return tokens["input_ids"].squeeze()

def zero_shot_classification(image: torch.Tensor, class_labels: List[str], model, device):
    """Classify an image with class labels and return similarity scores."""
    # Encode image
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        image_features = model.image_encoder(image)
        image_features = F.normalize(image_features, dim=-1)
        image_features = F.normalize(image_features.mean(dim=1), dim=-1)

    # Encode class labels as text
    class_text_features = []
    for label in class_labels:
        text_tokens = the_tokenizer(label).to(device)
        text_tokens = text_tokens.unsqueeze(0)
        with torch.no_grad():
            text_features = model.text_encoder(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            class_text_features.append(F.normalize(text_features.mean(dim=1), dim=-1))

    class_text_features = torch.stack(class_text_features).squeeze(1)

    # Compute similarities
    similarities = model.logit_scale.exp() * torch.matmul(image_features, class_text_features.T)
    
    # Convert similarities to a dictionary of scores
    similarity_scores = similarities.squeeze().tolist()
    label_scores = {label: score for label, score in zip(class_labels, similarity_scores)}
    
    return label_scores
