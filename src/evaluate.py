import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from clip import CLIP
from train.zeroshot_classification import zero_shot_classification


def evaluate(image_path: str, class_labels: List[str], model, device):
    """Evaluate an image using a model and return the predicted class and similarity scores."""
    # Preprocessing pipeline
    preprocess = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)),  # COCO mean, std
    ])

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    image_tensor = preprocess(image).to(device)

    # Get similarity scores
    label_scores = zero_shot_classification(image_tensor, class_labels, model, device)

    # Print similarity scores
    print("Similarity scores for each class:")
    for label, score in label_scores.items():
        print(f"{label}: {score:.4f}")

    # Determine the best matching class
    best_class = max(label_scores, key=label_scores.get)
    print(f"\nThe predicted class is: {best_class}")

    return best_class, label_scores


if __name__ == "__main__":

    image_path = "/kaggle/input/coco-2017-dataset/coco2017/test2017/000000000345.jpg"
    class_labels = ["person", "man", "boy", "woman", "sky", "girl", "bus", "tables", "books", "laptop", "students", "children", "chairs"]

    # Load the model and set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIP()  # Initialize your CLIP model
    model.to(device)

    # Evaluate the image
    evaluate(image_path, class_labels, model, device)
