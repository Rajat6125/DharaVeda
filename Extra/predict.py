"""
predict.py
==========
Standalone inference script for the plant disease classifier trained by
train_plant_disease_model.ipynb.

It loads the self-contained checkpoint (best_model.pth), rebuilds the exact
model architecture, applies the correct preprocessing, and returns the
predicted class name along with a confidence score.

Usage (CLI):
    python predict.py --checkpoint models/best_model.pth --image leaf.jpg
    python predict.py --checkpoint models/best_model.pth --image leaf.jpg --topk 5

Usage (as a library, e.g. in a Flask/FastAPI backend):
    from predict import PlantDiseaseClassifier

    clf = PlantDiseaseClassifier("models/best_model.pth")
    result = clf.predict("leaf.jpg")
    # result = {
    #     "predicted_class": "Tomato___Late_blight",
    #     "confidence": 0.9732,
    #     "top_k": [("Tomato___Late_blight", 0.9732), ("Tomato___Early_blight", 0.0181), ...]
    # }

    # Works with a PIL.Image, a file path (str/Path), or raw bytes:
    result = clf.predict(pil_image_object)
    result = clf.predict(open("leaf.jpg", "rb").read())
"""

import argparse
import io
import json
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class PlantDiseaseClassifier:
    """
    Wraps a trained checkpoint for easy, correct inference.

    The checkpoint produced by the training notebook is self-contained: it
    stores class_names, image_size, normalization mean/std, and model_name,
    so this class rebuilds the exact preprocessing/architecture used during
    training without any hardcoded assumptions.
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.class_names = checkpoint["class_names"]
        self.image_size = checkpoint.get("image_size", 224)
        self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
        self.std = checkpoint.get("std", [0.229, 0.224, 0.225])
        model_name = checkpoint.get("model_name", "efficientnet_b0")
        num_classes = checkpoint.get("num_classes", len(self.class_names))

        self.model = self._build_model(model_name, num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    @staticmethod
    def _build_model(model_name: str, num_classes: int) -> nn.Module:
        if model_name != "efficientnet_b0":
            raise ValueError(
                f"predict.py currently only knows how to rebuild 'efficientnet_b0', "
                f"got model_name='{model_name}'. Add a branch here if you changed "
                f"the architecture in the training notebook."
            )
        model = models.efficientnet_b0(weights=None)  # weights come from the checkpoint, not ImageNet
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def _load_image(self, image: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """Accepts a file path, raw bytes, or an already-open PIL Image."""
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(image)
        return img.convert("RGB")

    def preprocess(self, image: Union[str, Path, bytes, Image.Image]) -> torch.Tensor:
        """Returns a preprocessed (1, C, H, W) tensor ready for the model."""
        img = self._load_image(image)
        tensor = self.transform(img)
        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, bytes, Image.Image],
        top_k: int = 3,
    ) -> dict:
        """
        Run inference on a single image.

        Returns:
            {
                "predicted_class": str,
                "confidence": float,          # 0-1
                "top_k": [(class_name, prob), ...]  # sorted descending, length top_k
            }
        """
        input_tensor = self.preprocess(image).to(self.device)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # shape (num_classes,)

        top_k = min(top_k, len(self.class_names))
        top_probs, top_idxs = torch.topk(probs, top_k)

        top_k_results = [
            (self.class_names[idx.item()], round(prob.item(), 4))
            for prob, idx in zip(top_probs, top_idxs)
        ]

        return {
            "predicted_class": top_k_results[0][0],
            "confidence": top_k_results[0][1],
            "top_k": top_k_results,
        }

    @torch.no_grad()
    def predict_batch(self, images: list, top_k: int = 3) -> list:
        """Convenience helper for predicting a list of images one at a time."""
        return [self.predict(img, top_k=top_k) for img in images]


def main():
    parser = argparse.ArgumentParser(description="Predict a plant disease from a leaf image.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    parser.add_argument("--image", required=True, help="Path to the image to classify")
    parser.add_argument("--topk", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    clf = PlantDiseaseClassifier(args.checkpoint)
    result = clf.predict(args.image, top_k=args.topk)

    print(f"\nPredicted class : {result['predicted_class']}")
    print(f"Confidence      : {result['confidence'] * 100:.2f}%\n")
    print(f"Top-{args.topk} predictions:")
    for i, (cls, prob) in enumerate(result["top_k"], start=1):
        print(f"  {i}. {cls:<40s} {prob * 100:6.2f}%")


if __name__ == "__main__":
    main()
