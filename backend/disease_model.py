"""
disease_model.py
================
Plant disease classifier for the DharaVeda backend.
Uses plain torch.no_grad() inference (no GradCAM backward pass)
to stay within Render free-tier memory/time limits.
"""

import io
import base64
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import cv2
import numpy as np


class PlantDiseaseClassifier:

    def __init__(self, checkpoint_path: Union[str, Path], device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Support both self-contained checkpoints and raw state dicts
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.class_names = checkpoint["class_names"]
            self.image_size  = checkpoint.get("image_size", 224)
            self.mean        = checkpoint.get("mean", [0.485, 0.456, 0.406])
            self.std         = checkpoint.get("std",  [0.229, 0.224, 0.225])
            model_name       = checkpoint.get("model_name", "efficientnet_b0")
            num_classes      = checkpoint.get("num_classes", len(self.class_names))
            state_dict       = checkpoint["model_state_dict"]
        else:
            import json
            class_names_path = checkpoint_path.parent / "class_names.json"
            if not class_names_path.exists():
                raise ValueError("class_names.json not found next to checkpoint.")
            with open(class_names_path) as f:
                self.class_names = json.load(f)
            self.image_size = 224
            self.mean       = [0.485, 0.456, 0.406]
            self.std        = [0.229, 0.224, 0.225]
            model_name      = "efficientnet_b0"
            num_classes     = len(self.class_names)
            state_dict      = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

        self.model = self._build_model(model_name, num_classes)
        self.model.load_state_dict(state_dict)
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
            raise ValueError(f"Only 'efficientnet_b0' is supported, got '{model_name}'.")
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m

    def _load_image(self, image: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(image)
        return img.convert("RGB")

    def check_image_quality(self, image: Union[str, Path, bytes, Image.Image]) -> dict:
        """Only reject obviously unusable images."""
        try:
            img    = self._load_image(image)
            gray   = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            blur   = cv2.Laplacian(gray, cv2.CV_64F).var()
            bright = float(np.mean(gray))
            if blur < 5.0:
                return {"valid": False, "reason": "Image is too blurry. Please retake in focus."}
            if bright < 15.0:
                return {"valid": False, "reason": "Image is too dark. Please use better lighting."}
            return {"valid": True, "reason": "OK"}
        except Exception:
            return {"valid": True, "reason": "OK"}

    def _make_highlight_overlay(self, img: Image.Image) -> str:
        """
        Lightweight visual overlay: highlights the green channel (leaf tissue)
        in the original image without requiring any backward pass.
        Returns a base64-encoded JPEG string.
        """
        try:
            arr = np.array(img)                          # H×W×3  RGB uint8
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            # Build a simple mask: pixels where green channel dominates
            r, g, b = arr[:, :, 0].astype(float), arr[:, :, 1].astype(float), arr[:, :, 2].astype(float)
            green_score = g - 0.5 * r - 0.5 * b        # positive = greenish
            green_score = np.clip(green_score, 0, None)
            if green_score.max() > 0:
                green_score /= green_score.max()

            # Apply a jet-like overlay where the leaf is
            heatmap = np.uint8(255 * green_score)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = np.clip(heatmap * 0.35 + bgr * 0.65, 0, 255).astype(np.uint8)

            _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf).decode("utf-8")
        except Exception:
            # Last-resort: return the plain image
            try:
                bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", bgr)
                return base64.b64encode(buf).decode("utf-8")
            except Exception:
                return ""

    @torch.no_grad()
    def predict_with_gradcam(
        self,
        image: Union[str, Path, bytes, Image.Image],
        top_k: int = 3,
    ) -> dict:
        """
        Run inference and return prediction + visual overlay.
        Named predict_with_gradcam for API compatibility, but uses a
        lightweight green-channel overlay instead of a backward pass.
        """
        img          = self._load_image(image)
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

        top_k       = min(top_k, len(self.class_names))
        top_probs, top_idxs = torch.topk(probs, top_k)

        top_k_results = [
            (self.class_names[idx.item()], round(prob.item(), 4))
            for prob, idx in zip(top_probs, top_idxs)
        ]

        heatmap_base64 = self._make_highlight_overlay(img)

        return {
            "predicted_class": top_k_results[0][0],
            "confidence":      top_k_results[0][1],
            "top_k":           top_k_results,
            "heatmap_base64":  heatmap_base64,
        }
