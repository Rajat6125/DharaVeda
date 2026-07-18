import io
from pathlib import Path
from typing import Union
import base64

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
import cv2
import numpy as np

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_category is None:
            target_category = output.argmax(dim=1).item()
            
        loss = output[0, target_category]
        loss.backward()
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam -= torch.min(cam)
        if torch.max(cam) != 0:
            cam /= torch.max(cam)
            
        self.forward_hook.remove()
        self.backward_hook.remove()
        
        return cam.cpu().detach().numpy()

class PlantDiseaseClassifier:
    """
    Wraps a trained checkpoint for easy, correct inference with GradCAM and image checks.
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        # Patch for PosixPath on Windows
        import pathlib
        import platform
        if platform.system() == 'Windows':
            pathlib.PosixPath = pathlib.WindowsPath
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Failed to load checkpoint with weights_only=False: {e}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.class_names = checkpoint["class_names"]
            self.image_size = checkpoint.get("image_size", 224)
            self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
            self.std = checkpoint.get("std", [0.229, 0.224, 0.225])
            model_name = checkpoint.get("model_name", "efficientnet_b0")
            num_classes = checkpoint.get("num_classes", len(self.class_names))
            state_dict = checkpoint["model_state_dict"]
        else:
            import json
            class_names_path = Path(checkpoint_path).parent / "class_names.json"
            if class_names_path.exists():
                with open(class_names_path, 'r') as f:
                    self.class_names = json.load(f)
            else:
                raise ValueError("class_names.json not found but required for raw state dict.")
                
            self.image_size = 224
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            model_name = "efficientnet_b0"
            num_classes = len(self.class_names)
            state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

        self.model = self._build_model(model_name, num_classes)
        
        # Strip prefixes from state_dict keys if they exist (e.g. from DataParallel)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v
            elif k.startswith('model.'):
                clean_state_dict[k[6:]] = v
            else:
                clean_state_dict[k] = v
                
        try:
            self.model.load_state_dict(clean_state_dict, strict=False)
        except Exception as e:
            print(f"Warning: load_state_dict had issues: {e}")
            if isinstance(checkpoint, torch.nn.Module):
                print("Falling back to full model object from checkpoint.")
                self.model = checkpoint
                
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
            raise ValueError(f"Currently only knows how to rebuild 'efficientnet_b0', got {model_name}")
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def _load_image(self, image: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            img = image
        elif isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image))
        else:
            img = Image.open(image)
        return img.convert("RGB")

    def check_image_quality(self, image: Union[str, Path, bytes, Image.Image]) -> dict:
        try:
            img = self._load_image(image)
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # 1. Blur detection — only reject extremely blurry images (< 5.0)
            blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_val < 5.0:
                return {"valid": False, "reason": "Image is too blurry. Please retake the photo in focus."}

            # 2. Dark detection — only reject nearly black images (< 15.0)
            mean_brightness = np.mean(gray)
            if mean_brightness < 15.0:
                return {"valid": False, "reason": "Image is too dark. Please retake with better lighting."}

            return {"valid": True, "reason": "OK"}
        except Exception as e:
            # If quality check itself fails, let the model try anyway
            return {"valid": True, "reason": "OK"}

    def preprocess(self, image: Union[str, Path, bytes, Image.Image]) -> torch.Tensor:
        img = self._load_image(image)
        tensor = self.transform(img)
        return tensor.unsqueeze(0)

    def predict_with_gradcam(self, image: Union[str, Path, bytes, Image.Image], top_k: int = 3) -> dict:
        img = self._load_image(image)
        input_tensor = self.preprocess(img).to(self.device)

        # ── Plain inference (no GradCAM) ─────────────────────────────
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top_k = min(top_k, len(self.class_names))
        top_probs, top_idxs = torch.topk(probs, top_k)

        top_k_results = [
            (self.class_names[idx.item()], round(prob.item(), 4))
            for prob, idx in zip(top_probs, top_idxs)
        ]

        # ── GradCAM — use the last Conv2d inside features[-1] ────────
        heatmap_base64 = ""
        try:
            # Find the last Conv2d in the feature extractor for a reliable hook target
            target_layer = None
            for module in self.model.features.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module

            if target_layer is not None:
                with torch.enable_grad():
                    for param in self.model.parameters():
                        param.requires_grad = True

                    inp = self.preprocess(img).to(self.device)
                    inp.requires_grad = True

                    cam_extractor = SimpleGradCAM(self.model, target_layer)
                    target_cat = top_idxs[0].item()
                    heatmap_np = cam_extractor(inp, target_cat)

                    for param in self.model.parameters():
                        param.requires_grad = False

                # Overlay on original image
                heatmap_resized = cv2.resize(heatmap_np, (img.size[0], img.size[1]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                superimposed = np.clip(heatmap_colored * 0.4 + orig_bgr * 0.6, 0, 255).astype(np.uint8)
                _, buffer = cv2.imencode('.jpg', superimposed)
                heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as cam_err:
            # GradCAM failed — fall back to returning the plain image
            print(f"GradCAM error (non-fatal): {cam_err}")
            try:
                orig_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', orig_bgr)
                heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception:
                heatmap_base64 = ""

        return {
            "predicted_class": top_k_results[0][0],
            "confidence": top_k_results[0][1],
            "top_k": top_k_results,
            "heatmap_base64": heatmap_base64
        }
