"""Inference for the Vision Transformer plant-disease classifier.

Mirrors the CNN project's predictor interface: top class, confidence, a
`confident` flag (false below the threshold), per-class probabilities and the
full diagnosis. Preprocessing is identical to the CNN's (RGB, squash resize to
96x96, scale to [0, 1]).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diagnosis import get_diagnosis
from model import build_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILENAME = "vit_plant_disease.pt"


class PlantDiseasePredictor:
    def __init__(self, model_dir=None, confidence_threshold=0.5, device=None):
        model_dir = model_dir or os.path.join(BASE_DIR, "models")
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(os.path.join(model_dir, "class_indices.json"), "r", encoding="utf-8") as f:
            self.class_indices = {int(k): v for k, v in json.load(f).items()}

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(
            arch=self.config.get("arch", "vit_tiny"),
            img_size=self.config.get("img_size", 96),
            patch_size=self.config.get("patch_size", 8),
            num_classes=self.config.get("num_classes", len(self.class_indices)),
        ).to(self.device)
        state = torch.load(os.path.join(model_dir, MODEL_FILENAME), map_location=self.device,
                           weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        self.img_size = self.config.get("img_size", 96)
        self.confidence_threshold = confidence_threshold
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    def preprocess(self, image):
        if isinstance(image, str) or not hasattr(image, "convert"):
            image = Image.open(image)
        image = image.convert("RGB")
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image):
        x = self.preprocess(image)
        logits = self.model(x)
        probs = logits.float().softmax(dim=1)[0].cpu().numpy()
        order = np.argsort(probs)[::-1]
        top_idx = int(order[0])
        top_class = self.class_indices[top_idx]
        top_conf = float(probs[top_idx])
        all_probs = {self.class_indices[i]: float(probs[i]) for i in range(len(probs))}
        diagnosis = get_diagnosis(top_class, confidence=top_conf)
        return {
            "model": "VisionTransformer",
            "top_class": top_class,
            "top_confidence": top_conf,
            "confident": top_conf >= self.confidence_threshold,
            "probabilities": all_probs,
            "top3": [
                {"class": self.class_indices[int(i)], "confidence": float(probs[int(i)])}
                for i in order[:3]
            ],
            "diagnosis": diagnosis,
        }


def main():
    parser = argparse.ArgumentParser(description="Predict plant disease with the Vision Transformer")
    parser.add_argument("--image", required=True, help="path to leaf image")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    predictor = PlantDiseasePredictor(model_dir=args.model_dir, confidence_threshold=args.threshold)
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
