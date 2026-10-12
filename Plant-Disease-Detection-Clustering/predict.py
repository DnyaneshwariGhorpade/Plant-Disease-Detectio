import os
import sys
import numpy as np
from PIL import Image
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

print("Loading models...")
try:
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_model.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
except Exception as e:
    print(f"Error loading models. Make sure you have run the training script. {e}")
    sys.exit(1)

IMG_SIZE = (64, 64)

def predict_cluster(image_path):
    print(f"Loading image: {image_path}")
    
    # 1. Load image and resize (same as training)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    
    # 2. Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # 3. Flatten and expand dims to make it shape (1, features)
    features = img_array.flatten().reshape(1, -1)
    
    print("Applying dimensionality reduction (PCA)...")
    reduced_features = pca.transform(features)
    
    print("Predicting cluster...")
    cluster = kmeans.predict(reduced_features)[0]
    
    return cluster

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)
        
    cluster_id = predict_cluster(img_path)
    print(f"\n=====================")
    print(f"Result: Cluster {cluster_id}")
    print(f"=====================")
