import os
import sys
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import joblib

DATA_DIR = r"d:\AIML FA2\Plant-Disease-Detection-CNN\data\PlantVillage"
SAVE_DIR = r"d:\AIML FA2\Plant-Disease-Detection-Clustering\models"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print("Gathering image paths...")
image_paths = []
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images.")

# Hyperparameters for traditional ML approach
IMG_SIZE = (64, 64) # Resize to a smaller size to make flattening viable
BATCH_SIZE = 1000

def extract_raw_features(img_paths):
    """
    Extract raw pixel features by resizing and flattening the images.
    No CNNs or deep learning models are used here.
    """
    features = []
    for i, p in enumerate(img_paths):
        try:
            # 1. Load image and resize
            img = Image.open(p).convert('RGB')
            img = img.resize(IMG_SIZE)
            
            # 2. Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # 3. Flatten the image into a 1D vector (64 * 64 * 3 = 12288 dimensions)
            features.append(img_array.flatten())
            
        except Exception as e:
            print(f"Error loading {p}: {e}")
            
        if (i + 1) % BATCH_SIZE == 0:
            print(f"Processed {i + 1}/{len(img_paths)} images...")
            
    return np.array(features)

print("Extracting raw flattened pixel features (from scratch, no CNN)...")
features = extract_raw_features(image_paths)
print(f"Extracted raw features shape: {features.shape}")

print("Applying PCA for dimensionality reduction...")
# Reduce the 12288 dimensions down to a smaller set for KMeans
pca = PCA(n_components=128, random_state=42)
reduced_features = pca.fit_transform(features)

print("Training KMeans clustering...")
# 15 subdirectories typically in this subset
kmeans = MiniBatchKMeans(n_clusters=15, random_state=42, batch_size=1024)
kmeans.fit(reduced_features)

print("Saving models...")
joblib.dump(pca, os.path.join(SAVE_DIR, "pca_model.pkl"))
joblib.dump(kmeans, os.path.join(SAVE_DIR, "kmeans_model.pkl"))
print("Clustering training completed successfully!")
