import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
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

print("Loading MobileNetV2 feature extractor...")
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_paths, batch_size=128):
    features = []
    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i+batch_size]
        batch_imgs = []
        for p in batch_paths:
            try:
                img = image.load_img(p, target_size=(224, 224))
                batch_imgs.append(image.img_to_array(img))
            except Exception as e:
                print(f"Error loading {p}: {e}")
        if not batch_imgs:
            continue
        batch_imgs = np.array(batch_imgs)
        batch_imgs = preprocess_input(batch_imgs)
        batch_features = model.predict(batch_imgs, verbose=0)
        features.append(batch_features)
        if (i // batch_size) % 10 == 0:
            print(f"Processed {min(i+batch_size, len(img_paths))}/{len(img_paths)} images...")
    return np.vstack(features)

print("Extracting features...")
features = extract_features(image_paths)
print(f"Extracted features shape: {features.shape}")

print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=128, random_state=42)
reduced_features = pca.fit_transform(features)

print("Training KMeans clustering (MiniBatchKMeans)...")
# 15 subdirectories typically in this subset
kmeans = MiniBatchKMeans(n_clusters=15, random_state=42, batch_size=1024)
kmeans.fit(reduced_features)

print("Saving models...")
joblib.dump(pca, os.path.join(SAVE_DIR, "pca_model.pkl"))
joblib.dump(kmeans, os.path.join(SAVE_DIR, "kmeans_model.pkl"))
print("Clustering training completed successfully!")
