import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
import joblib

# Suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

print("Loading models...")
try:
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_model.pkl"))
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
except Exception as e:
    print(f"Error loading models. Make sure you have run the training script. {e}")
    sys.exit(1)

model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def predict_cluster(image_path):
    print(f"Loading image: {image_path}")
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = preprocess_input(img_array)
    
    features = model.predict(img_array, verbose=0)
    reduced_features = pca.transform(features)
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
