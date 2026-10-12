# Plant Disease Detection - Clustering

This repository section uses pure unsupervised learning (Clustering) to group plant diseases from the PlantVillage dataset from scratch, without relying on any Convolutional Neural Networks (CNN) or external deep learning models.

## Approach
1. **Raw Feature Extraction:** Images are loaded, resized to 64x64 pixels to save memory, converted to numpy arrays, and flattened into 1D vectors (size 12,288). No CNNs are used.
2. **Dimensionality Reduction:** `PCA` (Principal Component Analysis) is applied to reduce the raw 12,288 dimensions down to 128, which speeds up clustering and removes noise.
3. **Clustering:** `MiniBatchKMeans` groups the reduced features into 15 clusters (representing the 15 classes in this slice of the dataset).

## Usage
The dataset is taken directly from the CNN folder's dataset location (`data/PlantVillage`).

To train the model from scratch, simply run:
```bash
python train_clustering.py
```
This will generate `pca_model.pkl` and `kmeans_model.pkl` inside the `models` directory.

To predict the cluster of a new image:
```bash
python predict.py "path/to/image.jpg"
```
