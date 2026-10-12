# Data — intentionally not stored here

The PlantVillage images, the stratified split CSVs and the diagnosis knowledge base are
**not copied into this folder**. They are read (read-only) from the sibling CNN project at run time:

```
../Plant-Disease-Detection-CNN/data/PlantVillage   20,636 images, 15 class folders
../Plant-Disease-Detection-CNN/data/split/*.csv    train.csv / val.csv / test.csv / class_indices.json
../Plant-Disease-Detection-CNN/data/diagnosis.json disease knowledge base
```

This guarantees the CNN and the transformer are trained on identical images in identical
partitions, and it keeps ~20k images from being duplicated on disk. The CNN project is never
modified.

If you keep only this folder and it cannot find the sibling project, pass the split directory
explicitly:

```
python src/train.py --split-dir "D:\path\to\Plant-Disease-Detection-CNN\data\split"
```

On the first run, `src/preprocessing.py` verifies that the splits are disjoint and exhaustive,
that every listed image exists and that `class_indices.json` matches the labels — and writes a
SHA-256 fingerprint of each split to `reports/data_provenance.json`.
