import io
import os
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import PlantDiseasePredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="Plant Disease Detection (Vision Transformer)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

predictor = PlantDiseasePredictor()

ALLOWED = {".jpg", ".jpeg", ".png"}


@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict_leaf(image: UploadFile = File(...)):
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Use jpg, jpeg or png.")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. File may be corrupted.")

    try:
        result = predictor.predict(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return result
