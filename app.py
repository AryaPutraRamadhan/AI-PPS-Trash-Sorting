
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import io

app = FastAPI()

# GANTI dengan repository Hugging Face milik kalian
HF_REPO_ID = "ArsFACE/trash-classifier"

# Download model dari Hugging Face
model_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    filename="best.pt"
)

# Load model YOLO
model = YOLO(model_path)


@app.get("/")
def root():
    return {"status": "ok", "classes": model.names}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(
        io.BytesIO(await file.read())
    ).convert("RGB")

    result = model.predict(
        img,
        conf=0.4,
        imgsz=640
    )[0]

    if len(result.boxes) == 0:
        return {"label": "none", "confidence": 0.0}

    i = int(result.boxes.conf.argmax())

    return {
        "label": model.names[int(result.boxes.cls[i])],
        "confidence": float(result.boxes.conf[i]),
    }
