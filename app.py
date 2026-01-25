from fastapi import FastAPI,UploadFile,File 
from fastapi.middleware.cors import CORSMiddleware

import shutil 
import os 

from src.cancer_clf.pipelines.predict import PredictionPipeline

app = FastAPI(title="Chest Cancer Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR,exist_ok=True)

model = PredictionPipeline(filename=None)

@app.get("/")
def get_health_check(self):
    return {"status":"ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR,file.filename)

    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    model.filename = file_path
    result = model.predict()

    return result