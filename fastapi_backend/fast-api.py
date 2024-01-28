import os.path
import pickle
import uvicorn
from fastapi import FastAPI
from keras.models import load_model
from pydantic import BaseModel

try:
    from chestxai import generate_caption

    parent_fp = "models"

    filepaths = {'tokenizer': os.path.join(parent_fp, 'tokenizer.pkl'),
                 'extract': os.path.join(parent_fp, 'extractor.pkl'),
                 'predict': os.path.join(parent_fp, 'predictor.h5')}
except:
    from fastapi_backend.chestxai import generate_caption

    parent_fp = os.path.join("fastapi_backend", "models")

    filepaths = {'tokenizer': os.path.join(parent_fp, 'tokenizer.pkl'),
                 'extract': os.path.join(parent_fp, 'extractor.pkl'),
                 'predict': os.path.join(parent_fp, 'predictor.h5')}


def get_models_from_local() -> dict:
    with open(filepaths["extract"], "rb") as f:
        extract_model = pickle.load(f)

    with open(filepaths["tokenizer"], "rb") as f:
        tokenizer = pickle.load(f)

    pred_model = load_model(filepath=filepaths["predict"])

    dct = {"extract": extract_model,
           "predict": pred_model,
           "tokenizer": tokenizer,
           }

    return dct


models = get_models_from_local()
app = FastAPI()


class ImageModel(BaseModel):
    img_array: list = None

    class Config:
        arbitrary_types_allowed = True


@app.get('/status')
def status():
    return {"message": "ChestXAI API is running."}


@app.post('/generate')
def predict_caption(inputs: ImageModel):
    print("trigger post request..")
    caption = generate_caption(inputs.img_array, models)
    return {"caption": caption}
