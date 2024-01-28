import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from keras.models import load_model
from chestxai import generate_caption
from configs import filepaths

def get_models_from_aws() -> dict:
    dct = {"extract": "extraction model alindi",
           "predict": "prediction model alindi",
           "tokenizer": "tokenizer alindi",
           }
    return dct

def get_models_from_local() -> dict:
    with open(filepaths["local"]["extract"], "rb") as f:
        extract_model = pickle.load(f)

    with open(filepaths["local"]["tokenizer"], "rb") as f:
        tokenizer = pickle.load(f)

    pred_model = load_model(filepath=filepaths["local"]["predict"])

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


@app.post('/generate/')
def predict_caption(inputs: ImageModel):
    caption = generate_caption(inputs.img_array, models)
    return {"caption": caption}
