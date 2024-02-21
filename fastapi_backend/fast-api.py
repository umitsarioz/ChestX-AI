import os
import pickle

from fastapi import FastAPI
from keras.models import load_model
from pydantic import BaseModel

from chestxai import generate_caption


def load_models_files_from_gdrive() -> dict:
    os.system('sh script.sh')  # download files from google drive
    parent_fp = os.path.join(os.path.dirname(__file__), "models")
    filepaths = {'tokenizer': os.path.join(parent_fp, 'token_dct.pkl'),
                 'extract': os.path.join(parent_fp, 'xception.keras'),
                 'basic_model': os.path.join(parent_fp, 'basic_model.h5')}

    with open(filepaths["tokenizer"], "rb") as f:
        token_dct = pickle.load(f)

    extract_model = load_model(filepath=filepaths["extract"], compile=False)
    basic_model = load_model(filepath=filepaths["basic_model"], compile=False)

    dct = {"extract": extract_model,
           "basic_model": basic_model,
           "tokenizer": token_dct.get('tokenizer'),
           "vocabsize": token_dct.get('vocabsize'),
           "seq_len": token_dct.get('seq_len')
           }

    return dct


models = load_models_files_from_gdrive()

app = FastAPI()


class ImageModel(BaseModel):
    img_array: list = None
    model_name: str = ""


def select_data_by_model(model_name: str):
    if model_name == 'model_v1_CNN+LSTM':
        data = {'extract': models.get('extract'),
                'predict': models.get('basic_model'),
                'tokenizer': models.get('tokenizer'),
                'seq_len': models.get('seq_len')}
        return data
    elif model_name == 'model_v2_self-attention':
        # To-Do: It will changed when Attention model train is completed..
        data = {'extract': models.get('extract'),
                'predict': models.get('basic_model'),
                'tokenizer': models.get('tokenizer'),
                'seq_len': models.get('seq_len')}
        return data
    else:
        raise Exception("Undefined model.")


@app.get('/status')
def status():
    return {"message": "ChestXAI API is running."}


@app.post('/generate')
def predict_caption(inputs: ImageModel):
    print("trigger post request..")
    forecast_data = select_data_by_model(model_name=inputs.model_name)
    caption = generate_caption(inputs.img_array, forecast_data)
    return {"caption": caption}
