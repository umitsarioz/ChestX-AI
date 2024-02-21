import numpy as np
from PIL import Image
from keras.applications import xception
from keras.utils import img_to_array, pad_sequences


def extract_an_image_feature(img: list, model) -> np.ndarray:
    print("extracting image features...")
    img_numpy = np.asarray(img)
    img = Image.fromarray(np.uint8(img_numpy)).convert('RGB')
    img_dim = model.input_shape[1:3]
    img = img.resize(img_dim)
    img = img_to_array(img)
    w, h, ch = img.shape[0], img.shape[1], img.shape[2]
    img = img.reshape((1, w, h, ch))

    if model.name == 'xception':
        img = xception.preprocess_input(img)
    else:
        raise Exception("undefined model name")

    feature = model.predict(img, verbose=0)
    return feature


def to_prediction(img_features, model, tokenizer, seq_length:int) -> str:
    print("prediction....")
    in_text = 'start'
    for i in range(seq_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=seq_length)
        # predict next word
        yhat = model.predict([img_features, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = tokenizer.index_word[yhat]
        # stop if we cannot map the word
        if word is None or word == 'end':
            in_text += ' ' + word
            break
        else:
            in_text += ' ' + word

    pred_text = in_text.replace('<unk>', ' ')
    pred_text = pred_text.replace('start ', '')
    pred_text = pred_text.replace(' end', '.').split()
    pred_text = ' '.join([pred_text[0].capitalize()] + pred_text[1:])

    return pred_text


def generate_caption(img: list = None, forecast_data: dict = None) -> str:
    img_features = extract_an_image_feature(img=img, model=forecast_data["extract"])
    caption = to_prediction(img_features=img_features, model=forecast_data["predict"], tokenizer=forecast_data["tokenizer"],seq_length=forecast_data["seq_len"])
    return caption
