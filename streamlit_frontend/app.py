import time

import numpy as np
import requests
import streamlit as st
from PIL import Image


def btn_add_image():
    uploaded_file = st.file_uploader("Load image file", type=["png"])
    img = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        new_st_img = img.resize((200, 200))
        st.image(new_st_img, caption=uploaded_file.name)

        img = np.asarray(img).tolist()

    return uploaded_file, img


def cbbox_select_model():
    option = st.selectbox("Select a model",
                          ('model_v1_CNN+LSTM', 'model_v2_self-attention'),
                          index=None,
                          placeholder="Select a model ",
                          )
    return option


def btn_generate_caption(uploaded_file=None, model_option=None, img: list = None):
    if st.button("Generate Caption"):
        if model_option is None:
            st.warning("Select a prediction model")
        elif uploaded_file is None:
            st.warning("Load an image.")
        else:
            payload = {'img_array': img,'model_name':model_option}
            response = requests.post(url="http://service:8032/generate", json=payload) # http://service:8032/generate
            if response.status_code == 200:
                with st.spinner('Caption is generating...'):
                    time.sleep(2)
                    predicted_caption = response.json()["caption"]
                st.info("Caption:")
                st.success(predicted_caption)
            else:
                st.error("Fetching error for generating caption prediction.")


def main():
    st.title("🫁 ChestX-AI")
    st.info("Load a png chest x-ray image to predict caption.")
    model_option = cbbox_select_model()
    uploaded_file, img = btn_add_image()
    btn_generate_caption(uploaded_file=uploaded_file, model_option=model_option, img=img)


if __name__ == "__main__":
    main()
