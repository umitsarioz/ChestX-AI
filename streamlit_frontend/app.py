import time

import numpy as np
import requests
import streamlit as st
from PIL import Image


def add_button(uploaded_file=None, img: list = None):
    if st.button("Generate Caption"):
        if uploaded_file is not None:
            payload = {'img_array': img}
            response = requests.post(url="http://service:8032/generate", json=payload)
            if response.status_code == 200:
                with st.spinner('Caption is generating...'):
                    time.sleep(2)
                    predicted_caption = response.json()["caption"]
                st.info("Caption:")
                st.success(predicted_caption)
            else:
                st.error("Fetching error for generating caption prediction.")
        else:
            st.warning("Load an image.")


def add_image():
    uploaded_file = st.file_uploader("Load image file", type=["png"])
    img = None
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        new_st_img = img.resize((200, 200))
        st.image(new_st_img, caption=uploaded_file.name)

        img = np.asarray(img).tolist()

    return uploaded_file, img


def main():
    st.title("🫁 ChestX-AI")
    st.info("Load a png chest x-ray image to predict caption.")
    uploaded_file, img = add_image()
    add_button(uploaded_file=uploaded_file, img=img)


if __name__ == "__main__":
    main()
