import os
import requests

import streamlit as st
from pathlib import Path

st.set_page_config(layout='centered')
st.title("Visual Search API")
uploaded_file = st.file_uploader("Choose a file", type = ["png", "jpg", "jpeg"])

assert uploaded_file is not None

QUERY_PATH = Path("./upload/") / uploaded_file.name 
RESPONSE_PATH = Path("./response/") 

URL_ENDPOINT = 'http://127.0.0.1:8000/uploadfile/'

os.makedirs("./upload/", exist_ok = True)
os.makedirs("./response/", exist_ok = True)

with open(str(QUERY_PATH), 'wb') as myfile:
    contents = uploaded_file.getvalue()    # To read file as bytes:
    myfile.write(contents)

st.image(contents, caption = "uploaded query image")

files = {
        "query_image": (QUERY_PATH.name,
                open(QUERY_PATH, 'rb'),
                "image/png"),
}

response = requests.post(URL_ENDPOINT, files=files)

if response.status_code == 200:
    with open(str(RESPONSE_PATH / "out.png"), 'wb') as f:
        f.write(response.content)

st.image(response.content, caption = "Closest matching images")
