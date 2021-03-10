import os
import requests

import streamlit as st
from pathlib import Path

##### TODO: CONFIGS
URL_ENDPOINT = 'http://127.0.0.1:8000/uploadfile/'   # TODO add live ping

QUERY_PATH_DIR = Path("ui/upload/") 
RESPONSE_PATH = Path("ui/response/") 

os.makedirs(str(QUERY_PATH_DIR), exist_ok = True)
os.makedirs(str(RESPONSE_PATH), exist_ok = True)
#####

st.set_page_config(layout='centered')
st.title("Visual Search API")

uploaded_file = st.file_uploader("Choose a file", type = ["png", "jpg", "jpeg"])

assert uploaded_file is not None, "File upload error"

ID = uploaded_file.id
FILE_TYPE = uploaded_file.type.split("/")[-1]
FILE_PATH = QUERY_PATH_DIR / Path(ID + "." + FILE_TYPE)

with open(str(FILE_PATH), 'wb') as myfile:
    contents = uploaded_file.getvalue()    # To read file as bytes:
    myfile.write(contents)

st.image(contents, caption = "uploaded query image")

files = {
        "query_image": (FILE_PATH.name,
                open(FILE_PATH, 'rb'),
                f"image/{FILE_TYPE}"),
}

response = requests.post(URL_ENDPOINT, files=files)

if response.status_code == 200:
    # TODO: Save file by chunking
    with open(str(RESPONSE_PATH / Path(f"{ID}_result.{FILE_TYPE}")), 'wb') as f:
        f.write(response.content)

st.image(response.content, caption = "Closest matching images")
