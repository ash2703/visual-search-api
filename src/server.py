from fastapi import FastAPI, File, UploadFile, Query, Depends
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances


import uuid
import io
import numpy as np
from PIL import Image
from pathlib import Path

import logging

from src.features import FeatureExtractor
from src.search import FSearch
from src.utils import *
from src.configs import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Starting server')
app = FastAPI()

IMG_PATH = config.IMG_PATH
FEATURE_PATH = config.FEATURE_PATH

try:
    fe=FeatureExtractor()
    fs = FSearch(FEATURE_PATH, IMG_PATH, d = config.FEATURE_DIM)
    logger.info(f"model and feature databse loaded")
except Exception as e:
    logger.info("Exception occurred", exc_info=True)


@app.post("/uploadfile/")
async def create_upload_file(query_image: UploadFile = File(...)):
    """
    query_image: query image bytes
    """
    REQUEST_ID = Path(str(uuid.uuid4())[:8] + "." + query_image.filename.split(".")[-1])
    QUERY_PATH = Path("./static/uploaded/") / REQUEST_ID 

    logger.info("parsing image for feature extraction")
    with open(str(QUERY_PATH), 'wb') as myfile:
        contents = await query_image.read()
        myfile.write(contents)

    img_feats = fe.extract(QUERY_PATH).numpy().reshape(1, -1)
    _, ids = fs.query(img_feats)
    
    match_img_paths = [fs.img_paths[id] for id in ids[0]]  #single query hence extract 1st index
    logger.info(f"{str(match_img_paths)}")
    response = stack_images_side_by_side(QUERY_PATH, match_img_paths)

    _, encoded_img = cv2.imencode('.PNG', response)
    logger.info("succesfully encoded, now streaming")
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
