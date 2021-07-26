import cv2
import os
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import lib
import base64
import datetime
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

lib = lib.Lib()


class Frame(BaseModel):
    frame: str


@app.post("/predict")
async def predict(frame: Frame):
    image_data = base64.b64decode(frame.frame)
    filename = "{}.jpg".format('image'+str(datetime.datetime.now()))
    result = None
    with open(filename, "wb") as file:
        file.write(image_data)
        file.close()

    result = await lib.predict(filename)
    # clean up
    if os.path.exists(filename):
        os.remove(filename)
    return {"prediction": result}
