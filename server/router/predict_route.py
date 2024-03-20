from fastapi import APIRouter, UploadFile
import os, sys, time
from loguru import logger

sys.path.append(r'../')

from server.voice_model import load_whisper, predict_whisper

predict_router = APIRouter()

FILE_DIR = r'./data'

model = None


def get_whisper():
    global model
    model = load_whisper()


@predict_router.post("")
async def predict(upload_file: UploadFile):
    content = await upload_file.read()
    audio_file = os.path.join(FILE_DIR, upload_file.filename)
    with open(audio_file, "wb") as file:
        file.write(content)

    language, text = predict_whisper(model, audio_file)

    # print the recognized text
    return {language, text}
