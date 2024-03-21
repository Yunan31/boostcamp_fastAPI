from fastapi import APIRouter, UploadFile
import os, sys, time
from loguru import logger

sys.path.append(r'../')

from server.voice_model import load_whisper_pipeline, predict_whisper

predict_router = APIRouter()

FILE_DIR = r'./data'

stt_pipeline = None


def get_whisper_pipeline():
    global stt_pipeline
    stt_pipeline = load_whisper_pipeline()


@predict_router.post("")
async def predict(upload_file: UploadFile):
    content = await upload_file.read()
    audio_file = os.path.join(FILE_DIR, upload_file.filename)
    with open(audio_file, "wb") as file:
        file.write(content)

    text = predict_whisper(stt_pipeline, audio_file)

    # print the recognized text
    return {text}
