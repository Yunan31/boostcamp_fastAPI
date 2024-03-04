from fastapi import APIRouter, UploadFile
import os, sys, time
import whisper
from loguru import logger

sys.path.append(r'/')

from voice_model import get_whisper, predict_whisper

predict_router = APIRouter()

FILE_DIR = r'data'

@predict_router.post("")
async def predict(upload_file: UploadFile):
    model = get_whisper()

    content = await upload_file.read()
    audio_file = os.path.join(FILE_DIR, upload_file.filename)
    with open(audio_file, "wb") as file:
        file.write(content)

    start = time.time()
    language, text = predict_whisper(model, audio_file)
    logger.info(f"Time taken: {time.time() - start}")

    # print the recognized text
    return {language, text}
