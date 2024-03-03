from fastapi import APIRouter, UploadFile
import os, sys

sys.path.append(r'../')

from server.voice_model import load_whisper, predict_whisper

predict_router = APIRouter()

FILE_DIR = r'./data'

@predict_router.post("/")
async def predict(upload_file: UploadFile):
    model = load_whisper()
    content = await upload_file.read()
    audio_file = os.path.join(FILE_DIR, upload_file.filename)
    with open(audio_file, "wb") as file:
        file.write(content)
    language, text = predict_whisper(model, audio_file)
    return {language, text}
