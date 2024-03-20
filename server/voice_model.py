import whisper
from loguru import logger
import time
import torch
from dotenv import load_dotenv
import os

load_dotenv()


def load_whisper():
    model = whisper.load_model(os.getenv("MODEL_NAME"))
    logger.info("load_whisper triggered")
    return model


def predict_whisper(model, audio_file):
    start = time.time()

    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(os.getenv("DEVICE"))

    # detect the spoken language
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    logger.info(f"Time taken: {time.time() - start}")

    # print the recognized text
    return language, result.text