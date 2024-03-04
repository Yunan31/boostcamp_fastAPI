from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger

from predict_route import predict_router
from voice_model import load_whisper


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_whisper("tiny")
    logger.info("Model loaded on startup")
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(predict_router, tags=["Predict"], prefix="/predict")

@app.get("/")
async def root():
    return {"message": "Model serving test server"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)