from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
from dotenv import load_dotenv
import os

from router.predict_route import predict_router, get_whisper

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_whisper()
    logger.info("Model loaded on startup")
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(predict_router, tags=["Predict"], prefix="/predict")


@app.get("/")
async def root():
    return {"message": "Model serving test server"}


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=os.getenv("PORT"))