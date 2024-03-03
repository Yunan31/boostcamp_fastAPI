from fastapi import FastAPI
import uvicorn

from router.predict_route import predict_router
from voice_model import load_whisper


app = FastAPI()

app.include_router(predict_router, tags=["Predict"], prefix="/predict")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.on_event("startup")
async def load_model():
    global model
    model = load_whisper()
    return model


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)