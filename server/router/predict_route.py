from fastapi import APIRouter, Depends
import os, sys, time
from loguru import logger
from models.metadata import Metadata
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
sys.path.append(r'../')

from server.voice_model import load_whisper_pipeline, predict_whisper
from server.classification_model import load_classifiers, predict_classification
from server.extract_feature import load_opensmile, extract_feature

predict_router = APIRouter()

DATA_DIR = os.getenv("DATA_DIR")

stt_pipeline = None
models = None
tokenizer = None
smile = None


def get_whisper_pipeline():
    global stt_pipeline
    stt_pipeline = load_whisper_pipeline()

def get_classifiers():
    global models, tokenizer
    models, tokenizer = load_classifiers()

def get_opensmile():
    global smile
    smile = load_opensmile()


@predict_router.post("")
async def predict(request: Metadata = Depends()):
    # excution time check
    start_time = time.time()

    # get metadata
    metadata = request.model_dump()
    logger.info(metadata)

    # make directory if id directory does not exist
    file_dir = os.path.join(DATA_DIR, metadata['id'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        result_df = pd.DataFrame(columns=['question', 'prob'])
        result_df.to_csv(os.path.join(file_dir, f"{metadata['id']}_result.csv"), index=False)

    # save the audio file
    upload_file = metadata['audio_file']
    content = await upload_file.read()
    audio_file = os.path.join(file_dir, upload_file.filename)
    with open(audio_file, "wb") as file:
        file.write(content)

    # extract features from the audio file
    audio_data = extract_feature(smile, audio_file)

    # predict the STT result
    text = predict_whisper(stt_pipeline, audio_file)

    # save the processing dataframe
    question = metadata['question']
    meta_df = pd.DataFrame(metadata, columns=metadata.keys(), index=[0])
    meta_df['audio_file'] = audio_file
    meta_df['stt'] = text
    audio_data['id'] = metadata['id']

    data_path = os.path.join(file_dir, f"{metadata['id']}_predict.csv")
    predict_df = meta_df.drop(columns=['key']).merge(audio_data, on='id')
    if not os.path.exists(data_path):
        concat_df = predict_df
        concat_df.to_csv(data_path, index=False)
    else:
        concat_df = pd.read_csv(data_path)
        concat_df = pd.concat([concat_df, predict_df])
        concat_df.to_csv(data_path, index=False)

    predict_df = predict_df.drop(columns=['id', 'audio_file', 'created_at'])

    # predict the classification result
    result = predict_classification(predict_df, models[question-1], tokenizer)

    # save the result
    result_df = pd.read_csv(os.path.join(file_dir, f"{metadata['id']}_result.csv"))
    prediction = pd.DataFrame({'question': question, 'prob': result}, index=[0])
    result_df = pd.concat([result_df, prediction])
    result_df.to_csv(os.path.join(file_dir, f"{metadata['id']}_result.csv"), index=False)
    
    logger.info(f"Total Time taken: {time.time() - start_time}")
    return {"prob": result}
