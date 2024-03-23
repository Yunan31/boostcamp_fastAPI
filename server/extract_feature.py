import opensmile
import pandas as pd
from loguru import logger 

def load_opensmile():
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    logger.info("load_opensmile complete")
    return smile

def extract_feature(smile, file_path):
    audio_data = pd.DataFrame()
    features = smile.process_file(file_path)
    audio_data = pd.concat([audio_data, features])
    return audio_data