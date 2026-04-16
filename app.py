import os
import sys
import certifi
from dotenv import load_dotenv
import pymongo

import pandas as pd

from logger.logger import logging
from exception.exception import ProjectError

from pipeline.training_pipeline import TrainingPipeline

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from starlette.responses import RedirectResponse  

from constants.training_pipeline import COLLECTION_NAME, DB_NAME, TARGET_COLUMN
from utils.main_utils.utils import read_pickle_object
from utils.ml_utils.model.estimator.estimator import PredictorModel

ca = certifi.where()

mongodb_url = load_dotenv("MONGODB_URL")

client = pymongo.MongoClient(mongodb_url, tls = True, tlsCertificateKeyFile = ca)
database = client[DB_NAME]
collection = database[COLLECTION_NAME]

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

@app.get('/', tags = ['authnetication'])
async def index():
    return RedirectResponse(url= '/docs')

@app.get('/predict')
async def predict(req: Request, file: UploadFile = File(...)):
    try:

        df = pd.read_csv(file)

        preprocessor = read_pickle_object('/final_model/preprocessor.pkl')
        model = read_pickle_object('/final_model/model.pkl')

        predictor_model = PredictorModel(preprocessor, model)

        y_pred = predictor_model.predict(df)

        df[TARGET_COLUMN] = y_pred

        response = df.to_json(orient='records')
        return Response(response, media_type='application/json')
    
    except Exception as e:
        raise ProjectError(e, sys)
    


@app.get('/train')
async def train_route():
    try:
        pipeline = TrainingPipeline()
        pipeline.initiate_pipeline()
        Response("Training completed and successful")

    except Exception as e:
        raise ProjectError(e, sys)


if __name__ == '__main__':
    run(app, host='localhost', port = 8000)
    