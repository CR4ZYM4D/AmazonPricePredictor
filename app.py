import os
import io
import sys
import datetime

import certifi
from dotenv import load_dotenv
import pymongo

import mlflow
import pandas as pd

from logger.logger import logging
from exception.exception import ProjectError

from components.data_ingestion import IngestionComponent
from entity.config_entity import IngestionConfig
from pipeline.training_pipeline import TrainingPipeline
from pipeline.prediction_pipeline import PredictionPipeline
from schema.product_input import ProductInput

from fastapi import FastAPI, File, UploadFile, Request, Body
from jinja2 import Environment, FileSystemLoader
from starlette.templating import Jinja2Templates
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse  
from uvicorn import run

from constants.training_pipeline import COLLECTION_NAME, DB_NAME, TARGET_COLUMN
from utils.main_utils.utils import read_pickle_object
from utils.ml_utils.model.estimator.estimator import PredictorModel

load_dotenv()
mongodb_url = os.getenv("MONGODB_URL")

if not mongodb_url:
    logging.error("MONGODB URL not found!! check .env file")
    raise ProjectError("MongoDB url not found!!", sys)



ca = certifi.where()
client = pymongo.MongoClient(mongodb_url, tls = True, tlsCAFile = ca)
database = client[DB_NAME]
collection = database[COLLECTION_NAME]

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

RUN_ID = os.getenv("RUN_ID")
ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH")
MODEL_VERSION = os.getenv("MODEL_VERSION")

logging.info(f"Loading model from DagsHub: models:/{ARTIFACT_PATH}/{MODEL_VERSION}")
loaded_model = mlflow.pyfunc.load_model(f"models:/{ARTIFACT_PATH}/{MODEL_VERSION}")

# init fast API app
app = FastAPI()

templates = Jinja2Templates(env=Environment(loader=FileSystemLoader(".")))

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*']
)

# routing

@app.get("/", tags=["ui"])
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/upload-csv", tags=["data"])
async def upload_csv(file: UploadFile = File(...)):
    """Bulk insert rows from uploaded CSV into MongoDB Atlas."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        records = df.to_dict(orient="records")
        if not records:
            return {"status": "error", "message": "CSV is empty"}

        # add metadata to each record
        for r in records:
            r["uploaded_at"] = datetime.datetime.utcnow()
            r["ingested"]    = False

        result = collection.insert_many(records)
        return {
            "status": "success",
            "inserted_count": len(result.inserted_ids)
        }
    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)

@app.post("/upload", tags=["data"])
async def upload_product(data: ProductInput = Body(...)):
    """Receives JSON from the HTML form, writes to MongoDB Atlas."""
    try:
        document = data.dict()
        document["uploaded_at"] = datetime.datetime.utcnow()
        document["ingested"]    = False
        result = collection.insert_one(document)
        return {"status": "success", "inserted_id": str(result.inserted_id)}
    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)


@app.get("/ingest", tags=["pipeline"])
async def ingest_route():
    """Pulls un-ingested Atlas documents through the DataIngestion component."""
    try:
        config    = IngestionConfig()
        ingestion = IngestionComponent(ingestion_config=config)
        artifact  = ingestion.initiate_data_ingestion()
        return {
            "status":     "ingestion complete",
            "train_file": str(artifact.trained_file_path),
            "test_file":  str(artifact.test_file_path),
        }
    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)


# @app.post("/predict", tags=["inference"])
# async def predict(req: Request, file: UploadFile = File(...)):
#     try:
#         df    = pd.read_csv(file.file)
#         y_pred = loaded_model.predict(df)
#         df[TARGET_COLUMN] = y_pred
#         return Response(df.to_json(orient="records"), media_type="application/json")
#     except Exception as e:
#         logging.error(e)
#         raise ProjectError(e, sys)

@app.post("/predict", tags=["prediction"])
async def predict(request: Request):
    try:
        pipeline = PredictionPipeline()
        df = pipeline.get_processed_dataframe()

        y_pred = loaded_model.predict(df)
        df[TARGET_COLUMN] = y_pred

        return df[['sample_id', TARGET_COLUMN]].to_dict(orient="records")

    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)


@app.get("/train", tags=["pipeline"])
async def train_route():
    try:
        pipeline = TrainingPipeline()
        pipeline.initiate_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)

if __name__ == '__main__':
    run(app, host='0.0.0.0', port = 8000)
    