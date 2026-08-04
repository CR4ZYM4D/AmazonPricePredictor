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

from fastapi import FastAPI, File, UploadFile, Request, Body, Depends, HTTPException, status
from jinja2 import Environment, FileSystemLoader
from starlette.templating import Jinja2Templates
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator
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

Instrumentator().instrument(app).expose(app)

ALLOWED_ADMIN_IP = os.getenv("ALLOWED_ADMIN_IP")
TRUST_PROXY = os.getenv("TRUST_PROXY", "false").lower() == "true"

async def verify_admin_ip(request: Request):
    """
    Security dependency to ensure only the administrator's IP
    can trigger heavy compute or data pipelines.
    """
    if TRUST_PROXY:
        forwarded_for = request.headers.get("X-Forwarded-For")
        client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else request.client.host
    else:
        # No trusted proxy in front — request.client.host is the real client IP
        client_ip = request.client.host

    if not ALLOWED_ADMIN_IP:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin IP validation is unconfigured on the server."
        )

    if client_ip != ALLOWED_ADMIN_IP:
        logging.warning(f"Unauthorized pipeline execution attempt blocked from IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: You do not have permission to run this pipeline."
        )

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
    """Receives JSON from the HTML form, writes to MongoDB, and returns an immediate price prediction."""
    try:
        # 1. Insert the record into MongoDB as un-ingested data
        document = data.dict()
        document["uploaded_at"] = datetime.datetime.utcnow()
        document["ingested"]    = False
        result = collection.insert_one(document)
        
        # 2. Run the prediction pipeline to process this new single record
        pipeline = PredictionPipeline(mongodb_client= client)
        df = pipeline.get_processed_dataframe()

        # 3. Predict the price using your loaded model
        y_pred = loaded_model.predict(df)
        df[TARGET_COLUMN] = y_pred
        
        # 4. Extract the specific predicted price for this item's sample_id
        matched_row = df[df['sample_id'] == data.sample_id]
        predicted_price = float(matched_row[TARGET_COLUMN].values[0]) if not matched_row.empty else 0.0

        # 5. Return the single prediction alongside all active batch records for the table display
        return {
            "status": "success", 
            "inserted_id": str(result.inserted_id),
            "predicted_price": predicted_price,
            "records": df[['sample_id', TARGET_COLUMN]].to_dict(orient="records")
        }
    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)


@app.get("/ingest", tags=["pipeline"], dependencies=[Depends(verify_admin_ip)])
async def ingest_route():
    """Pulls un-ingested Atlas documents through the DataIngestion component."""
    try:
        config    = IngestionConfig()
        ingestion = IngestionComponent(ingestion_config=config, mongo_client= client)
        artifact  = ingestion.initiate_data_ingestion()
        return {
            "status":     "ingestion complete",
            "train_file": str(artifact.trained_file_path),
            "test_file":  str(artifact.test_file_path),
        }
    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)

@app.post("/predict", tags=["prediction"])
async def predict(request: Request):
    try:
        pipeline = PredictionPipeline(mongodb_client= client)
        df = pipeline.get_processed_dataframe()

        y_pred = loaded_model.predict(df)
        df[TARGET_COLUMN] = y_pred

        return df[['sample_id', TARGET_COLUMN]].to_dict(orient="records")

    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)


@app.get("/train", tags=["pipeline"], dependencies=[Depends(verify_admin_ip)])
async def train_route():
    try:
        pipeline = TrainingPipeline(mongodb_client= client)
        pipeline.initiate_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)

if __name__ == '__main__':
    run(app, host='0.0.0.0', port = 8000)
    