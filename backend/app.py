from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import joblib
import pickle
from io import StringIO
import json
import logging

# Added imports for database
from sqlalchemy import create_engine, Column, Integer, Float, String, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from backend.models import Base, PredictionHistory  # Import Base and models
from drift_utils import detect_drift


class PredictionRecord(BaseModel):
    Predicted_Default: int
    Default_Probability: float
    Risk_Category: str

class PredictResponse(BaseModel):
    batch_id: str
    predictions: List[Dict[str, Any]]
app = FastAPI(title="Loan Default Prediction API")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loan-default-prediction")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./loan_db.sqlite3")

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

# Load model, feature list, and scaler
model = joblib.load("backend/Models/logistic_top_model.pkl")

with open("backend/Models/top_model_features.pkl", "rb") as f:
    top_features = pickle.load(f)  # includes 'const'

scaler = joblib.load("backend/Models/StandardScaler.pkl")  #Standard scaler

# List of scaled numeric features (must match training order)
scaled_columns = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio"
]

class PredictionResult(BaseModel):
    Predicted_Default: int
    Default_Probability: float
    Risk_Category: str

import uuid

@app.post("/predict/", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    logger.info("Received /predict/ request")
    content = await file.read()

    try:
        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception:
        return JSONResponse(content={"error": "Invalid file format"}, status_code=400)

    logger.info(f"Dataframe loaded with {len(df)} rows")

    # Normalize categorical text
    for col in ["EmploymentType", "HasCoSigner", "HasDependents"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Save original values before scaling for storage
    original_values = df[["Age", "Income", "InterestRate", "EmploymentType"]].copy()

    # Step 1: Create dummy variables
    df = pd.get_dummies(df, drop_first=False)
    logger.info(f"Columns after get_dummies: {df.columns.tolist()}")

    # Step 2: Fill in missing columns with 0 (categorical & numeric)
    for col in top_features[1:]:  # skip 'const'
        if col not in df.columns:
            df[col] = 0

    # Step 3: Apply StandardScaler to scaled_columns
    for col in scaled_columns:
        if col not in df.columns:
            df[col] = 0  # fill missing numeric column with 0
    df[scaled_columns] = scaler.transform(df[scaled_columns])

    # Step 4: Reorder and insert constant
    try:
        X = df[top_features[1:]].copy()
    except KeyError as e:
        return JSONResponse(content={"error": f"Column mismatch: {e}"}, status_code=400)

    X.insert(0, "const", 1)

    # Step 5: Make predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    df["Predicted Default"] = preds
    df["Default Probability"] = probs
    df["Risk Category"] = pd.cut(
        probs, bins=[0, 0.33, 0.66, 1], labels=["Low", "Medium", "High"]
    )
    logger.info("File received and being processed...")

    # Generate a unique batch_id for this upload
    batch_id = str(uuid.uuid4())

    # Save prediction history to database
    session = SessionLocal()
    try:
        logger.info(f"Saving {len(df)} records to database with batch_id={batch_id}")
        for idx, row in df.iterrows():
            income_val = original_values.at[idx, "Income"]
            age_val = original_values.at[idx, "Age"]
            interest_rate_val = original_values.at[idx, "InterestRate"]
            employment_type_val = original_values.at[idx, "EmploymentType"]
            predicted_default_val = int(row["Predicted Default"])
            default_probability_val = float(row["Default Probability"])
            if income_val is not None:
                logger.info(f"Saving record idx={idx} income={income_val} age={age_val} interest_rate={interest_rate_val} employment_type={employment_type_val} predicted_default={predicted_default_val} default_probability={default_probability_val} batch_id={batch_id}")
                record = PredictionHistory(
                    batch_id=batch_id,
                    income=income_val,
                    predicted_default=predicted_default_val,
                    age=age_val,
                    interest_rate=interest_rate_val,
                    employment_type=employment_type_val,
                    default_probability=default_probability_val,
                )
                session.add(record)
        session.commit()
        logger.info("Database commit successful")
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving prediction history: {e}")
    finally:
        session.close()

    # Return the batch_id along with the prediction data
    return {"batch_id": batch_id, "predictions": df.to_dict(orient="records")}

@app.get("/test-db-connection")
async def test_db_connection():
    session = SessionLocal()
    try:
        test_record = PredictionHistory(
            income=12345.67,
            predicted_default=0,
            age=30,
            interest_rate=5.5,
            employment_type="Full-time",
            default_probability=0.1,
        )
        session.add(test_record)
        session.commit()
        logger.info("Test record added to database successfully")
        return {"status": "success", "message": "Test record added to database"}
    except Exception as e:
        session.rollback()
        logger.error(f"Error adding test record: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        session.close()

from typing import List, Dict, Any

@app.get("/feature-importance", response_model=List[Dict[str, Any]])
async def get_feature_importance() -> List[Dict[str, Any]]:
    try:
        with open("backend/Models/feature_importance.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/kpi-metrics", response_model=Dict[str, Any])
async def get_kpi_metrics() -> Dict[str, Any]:
    try:
        with open("backend/Models/kpi_metrics.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/average-income-by-default", response_model=List[Dict[str, Any]])
async def average_income_by_default() -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        query = session.query(
            PredictionHistory.predicted_default,
            func.count(PredictionHistory.id).label("count"),
            func.avg(PredictionHistory.income).label("average_income")
        ).group_by(PredictionHistory.predicted_default).all()

        result = []
        for row in query:
            default_status = "Default" if row.predicted_default == 1 else "No Default"
            result.append({
                "defaultStatus": default_status,
                "averageIncome": round(row.average_income, 2),
                "count": row.count
            })
        return result
    except Exception as e:
        logger.error(f"Error fetching average income by default: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        session.close()

@app.get("/default-rate-by-employment", response_model=List[Dict[str, Any]])
async def default_rate_by_employment() -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        query = session.query(
            PredictionHistory.employment_type,
            func.count(PredictionHistory.id).label("count"),
            func.avg(PredictionHistory.default_probability).label("defaultRate")
        ).group_by(PredictionHistory.employment_type).all()

        result = [
            {
                "employmentType": row.employment_type or "Unknown",
                "count": row.count,
                "defaultRate": float(row.defaultRate) if row.defaultRate is not None else 0.0,
            }
            for row in query
        ]
        return result
    except Exception as e:
        logger.error(f"Error fetching default rate by employment: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        session.close()

from fastapi import Query

@app.get("/loan-risk-data", response_model=List[Dict[str, Any]])
async def loan_risk_data(batch_id: str = Query(None)) -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        query = session.query(
            PredictionHistory.id,
            PredictionHistory.age,
            PredictionHistory.income,
            PredictionHistory.interest_rate,
            PredictionHistory.employment_type,
            PredictionHistory.predicted_default,
            PredictionHistory.default_probability,
        )
        if batch_id:
            query = query.filter(PredictionHistory.batch_id == batch_id)
        query = query.all()

        result = []
        for row in query:
            actual_default = bool(row.predicted_default)
            default_probability = row.default_probability if row.default_probability is not None else 0.0

            # Determine risk category based on default probability
            if default_probability >= 0.66:
                risk_category = "High"
            elif default_probability >= 0.33:
                risk_category = "Medium"
            else:
                risk_category = "Low"

            result.append({
                "id": str(row.id),
                "age": row.age,
                "income": row.income,
                "interestRate": row.interest_rate,
                "employmentType": row.employment_type,
                "defaultProbability": default_probability,
                "riskCategory": risk_category,
                "actualDefault": actual_default,
            })
        return result
    except Exception as e:
        logger.error(f"Error fetching loan risk data: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        session.close()


@app.post("/drift-on-final-model")
async def drift_on_final_model(new_batch: UploadFile = File(...)):
    try:
        # Load final model features (used in production)
        with open("backend/Models/top_model_features.pkl", "rb") as f:
            top_features = pickle.load(f)
        top_features = [f for f in top_features if f != "const"]

        # Load reference and new data
        reference_df = pd.read_csv("backend/Models/baseline.csv")  # save this manually from training
        new_df = pd.read_csv(new_batch.file)

        # Ensure only matching columns
        reference_df = reference_df[top_features]
        new_df = new_df[top_features]

        drift_result = detect_drift(reference_df, new_df, numeric_columns=top_features)
        return {"drift_report": drift_result}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/upload-baseline")
async def upload_baseline(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        df.to_csv("backend/Models/baseline.csv", index=False)
        return {"status": "success", "message": "Baseline uploaded"}
    except Exception as e:
        return {"error": str(e)}