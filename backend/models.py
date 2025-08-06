from sqlalchemy import Column, Integer, Float, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

import uuid
from sqlalchemy import Column, Integer, Float, String, DateTime, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String, index=True, nullable=False, default=lambda: str(uuid.uuid4()))
    age = Column(Float, nullable=True)
    income = Column(Float, nullable=False)
    interest_rate = Column(Float, nullable=True)
    employment_type = Column(String, nullable=True)
    predicted_default = Column(Integer, nullable=False)
    default_probability = Column(Float, nullable=True)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
