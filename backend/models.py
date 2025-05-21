from sqlalchemy import Column, Integer, String, DateTime, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
import enum
from datetime import datetime

Base = declarative_base()

class TaskStatus(str, enum.Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String, nullable=False)
    dataset_url = Column(String, nullable=False)
    hyperparams = Column(JSON, nullable=True)
    status = Column(Enum(TaskStatus), default=TaskStatus.pending)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    result_model_url = Column(String, nullable=True)

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer)
    url = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow) 