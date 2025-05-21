from pydantic import BaseModel
from typing import Optional, Any
from enum import Enum

class TaskStatus(str, Enum):
    pending = "pending"
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"

class TaskCreate(BaseModel):
    model_type: str
    dataset_url: str
    hyperparams: Optional[Any] = None

class TaskOut(BaseModel):
    id: int
    model_type: str
    dataset_url: str
    hyperparams: Optional[Any]
    status: TaskStatus
    result_model_url: Optional[str]
    result_model_filename: Optional[str] = None

    class Config:
        orm_mode = True

class ModelOut(BaseModel):
    id: int
    task_id: int
    url: str
    class Config:
        orm_mode = True
 