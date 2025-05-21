import os
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from supabase import create_client, Client
from schemas import TaskCreate, TaskOut, ModelOut, TaskStatus
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Any
from pydantic import BaseModel
import redis
import json
import math
import requests

# Загрузка переменных окружения
load_dotenv()

SUPABASE_URL: str = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL и SUPABASE_KEY должны быть установлены в .env")

# Инициализация клиента Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Настройка Redis
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url)

# FastAPI приложение
app = FastAPI()

# Директория для хранения финальных моделей на API сервере
# Убедитесь, что эта переменная указывает на ту же директорию,
# куда finalizer.py сохраняет модели, или API имеет к ней доступ.
FINAL_MODELS_DIR = os.getenv('FINAL_MODELS_DIR', 'final_models')

# Создаем роутер с префиксом /api
router = APIRouter(prefix="/api")

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене замените на конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskStatusUpdate(BaseModel):
    status: TaskStatus
    error: Optional[str] = None
    result_model_filename: Optional[str] = None

@router.get("/")
def root():
    return {"message": "TrainNet.ai backend is running"}

@router.post("/train", response_model=TaskOut)
def create_task(task: TaskCreate):
    try:
        data = task.model_dump() if hasattr(task, 'model_dump') else task.dict()
        response = supabase.table("tasks").insert(data).execute()
        print("Ответ Supabase:", response)
        
        response_data = response.model_dump()
        if response_data.get("error"):
            raise HTTPException(status_code=500, detail=str(response_data["error"]))
        
        task_data_raw = response_data["data"][0]
        # Преобразуем в TaskOut, чтобы убедиться, что все поля на месте, включая опциональные
        task_data = TaskOut(**task_data_raw).model_dump()
        task_id = task_data["id"]

        # --- Разбиваем задачу на шарды ---
        dataset_url = task_data["dataset_url"]
        hyperparams = task_data.get("hyperparams", {})
        num_shards = hyperparams.get("num_shards", 2)  # Можно передавать с фронта или считать по размеру датасета

        # Пример: скачиваем датасет и делим на чанки
        try:
            response = requests.get(dataset_url)
            dataset = response.json() if response.headers.get("content-type") == "application/json" else json.loads(response.text)
        except Exception as e:
            print(f"Ошибка при скачивании датасета: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download dataset: {e}")
        total = len(dataset)
        shard_size = math.ceil(total / num_shards)

        for i in range(num_shards):
            shard_info = {
                "task_id": task_id,
                "shard_id": i,
                "start": i * shard_size,
                "end": min((i + 1) * shard_size, total),
                "model_type": task_data["model_type"],
                "dataset_url": dataset_url,
                "hyperparameters": hyperparams
            }
            redis_client.rpush(f"task_{task_id}_shards", json.dumps(shard_info))

        # Сохраняем количество шардов для финализатора
        redis_client.set(f"task_{task_id}_num_shards", num_shards)

        print(f"Задача {task_id} разбита на {num_shards} шардов и добавлена в очередь task_{task_id}_shards")

        return task_data
    except Exception as e:
        print(f"Ошибка при создании задачи: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks", response_model=list[TaskOut])
def get_tasks():
    response = supabase.table("tasks").select("*").execute()
    response_data = response.model_dump()
    if response_data.get("error"):
        raise HTTPException(status_code=500, detail=str(response_data["error"]))
    # Преобразуем каждый элемент в TaskOut для консистентности
    return [TaskOut(**task).model_dump() for task in response_data["data"]]

@router.get("/tasks/{task_id}", response_model=TaskOut)
def get_task(task_id: int):
    response = supabase.table("tasks").select("*").eq("id", task_id).single().execute()
    response_data = response.model_dump()
    if response_data.get("error"):
        raise HTTPException(status_code=500, detail=str(response_data["error"]))
    if not response_data.get("data"):
        raise HTTPException(status_code=404, detail="Task not found")
    # Преобразуем в TaskOut и возвращаем как dict для соответствия response_model
    return TaskOut(**response_data["data"]).model_dump()

@router.get("/tasks/{task_id}/status", response_model=TaskOut)
def get_task_status(task_id: int):
    response = supabase.table("tasks").select("*").eq("id", task_id).single().execute()
    response_data = response.model_dump()
    if response_data.get("error"):
        raise HTTPException(status_code=500, detail=str(response_data["error"]))
    if not response_data.get("data"):
        raise HTTPException(status_code=404, detail="Task not found")
    # Преобразуем в TaskOut и возвращаем как dict для соответствия response_model
    return TaskOut(**response_data["data"]).model_dump()

@router.get("/models", response_model=list[ModelOut])
def get_models():
    response = supabase.table("models").select("*").execute()
    response_data = response.model_dump()
    if response_data.get("error"):
        raise HTTPException(status_code=500, detail=str(response_data["error"]))
    return response_data["data"]

@router.get("/models/{model_id}", response_model=ModelOut)
def get_model(model_id: int):
    response = supabase.table("models").select("*").eq("id", model_id).execute()
    response_data = response.model_dump()
    if response_data.get("error"):
        raise HTTPException(status_code=500, detail=str(response_data["error"]))
    if not response_data.get("data"):
        raise HTTPException(status_code=404, detail="Model not found")
    return response_data["data"][0]

@router.post("/webhook/stripe")
def stripe_webhook():
    # Заглушка для MVP
    return JSONResponse({"status": "ok"})

@router.patch("/tasks/{task_id}/status")
def update_task_status(task_id: int, status_update: TaskStatusUpdate):
    update_data = {"status": status_update.status.value}
    if status_update.error:
        update_data["error_message"] = status_update.error
    if status_update.result_model_filename:
        update_data["result_model_filename"] = status_update.result_model_filename

    response = supabase.table("tasks").update(update_data).eq("id", task_id).execute()
    
    response_data = response.model_dump()
    if response_data.get("error"):
        raise HTTPException(status_code=500, detail=str(response_data["error"]))
    if not response_data.get("data"):
        raise HTTPException(status_code=404, detail="Task not found")
    # Возвращаем обновленные данные задачи, преобразованные в TaskOut
    return TaskOut(**response_data["data"][0]).model_dump()

@router.get("/tasks/{task_id}/download_model")
def download_model(task_id: int):
    # Получаем информацию о задаче из Supabase
    task_response = supabase.table("tasks").select("result_model_filename").eq("id", task_id).single().execute()
    
    task_response_data = task_response.model_dump()

    if task_response_data.get("error"):
        raise HTTPException(status_code=500, detail=f"Error fetching task: {task_response_data['error']}")
    
    task_data = task_response_data.get("data")
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    model_filename = task_data.get("result_model_filename")
    if not model_filename:
        raise HTTPException(status_code=404, detail="Model filename not found for this task or task not completed with a model.")
    
    # Задаем абсолютный путь к директории с моделями (worker/final_models)
    FINAL_MODELS_DIR = os.getenv("FINAL_MODELS_DIR", "/Users/as/projects/ayaal/TrainNet/worker/final_models")
    
    # Полный путь к файлу модели
    file_path = os.path.join(FINAL_MODELS_DIR, model_filename)
    
    print(f"Пытаемся получить доступ к файлу модели: {file_path}")
    
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        # Логирование для отладки
        print(f"ОШИБКА: Файл {file_path} не найден.")
        print(f"Текущая директория: {os.getcwd()}")
        print(f"Содержимое директории {FINAL_MODELS_DIR} (если существует):")
        try:
            if os.path.exists(FINAL_MODELS_DIR):
                print(os.listdir(FINAL_MODELS_DIR))
            else:
                print(f"Директория {FINAL_MODELS_DIR} не существует")
        except Exception as e:
            print(f"Ошибка при чтении директории: {e}")
            
        raise HTTPException(status_code=404, detail=f"Model file '{model_filename}' not found on server. Check FINAL_MODELS_DIR configuration and if finalizer saved the file correctly.")
    
    # Возвращаем файл для скачивания
    return FileResponse(path=file_path, filename=model_filename, media_type='application/octet-stream')

# Подключаем роутер к приложению
app.include_router(router)
