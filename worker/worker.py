import os
import sys
import json
import time
import requests
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import redis

# --- Получаем shard_id и total_shards из аргументов командной строки или переменных окружения ---
if len(sys.argv) >= 3:
    shard_id = int(sys.argv[1])
    total_shards = int(sys.argv[2])
else:
    total_shards = int(os.getenv('TOTAL_WORKERS', '1'))
    shard_id = int(os.getenv('SHARD_ID', '0'))

# --- Настройки Redis ---
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url)

# --- Основной цикл обработки задач ---
while True:
    # shard_id==0 берёт задачу из очереди и кладёт в current_task
    if shard_id == 0:
        print('[LOG] Ожидание задачи в очереди Redis...')
        task_data = redis_client.blpop('training_tasks', timeout=0)
        if not task_data:
            print('[LOG] Нет задач в очереди. Завершение.')
            break
        task = json.loads(task_data[1].decode())
        redis_client.set('current_task', json.dumps(task))
        print(f'[LOG] Задача помещена в current_task: {task}')
    else:
        # Остальные воркеры ждут появления задачи в current_task
        while True:
            task_data = redis_client.get('current_task')
            if task_data:
                break
            print('[LOG] Ожидание задачи в current_task...')
            time.sleep(1)
        task = json.loads(task_data.decode())
        print(f'[LOG] Получена задача из current_task: {task}')

    # --- Параметры задачи ---
    model_type = task.get('model_type', 'distilbert')
    dataset_url = task.get('dataset_url')
    hyperparams = task.get('hyperparameters', {})
    epochs = hyperparams.get('epochs', 2)
    batch_size = hyperparams.get('batch_size', 32)
    task_id = task.get('id', 0)

    # --- Скачиваем датасет ---
    print(f'[LOG] Скачивание датасета: {dataset_url}')
    response = requests.get(dataset_url)
    data = json.loads(response.text)

    # --- Разбиваем датасет на шарды ---
    shard_size = len(data) // total_shards
    start_idx = shard_id * shard_size
    end_idx = start_idx + shard_size if shard_id < total_shards - 1 else len(data)
    worker_data = data[start_idx:end_idx]
    print(f"[LOG] Этот воркер обучается на shard {shard_id+1} из {total_shards}, размер: {len(worker_data)}")

    class CustomDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            encoding = self.tokenizer(
                item['text'],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(item['label'], dtype=torch.long)
            }

    def average_weights(weight_files):
        avg_state_dict = None
        for i, file in enumerate(weight_files):
            state_dict = torch.load(file)
            if avg_state_dict is None:
                avg_state_dict = {k: v.clone() for k, v in state_dict.items()}
            else:
                for k in avg_state_dict:
                    avg_state_dict[k] += state_dict[k]
        for k in avg_state_dict:
            avg_state_dict[k] /= len(weight_files)
        return avg_state_dict

    def wait_for_all_weights(weight_files, timeout=300):
        start = time.time()
        while True:
            if all(os.path.exists(f) for f in weight_files):
                return
            if time.time() - start > timeout:
                raise TimeoutError("Не дождались всех файлов весов!")
            time.sleep(1)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = CustomDataset(worker_data, tokenizer)
    print(f"[LOG] Размер датасета: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f'[LOG] Начало эпохи {epoch+1}')
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            print(f'[LOG] Batch {batch_idx+1}: loss={loss.item()}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f'[LOG] Конец эпохи {epoch+1}')
        # --- Синхронизация весов через файлы (можно заменить на Redis) ---
        weight_file = f"weights_shard_{shard_id}.pt"
        tmp_file = weight_file + ".tmp"
        torch.save(model.state_dict(), tmp_file)
        os.replace(tmp_file, weight_file)
        print(f"[LOG] Веса сохранены в {weight_file}")
        weight_files = [f"weights_shard_{i}.pt" for i in range(total_shards)]
        wait_for_all_weights(weight_files)
        print("[LOG] Все веса получены, начинаем усреднение...")
        avg_state_dict = average_weights(weight_files)
        model.load_state_dict(avg_state_dict)
        print("[LOG] Веса усреднены и загружены")
    print('[LOG] Обучение завершено успешно')

    # Финальная синхронизация и сохранение итоговой модели
    weight_files = [f"weights_shard_{i}.pt" for i in range(total_shards)]
    wait_for_all_weights(weight_files)
    print("[LOG] Финальная синхронизация весов...")
    avg_state_dict = average_weights(weight_files)
    model.load_state_dict(avg_state_dict)
    if shard_id == 0:
        torch.save(model.state_dict(), "final_model.pt")
        print("[LOG] Итоговая модель сохранена в final_model.pt")

        # --- Обновляем статус задачи через API ---
        api_url = os.getenv("API_URL", "http://localhost:8000/api")
        try:
            response = requests.patch(
                f"{api_url}/tasks/{task_id}/status",
                json={"status": "done"},
                headers={"Content-Type": "application/json"}
            )
            print(f"[LOG] Статус задачи {task_id} обновлён: {response.status_code} {response.text}")
        except Exception as e:
            print(f"[LOG] Не удалось обновить статус задачи: {e}")

        # --- Очищаем ключ задачи в Redis ---
        redis_client.delete('current_task')
        print("[LOG] Ключ current_task удалён из Redis")

    # --- Удаляем временные файлы весов ---
    for f in weight_files:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists("final_model.pt") and shard_id != 0:
        os.remove("final_model.pt")
    print("[LOG] Временные файлы весов удалены")

    # shard_id==0 ждёт, пока все воркеры удалят свои файлы, чтобы не начать новую задачу раньше времени
    if shard_id == 0:
        time.sleep(2)

    # ВСЕ воркеры ждут, пока ключ current_task будет удалён
    while redis_client.get('current_task'):
        time.sleep(0.5)
    print(f"[LOG] Воркер {shard_id} увидел, что current_task удалён, ждёт новую задачу...")