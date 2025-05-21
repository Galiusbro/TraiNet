import os
import time
import json
import redis
import requests
import torch
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Настройка Redis
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
redis_client = redis.from_url(redis_url)

# Директория для сохранения файлов весов
WEIGHTS_STORAGE_DIR = os.getenv('WEIGHTS_STORAGE_DIR', '/tmp/trainnet_weights')
os.makedirs(WEIGHTS_STORAGE_DIR, exist_ok=True) # Создаем директорию, если ее нет

def safe_hset(key, field, value, max_retries=3):
    # This function now uses its own local Redis client for write attempts
    # and no longer modifies the global redis_client.
    last_exception = None
    for attempt in range(max_retries):
        try:
            # Create a new local Redis client for this attempt to ensure freshness
            local_r = redis.from_url(redis_url)
            local_r.hset(key, field, value)
            print(f"[LOG] Successfully wrote to Redis: key='{key}', field='{field}' on attempt {attempt+1}")
            return # Success
        except redis.exceptions.ConnectionError as e:
            last_exception = e
            print(f"[LOG] Redis HSET connection error (attempt {attempt+1}/{max_retries}) for key='{key}': {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt  # Exponential backoff: 1s, 2s for 3 retries (total 3 attempts)
                print(f"[LOG] Retrying HSET in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                print(f"[LOG] Failed to write to Redis for key='{key}' after {max_retries} attempts.")
    # If loop finishes, all retries failed. Raise the last known exception.
    raise Exception(f"Failed to write to Redis (key='{key}') after {max_retries} attempts") from last_exception

def process_shard(shard_info):
    # Скачиваем датасет
    response = requests.get(shard_info["dataset_url"])
    data = response.json() if response.headers.get("content-type") == "application/json" else json.loads(response.text)
    # Берём только свой кусок
    shard_data = data[shard_info["start"]:shard_info["end"]]
    print(f"[LOG] Обрабатываю шард {shard_info['shard_id']} ({shard_info['start']}:{shard_info['end']}) из задачи {shard_info['task_id']}")

    # Пример: обучение на DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    # Преобразуем данные в датасет
    class CustomDataset(torch.utils.data.Dataset):
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

    dataset = CustomDataset(shard_data, tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=shard_info["hyperparameters"].get("batch_size", 32), shuffle=True, num_workers=0)

    epochs = shard_info["hyperparameters"].get("epochs", 2)
    for epoch in range(epochs):
        print(f'[LOG] Эпоха {epoch+1} для шарда {shard_info["shard_id"]}')
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
    print(f'[LOG] Обучение шарда {shard_info["shard_id"]} завершено')

    # Сохраняем веса модели в файл
    try:
        weights_file_name = f"task_{shard_info['task_id']}_shard_{shard_info['shard_id']}_weights.pt"
        weights_file_path = os.path.join(WEIGHTS_STORAGE_DIR, weights_file_name)
        
        # Убедимся, что директория существует перед сохранением (на всякий случай, если была удалена после старта воркера)
        os.makedirs(WEIGHTS_STORAGE_DIR, exist_ok=True)
        
        torch.save(model.state_dict(), weights_file_path)
        print(f"[LOG] Веса для шарда {shard_info['shard_id']} сохранены в файл: {weights_file_path}")

        # Сохраняем путь к файлу весов в Redis
        safe_hset(f"task_{shard_info['task_id']}_results", str(shard_info["shard_id"]), weights_file_path)
        print(f"[LOG] Путь к файлу весов для шарда {shard_info['shard_id']} сохранен в Redis.")

    except Exception as e:
        print(f"[ERROR] Failed to save weights to file or write path to Redis for shard {shard_info['shard_id']}: {e}")
        # В зависимости от требований, можно либо пробросить исключение дальше,
        # либо пометить шард как ошибочный в Redis, и т.д.
        # Пока просто логируем и продолжаем, чтобы воркер не падал из-за ошибки одного шарда.
        # Для критичных задач, здесь должно быть более строгое управление ошибками.
        # Например, можно попробовать удалить частично сохраненный файл, если он есть.
        # if os.path.exists(weights_file_path):
        #     try:
        #         os.remove(weights_file_path)
        #         print(f"[LOG] Partially saved weights file {weights_file_path} removed due to error.")
        #     except OSError as oe:
        #         print(f"[ERROR] Could not remove partially saved file {weights_file_path}: {oe}")
        raise # Перевыбрасываем ошибку, чтобы она была видна в main loop и обработана там

def main():
    global redis_client # Indicate that we are using and potentially modifying the global redis_client
    print("[LOG] Универсальный воркер запущен")

    connection_errors_in_a_row = 0
    max_consecutive_connection_errors = 5 # Threshold before longer sleep

    while True:
        try:
            # Attempt to refresh the global redis_client connection at the start of each main loop iteration.
            current_r_client = redis.from_url(redis_url)
            current_r_client.ping() # Test the connection
            redis_client = current_r_client # Assign to global if ping is successful
            # print("[LOG] Global Redis connection refreshed and PING successful for main loop.")
            connection_errors_in_a_row = 0 # Reset error counter on success

            all_keys = redis_client.keys("task_*_shards")
            # Фильтруем только списки
            shard_queues = [key.decode() for key in all_keys if redis_client.type(key) == b'list']
            if not shard_queues:
                print("[LOG] Нет активных очередей задач, жду...")
                time.sleep(5)
                continue

            # print(f"[LOG] Attempting to BLPOP from queues: {shard_queues}")
            shard = redis_client.blpop(shard_queues, timeout=10)
            if shard:
                queue_name, shard_data = shard
                shard_info = json.loads(shard_data.decode())
                print(f"[LOG] Взял шард из {queue_name.decode()}")
                process_shard(shard_info)
            else:
                print("[LOG] Нет работы (BLPOP timeout), жду...")
                # No need to sleep here if blpop timed out, loop will restart, effectively waiting.
        except redis.exceptions.ConnectionError as e:
            connection_errors_in_a_row += 1
            print(f"[LOG] Redis connection error in main loop (attempt {connection_errors_in_a_row}): {e}.")
            sleep_duration = 5 * connection_errors_in_a_row # Increase sleep time on repeated errors
            if connection_errors_in_a_row > max_consecutive_connection_errors:
                print(f"[LOG] Too many consecutive Redis connection errors in main loop. Sleeping for 60s.")
                sleep_duration = 60
                # Reset counter after long sleep, or it will immediately trigger long sleep again
                connection_errors_in_a_row = 0 
            print(f"[LOG] Retrying main loop operations in {sleep_duration}s...")
            time.sleep(sleep_duration)
        except Exception as e:
            print(f"[ERROR] Unexpected error in main loop: {e}")
            # import traceback # For more detailed debugging if needed
            # print(traceback.format_exc())
            print(f"[LOG] Restarting main loop after unexpected error in 10s...")
            time.sleep(10)

if __name__ == "__main__":
    main() 