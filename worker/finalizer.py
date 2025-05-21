import os
import time
import torch
import redis
import requests
import re
from supabase import create_client, Client
from typing import Optional

# --- Настройки ---
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
API_URL = os.getenv('API_URL', 'http://localhost:8000/api')
CHECK_INTERVAL = int(os.getenv('FINALIZER_CHECK_INTERVAL', 10))  # секунд между проверками
# Директория, куда воркеры сохраняют веса шардов (должна совпадать с настройкой воркера)
WEIGHTS_STORAGE_DIR = os.getenv('WEIGHTS_STORAGE_DIR', '/tmp/trainnet_weights')
# Директория для сохранения итоговых моделей
FINAL_MODELS_DIR = os.getenv('FINAL_MODELS_DIR', 'final_models') 
os.makedirs(FINAL_MODELS_DIR, exist_ok=True) # Создаем директорию, если ее нет

# --- Инициализация клиента Supabase ---
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
supabase_client: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("[FINALIZER] WARNING: SUPABASE_URL и SUPABASE_KEY не установлены. Загрузка в Supabase будет пропущена.")
# --- Конец инициализации Supabase ---

redis_client = redis.from_url(REDIS_URL)

def average_weights(state_dicts_list):
    """Усредняет веса моделей (список PyTorch state_dict)"""
    if not state_dicts_list:
        return None
    
    avg_state_dict = None
    num_models = len(state_dicts_list)

    # Инициализация avg_state_dict первым state_dict
    # Клонируем, чтобы не изменять оригинальный state_dict
    avg_state_dict = {k: v.clone().float() for k, v in state_dicts_list[0].items()}

    # Добавляем веса остальных моделей
    for i in range(1, num_models):
        current_state_dict = state_dicts_list[i]
        for k in avg_state_dict:
            if k in current_state_dict:
                avg_state_dict[k] += current_state_dict[k].float()
            else:
                print(f"[AVERAGE_WEIGHTS] Warning: Key {k} not found in one of the state_dicts.")

    # Усредняем
    for k in avg_state_dict:
        avg_state_dict[k] /= num_models
        
    return avg_state_dict

def get_expected_num_shards(task_id):
    num_shards_key = f"task_{task_id}_num_shards"
    num_shards_raw = redis_client.get(num_shards_key)
    if num_shards_raw:
        try:
            return int(num_shards_raw)
        except ValueError:
            print(f"[FINALIZER] {task_id}: Invalid value for num_shards in Redis: {num_shards_raw.decode()}. Cannot process.")
            return 0 # Возвращаем 0, чтобы задача не обрабатывалась
    else:
        print(f"[FINALIZER] {task_id}: Key {num_shards_key} not found. Cannot determine expected number of shards.")
        return 0 # Возвращаем 0, если ключ отсутствует

def finalize_task(task_id, num_shards):
    results_key = f"task_{task_id}_results"
    # Получаем словарь {shard_id: path_to_weights_file}
    shard_paths_map_raw = redis_client.hgetall(results_key)

    if len(shard_paths_map_raw) < num_shards:
        print(f"[FINALIZER] {task_id}: Готово {len(shard_paths_map_raw)}/{num_shards} шардов (пути к весам). Жду...")
        return False

    print(f"[FINALIZER] {task_id}: Все {num_shards} шардов сообщили пути к весам. Загружаю и усредняю веса...")
    
    loaded_state_dicts = []
    all_files_found = True
    shard_ids_processed = set()

    for shard_id_bytes, path_bytes in shard_paths_map_raw.items():
        try:
            shard_id_str = shard_id_bytes.decode()
            shard_ids_processed.add(shard_id_str)
            weights_file_path = path_bytes.decode()

            if not os.path.exists(weights_file_path):
                print(f"[FINALIZER] {task_id}: Ошибка! Файл весов {weights_file_path} для шарда {shard_id_str} не найден. Пропускаю финализацию этой задачи.")
                all_files_found = False
                break # Прерываем загрузку для этой задачи
            
            # Загружаем state_dict, явно указывая map_location
            # Это важно, если модель обучалась на GPU, а финализатор работает на CPU (или наоборот)
            state_dict = torch.load(weights_file_path, map_location='cpu')
            loaded_state_dicts.append(state_dict)
            print(f"[FINALIZER] {task_id}: Веса для шарда {shard_id_str} загружены из {weights_file_path}")

        except Exception as e:
            print(f"[FINALIZER] {task_id}: Ошибка при загрузке весов для шарда {shard_id_bytes.decode()} из файла {path_bytes.decode()}: {e}")
            all_files_found = False # Считаем это ошибкой, требующей вмешательства
            break

    if not all_files_found:
        return False # Финализация не удалась из-за отсутствия файлов или ошибок загрузки
        
    # Проверка, что мы обработали все ожидаемые шарды.
    # Это полезно, если в Redis для какого-то шарда не оказалось пути.
    if len(loaded_state_dicts) != num_shards:
        print(f"[FINALIZER] {task_id}: Ожидалось {num_shards} наборов весов, но загружено {len(loaded_state_dicts)}. Пропускаю финализацию.")
        # Можно добавить логирование shard_ids_processed для отладки
        return False

    if not loaded_state_dicts:
        print(f"[FINALIZER] {task_id}: Нет загруженных весов для усреднения. Пропускаю.")
        return False

    avg_state_dict = average_weights(loaded_state_dicts)
    if avg_state_dict is None:
        print(f"[FINALIZER] {task_id}: Не удалось усреднить веса (результат None).")
        return False

    model_file_name = f"final_model_task_{task_id}.pt"
    model_path = os.path.join(FINAL_MODELS_DIR, model_file_name)
    
    try:
        torch.save(avg_state_dict, model_path)
        print(f"[FINALIZER] {task_id}: Итоговая модель сохранена в {model_path}")
    except Exception as e:
        print(f"[FINALIZER] {task_id}: Ошибка при сохранении итоговой модели в {model_path}: {e}")
        return False # Не удалось сохранить модель, не продолжаем

    # --- Загрузка модели в Supabase Storage (ЗАКОММЕНТИРОВАНО ДЛЯ MVP) ---
    model_public_url = None
    # if supabase_client:
    #     try:
    #         file_ext = model_file_name.split(".")[-1]
    #         supabase_file_path = f"final_models/task_{task_id}/{model_file_name}"
            
    #         with open(model_path, 'rb') as f:
    #             # Удаляем существующий файл, если он есть, чтобы избежать ошибки при upsert=False (если потребуется)
    #             # supabase_client.storage.from_("trainnet-models").remove([supabase_file_path]) # Опционально
                
    #             # Загружаем файл
    #             upload_response = supabase_client.storage.from_("trainnet-models").upload(
    #                 path=supabase_file_path,
    #                 file=f,
    #                 file_options={"cache_control": "3600", 
    #                               "upsert": "true", # Изменено с True на "true"
    #                               "content_type": "application/octet-stream"}
    #             )
    #         print(f"[FINALIZER] {task_id}: Ответ от Supabase upload: {upload_response}")

    #         # Получаем публичный URL
    #         url_response = supabase_client.storage.from_("trainnet-models").get_public_url(supabase_file_path)
    #         model_public_url = url_response
    #         print(f"[FINALIZER] {task_id}: Модель загружена в Supabase. Public URL: {model_public_url}")
    #     except Exception as e:
    #         print(f"[FINALIZER] {task_id}: Ошибка при загрузке модели в Supabase Storage: {e}")
    #         # Для MVP ошибка загрузки в Supabase не должна останавливать процесс, если локально сохранено
    #         print(f"[FINALIZER] {task_id}: Продолжение работы без загрузки в Supabase (MVP режим).")
    # else:
    #     print(f"[FINALIZER] {task_id}: Клиент Supabase не инициализирован, загрузка модели пропущена.")
    # --- Конец загрузки в Supabase ---

    # Попытка обновить статус в API
    try:
        payload = {"status": "done", "result_model_filename": model_file_name}
        # model_public_url здесь не добавляется, так как он None и мы его не используем для MVP
        
        response = requests.patch(
            f"{API_URL}/tasks/{task_id}/status",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"[FINALIZER] {task_id}: Статус обновлён через API: {response.status_code} {response.text}")
        if response.status_code >= 300: # Если API вернул ошибку
             print(f"[FINALIZER] {task_id}: API вернул ошибку при обновлении статуса. Временные файлы не будут удалены.")
             return False # Не удаляем файлы, если API не подтвердил
    except requests.exceptions.RequestException as e:
        print(f"[FINALIZER] {task_id}: Не удалось обновить статус через API: {e}. Временные файлы не будут удалены.")
        return False # Не удаляем файлы, если API недоступен

    # Удаление временных файлов весов шардов, если все прошло успешно
    print(f"[FINALIZER] {task_id}: Удаление временных файлов весов шардов...")
    deleted_count = 0
    errors_deleting = False
    for shard_id_bytes, path_bytes in shard_paths_map_raw.items():
        weights_file_path = path_bytes.decode()
        try:
            if os.path.exists(weights_file_path):
                os.remove(weights_file_path)
                deleted_count +=1
                # print(f"[FINALIZER] {task_id}: Удален файл {weights_file_path}")
        except OSError as e:
            print(f"[FINALIZER] {task_id}: Ошибка при удалении файла {weights_file_path}: {e}")
            errors_deleting = True # Отмечаем, что были ошибки, но продолжаем удалять остальные
    
    if errors_deleting:
        print(f"[FINALIZER] {task_id}: Были ошибки при удалении некоторых временных файлов весов.")
    else:
        print(f"[FINALIZER] {task_id}: Успешно удалено {deleted_count} временных файлов весов.")

    # Удаление ключей из Redis
    try:
        redis_client.delete(results_key)
        redis_client.delete(f"task_{task_id}_shards") # Очередь задач
        redis_client.delete(f"task_{task_id}_num_shards") # Количество шардов
        print(f"[FINALIZER] {task_id}: Временные ключи Redis удалены.")
    except redis.exceptions.RedisError as e:
        print(f"[FINALIZER] {task_id}: Ошибка при удалении ключей из Redis: {e}")
        # Это не критично для уже сохраненной модели, но стоит залогировать.

    # Опционально: если задача подразумевает создание уникальной директории для весов каждого задания
    # task_specific_weights_dir = os.path.join(WEIGHTS_STORAGE_DIR, f"task_{task_id}")
    # if os.path.isdir(task_specific_weights_dir):
    #     try:
    #         shutil.rmtree(task_specific_weights_dir)
    #         print(f"[FINALIZER] {task_id}: Удалена директория с временными весами: {task_specific_weights_dir}")
    #     except OSError as e:
    #         print(f"[FINALIZER] {task_id}: Ошибка при удалении директории {task_specific_weights_dir}: {e}")

    return True

def main():
    global redis_client, supabase_client # Объявляем global здесь, так как redis_client может быть переприсвоен
    print(f"[FINALIZER] Автоматический режим запущен. Интервал проверки: {CHECK_INTERVAL}с.")
    print(f"[FINALIZER] Директория для весов шардов (ожидаемая от воркеров): {WEIGHTS_STORAGE_DIR}")
    print(f"[FINALIZER] Директория для итоговых моделей: {FINAL_MODELS_DIR}")
    
    while True:
        processed_task_ids_in_cycle = set()
        try:
            # Ищем ключи, соответствующие результатам задач
            keys = redis_client.keys("task_*_results") 
            if not keys:
                # print("[FINALIZER] Нет активных задач для финализации. Жду...")
                pass

            for key_bytes in keys:
                key_str = key_bytes.decode()
                match = re.match(r"task_(\d+)_results", key_str)
                if not match:
                    # print(f"[FINALIZER] Ключ {key_str} не соответствует шаблону task_X_results, пропускаю.")
                    continue
                
                task_id_str = match.group(1)
                try:
                    task_id = int(task_id_str)
                except ValueError:
                    print(f"[FINALIZER] Некорректный task_id '{task_id_str}' из ключа {key_str}. Пропускаю.")
                    continue
                
                if task_id in processed_task_ids_in_cycle:
                    # print(f"[FINALIZER] Задача {task_id} уже была обработана в этом цикле. Пропускаю.")
                    continue

                # print(f"[FINALIZER] Обнаружена задача {task_id} для возможной финализации.")
                num_shards = get_expected_num_shards(task_id)
                
                if num_shards > 0:
                    # print(f"[FINALIZER] {task_id}: Ожидается {num_shards} шардов.")
                    if finalize_task(task_id, num_shards):
                        print(f"[FINALIZER] {task_id}: Финализация успешно завершена.")
                        processed_task_ids_in_cycle.add(task_id)
                    # else:
                        # print(f"[FINALIZER] {task_id}: Финализация еще не завершена или произошла ошибка.")
                # else:
                    # print(f"[FINALIZER] {task_id}: Не удалось определить количество шардов или оно равно 0. Пропускаю финализацию.")
            
        except redis.exceptions.ConnectionError as e:
            print(f"[FINALIZER] Ошибка соединения с Redis: {e}. Попробую переподключиться.")
            try:
                # global redis_client # Удаляем global отсюда, он уже объявлен в начале функции
                redis_client = redis.from_url(REDIS_URL)
                redis_client.ping()
                print("[FINALIZER] Переподключение к Redis успешно.")
            except redis.exceptions.ConnectionError as e_conn:
                print(f"[FINALIZER] Не удалось переподключиться к Redis: {e_conn}. Следующая попытка через {CHECK_INTERVAL}с.")
        except Exception as e:
            print(f"[FINALIZER] Непредвиденная ошибка в основном цикле: {e}")
            import traceback
            traceback.print_exc() # Для детальной отладки
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main() 