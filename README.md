# TrainNet.ai — MVP

## Структура проекта

- `backend/` — FastAPI backend (API, очередь, работа с Supabase)
- `worker/` — Python-агент для обучения моделей
- `frontend/` — минималистичный React-интерфейс
- `docker-compose.yml` — запуск всех сервисов локально

## Быстрый старт

1. Установи Docker и Docker Compose
2. Запусти:
   ```bash
   docker-compose up --build
   ```
3. Перейди на http://localhost:3000 (frontend)

---

Документация и ТЗ — в папке `!Doc/`.
