Repository for services project

# ML Backend Service

Сервис предоставляет HTTP API для получения предсказаний ML-модели.

## 🚀 Запуск

### 1. Установка зависимостей и запуск сервера

В директории backend:
docker-compose up --build 

### 2. Примеры обращений к серверу:

1) curl -X POST http://localhost:8000/forward \
  -H "Content-Type: application/json" \
  -d '{
        "features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
      }'

2) curl -X POST http://localhost:8000/forward_batch \
  -H "Content-Type: application/json" \
  -d '{
        "features": [                                        
          [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
          [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
        ]
      }'


3) curl -X POST http://localhost:8000/evaluate \     
  -F "file=@example.csv" 
