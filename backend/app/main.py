from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Tabular ML Service")
app.include_router(router)
