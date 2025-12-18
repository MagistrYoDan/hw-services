from fastapi import FastAPI
from app.routes.forward import router as forward_router
from app.routes.metadata import router as metadata_router

app = FastAPI(title="ML Tabular Service")

app.include_router(forward_router)
app.include_router(metadata_router)