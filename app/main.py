from fastapi import FastAPI
from app.api.cv_controller import router as cv_router

app = FastAPI(title="CV Analysis Service")

app.include_router(cv_router)
