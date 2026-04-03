from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router

app = FastAPI(
    title="Multimodal Medical Triage API",
    version="1.0.0",
)

ALLOWED_ORIGINS = [
    "https://triage.rhutammahajan.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Backend is running",
        "health": "/health",
        "docs": "/docs",
        "predict_routes": [
            "/api/predict",
            "/predict",
            "/api/analyze",
            "/analyze",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}