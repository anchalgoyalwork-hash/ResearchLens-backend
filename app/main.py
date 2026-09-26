from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.search import router as search_router


app = FastAPI(
    title="ResearchLens API",
    description="AI-powered scientific research assistant",
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(search_router)


@app.get("/")
def root():
    return {
        "app": "ResearchLens",
        "message": "ResearchLens backend is running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }