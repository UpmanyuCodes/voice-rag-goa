"""FastAPI Backend — Voice-Enabled RAG API Server."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
from pathlib import Path

from src.harness.orchestrator import PipelineOrchestrator
from src.retrieval.cache import SemanticQueryCache

# Shared orchestrator instance (initialized on startup)
_orchestrator: PipelineOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the pipeline on startup."""
    global _orchestrator
    _orchestrator = PipelineOrchestrator()
    yield
    # Cleanup on shutdown
    if _orchestrator:
        _orchestrator.vector_store.clear()


app = FastAPI(
    title="Voice-Enabled RAG — HH Goa 2026",
    description="Multilingual Voice-Enabled RAG System powered by ai4bharat/MSMARCO-XI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register routes
from src.api.routes import router
app.include_router(router)

# Serve the web UI from /web
WEB_DIR = Path(__file__).parent.parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(WEB_DIR / "index.html"))


def get_orchestrator() -> PipelineOrchestrator:
    return _orchestrator
