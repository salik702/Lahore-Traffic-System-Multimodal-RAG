from contextlib import asynccontextmanager
import asyncio
import logging
import os
import socket
import subprocess

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api.simulation import router as simulation_router
from app.core.config import get_settings
from app.rag.engine import RAGAnythingEngine, local_embed
from app.services.chat import ChatService
from app.services.database import ConversationStore
from app.services.repository import KnowledgeRepository
from app.simulation.manager import TwinManager


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    worker_env = os.environ.copy()
    worker_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    with socket.socket() as probe:
        semantic_worker_running = probe.connect_ex(("127.0.0.1", 8765)) == 0
    worker = None if semantic_worker_running else subprocess.Popen([settings.semantic_python_executable, str(settings.project_root / "backend" / "embedding_worker.py")], cwd=settings.project_root / "backend", env=worker_env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    repository = KnowledgeRepository(settings)
    store = ConversationStore(settings.database_path)
    twin = TwinManager(settings)
    rag = RAGAnythingEngine(settings)
    await rag.initialize()
    if rag.ready:
        try:
            # Cold imports and model loading can be slow on Windows. Allow the
            # worker to finish without applying this longer timeout to queries.
            vectors = await asyncio.wait_for(
                local_embed(["Punjab traffic regulations"], timeout=settings.semantic_startup_timeout),
                timeout=settings.semantic_startup_timeout,
            )
            if vectors.shape[1] != settings.embedding_dimension:
                raise ValueError("Embedding worker does not match the stored index")
        except Exception:
            rag.error = "Semantic embedding worker unavailable or incompatible; indexed retrieval is disabled"
            logging.getLogger(__name__).exception("Embedding warm-up failed")
    app.state.settings = settings
    app.state.repository = repository
    app.state.store = store
    app.state.rag = rag
    app.state.twin = twin
    app.state.chat = ChatService(settings, repository, rag, store, twin)
    yield
    await twin.pause()
    if worker and worker.poll() is None:
        worker.terminate()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Source-grounded multimodal traffic intelligence for Lahore and Punjab.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(router, prefix=settings.api_prefix)
app.include_router(simulation_router, prefix=settings.api_prefix)
app.mount("/media/signs", StaticFiles(directory=settings.data_dir / "raw" / "traffic_signs"), name="signs")
app.mount("/media/images", StaticFiles(directory=settings.data_dir / "raw" / "images"), name="images")
app.mount("/media/accidents", StaticFiles(directory=settings.data_dir / "raw" / "accidents" / "images"), name="accident-images")



@app.get("/")
async def root():
    return {"name": settings.app_name, "docs": "/docs", "health": f"{settings.api_prefix}/health"}
