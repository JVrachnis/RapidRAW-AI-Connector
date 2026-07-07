import logging
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from engine import ComfyClient, Settings
from . import registry
from .auth import make_auth_dependency
from .db import Db
from .jobs import JobContext, JobQueue
from .routes import router
from .store import SourceEvicted, SourceStore

logger = logging.getLogger("API")

def create_app(settings: Settings, load_caps: bool = True) -> FastAPI:
    if load_caps:
        registry.load_capabilities()

    db = Db(settings.gateway_db_path)
    store = SourceStore(db=db, root=settings.CACHE_DIR / "sources",
                        max_bytes=settings.GATEWAY_CACHE_MAX_GB * 1024 ** 3)

    def make_context(ctx: JobContext) -> JobContext:
        try:
            ctx.source = store.get(ctx.source_id)
        except SourceEvicted:
            ctx.source = None
        ctx.settings = settings
        return ctx

    queue = JobQueue(
        db=db, workdir_root=settings.CACHE_DIR / "jobs",
        handlers={c.id: c.handler for c in registry.REGISTRY.values()},
        job_timeout_s=settings.GATEWAY_JOB_TIMEOUT_S,
        result_ttl_hours=settings.GATEWAY_RESULT_TTL_HOURS,
        make_context=make_context)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"rr-ai-gateway starting; capabilities: {list(registry.REGISTRY)}")
        await queue.start()
        yield
        await queue.stop()

    auth = make_auth_dependency(settings.GATEWAY_TOKEN)
    app = FastAPI(title="rr-ai-gateway", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.settings = settings
    app.state.store = store
    app.state.queue = queue
    app.include_router(router, dependencies=[Depends(auth)])

    from .legacy import router as legacy_router
    app.include_router(legacy_router, dependencies=[Depends(auth)])

    @app.get("/health")
    async def health():
        comfy_up = await ComfyClient.check_health()
        return {"status": "ok" if comfy_up else "degraded",
                "comfy_url": settings.comfy_url, "connected": comfy_up,
                "capabilities": list(registry.REGISTRY),
                "queue_depth": queue.depth()}

    return app
