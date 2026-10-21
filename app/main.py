"""FastAPI entry point; use one worker to keep one model copy in memory."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.retrieval.search import SearchService


class IndexRequest(BaseModel):
    docs_path: str = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=200)


def create_app(service: SearchService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.service = service or SearchService()
        yield

    application = FastAPI(title="Local sparse semantic search", lifespan=lifespan)

    @application.get("/health")
    def health():
        return application.state.service.health()

    @application.post("/index")
    def index(request: IndexRequest):
        try:
            return application.state.service.index_directory(request.docs_path)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @application.post("/search")
    def search(request: SearchRequest):
        try:
            return application.state.service.search(request.query, request.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


app = create_app()
