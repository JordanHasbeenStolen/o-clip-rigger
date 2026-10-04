from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core import SEARCH_MODELS, TAG_MODELS, search, tag_image, total_vectors, get_path, warmup


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup()
    yield


app = FastAPI(title="o-clip-rigger API", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    model: str = Field("openclip")


class SearchResult(BaseModel):
    path: str
    url: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class Tag(BaseModel):
    tag: str
    score: float


class TagResponse(BaseModel):
    model: str
    tags: List[Tag]


@app.get("/health")
def health():
    return {"status": "ok", "vectors": {m: total_vectors(m) for m in SEARCH_MODELS}}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    if req.model not in SEARCH_MODELS:
        raise HTTPException(400, f"Unknown model: {req.model!r}, expected one of {list(SEARCH_MODELS)}")
    raw = search(req.query, req.top_k, model=req.model)
    results = [
        SearchResult(path=r["path"], url=f"/image/{r['index']}?model={req.model}", score=r["score"])
        for r in raw
    ]
    return SearchResponse(query=req.query, results=results)


@app.post("/tag", response_model=TagResponse)
async def tag_endpoint(
    file: UploadFile = File(...),
    model: str = Form("openclip"),
):
    if model not in TAG_MODELS:
        raise HTTPException(400, f"Unknown model: {model!r}, expected one of {list(TAG_MODELS)}")
    image_bytes = await file.read()
    tags = tag_image(image_bytes, model=model)
    return TagResponse(model=model, tags=tags)


@app.get("/image/{idx}")
def get_image(idx: int, model: str = "openclip"):
    if model not in SEARCH_MODELS:
        raise HTTPException(400, f"Unknown model: {model!r}, expected one of {list(SEARCH_MODELS)}")
    path = get_path(idx, model=model)
    if path is None:
        raise HTTPException(404, "Image not found")
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File missing on disk")
    return FileResponse(p)