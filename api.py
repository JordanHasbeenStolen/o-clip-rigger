from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core import search, total_vectors, get_path

app = FastAPI(title="o-clip-rigger API")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)


class SearchResult(BaseModel):
    path: str
    url: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


@app.get("/health")
def health():
    return {"status": "ok", "vectors": total_vectors()}


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest):
    raw = search(req.query, req.top_k)
    results = [
        SearchResult(path=r["path"], url=f"/image/{r['index']}", score=r["score"])
        for r in raw
    ]
    return SearchResponse(query=req.query, results=results)


@app.get("/image/{idx}")
def get_image(idx: int):
    path = get_path(idx)
    if path is None:
        raise HTTPException(404, "Image not found")
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "File missing on disk")
    return FileResponse(p)