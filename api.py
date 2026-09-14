from fastapi import FastAPI

app = FastAPI(title="o-clip-rigger API")


@app.get("/health")
def health():
    return {"status": "ok"}