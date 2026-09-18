from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.compare import load_comparison
from app.config import get_settings
from app.report import load_report
from app.store import corpus_snapshot

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Community Voices", docs_url=None, redoc_url=None)


@app.get("/")
def homepage() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    settings = get_settings()
    return {
        "status": "ok",
        "openai_api_key_configured": settings.api_key_configured,
    }


@app.get("/api/corpus")
def corpus() -> dict[str, object]:
    settings = get_settings()
    return corpus_snapshot(settings.chroma_path)


@app.get("/api/report")
def report() -> dict[str, object]:
    settings = get_settings()
    payload = load_report(settings.report_path)
    if payload is None:
        return {"status": "empty"}
    return {"status": "ready", "report": payload}


@app.get("/api/comparison")
def comparison() -> dict[str, object]:
    settings = get_settings()
    payload = load_comparison(settings.comparison_path)
    if payload is None:
        return {"status": "empty"}
    return {"status": "ready", "comparison": payload}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
