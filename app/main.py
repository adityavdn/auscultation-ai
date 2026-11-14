"""FastAPI service: analysis API, sample recordings, doctor inbox, agent chat, and the PWA."""
import json
import logging
import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import agent
from app.db import CaseStore
from app.fusion import fuse
from app.inference import Analyzer
from ml.audio import AudioError

log = logging.getLogger("app")
DISCLAIMER = "Research prototype. Decision support only. Not a medical device."
MAX_BYTES = 20 * 1024 * 1024
ROOT = Path(__file__).resolve().parent.parent


class CaseIn(BaseModel):
    patient_name: str = Field("Anonymous", max_length=100)
    urgency: Literal["routine", "soon", "urgent"] = "routine"
    summary: dict
    analysis: Optional[dict] = None
    transcript: list = Field(default_factory=list, max_length=200)


class ChatIn(BaseModel):
    role: Literal["patient", "doctor"] = "patient"
    history: list[dict] = Field(default_factory=list, max_length=100)
    analysis: Optional[dict] = None


class FromChatIn(BaseModel):
    patient_name: str = Field("Anonymous", max_length=100)
    history: list[dict] = Field(default_factory=list, max_length=100)
    analysis: Optional[dict] = None


async def _read(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 20 MB).")
    return data


def create_app(models_dir, samples_dir, db_path):
    app = FastAPI(title="Auscultation AI (research prototype)", description=DISCLAIMER)
    analyzer = Analyzer(models_dir)
    cases = CaseStore(db_path)
    samples_dir = Path(samples_dir)
    manifest_file = samples_dir / "manifest.json"
    samples = {s["id"]: s for s in json.loads(manifest_file.read_text())} if manifest_file.exists() else {}
    app.state.analyzer, app.state.cases = analyzer, cases

    @app.middleware("http")
    async def vercel_oidc(request: Request, call_next):
        # On Vercel the AI Gateway credential arrives per request; it is project-scoped, so caching it is safe.
        token = request.headers.get("x-vercel-oidc-token")
        if token:
            os.environ["VERCEL_OIDC_TOKEN"] = token
        return await call_next(request)

    def run(data, kind):
        try:
            return analyzer.analyze(data, kind)
        except AudioError as e:
            raise HTTPException(422, str(e))

    @app.post("/api/analyze/combined")
    async def analyze_combined(heart: Optional[UploadFile] = File(None), lung: Optional[UploadFile] = File(None)):
        if heart is None and lung is None:
            raise HTTPException(422, "Upload a heart recording, a lung recording, or both.")
        h = run(await _read(heart), "heart") if heart else None
        lu = run(await _read(lung), "lung") if lung else None
        return {"heart": h, "lung": lu, "fusion": fuse(h and h["heads"], lu and lu["heads"]),
                "disclaimer": DISCLAIMER}

    @app.post("/api/analyze/{kind}")
    async def analyze(kind: str, file: UploadFile = File(...)):
        if kind not in ("heart", "lung"):
            raise HTTPException(404, "Unknown kind; use heart or lung.")
        r = run(await _read(file), kind)
        heads = r["heads"]
        return r | {"fusion": fuse(heads if kind == "heart" else None, heads if kind == "lung" else None),
                    "disclaimer": DISCLAIMER}

    @app.get("/api/samples")
    def list_samples():
        return [{k: v for k, v in s.items() if k != "file"} for s in samples.values()]

    @app.get("/api/samples/{sample_id}/audio")
    def sample_audio(sample_id: str):
        if sample_id not in samples:
            raise HTTPException(404, "No such sample.")
        return FileResponse(samples_dir / samples[sample_id]["file"], media_type="audio/wav")

    @app.post("/api/cases")
    def add_case(case: CaseIn):
        return cases.add(case.patient_name, case.urgency, case.summary, case.analysis, case.transcript)

    @app.get("/api/cases")
    def list_cases():
        return cases.list()

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: int):
        c = cases.get(case_id)
        if not c:
            raise HTTPException(404, "No such case.")
        return c

    @app.post("/api/chat")
    def chat(body: ChatIn):
        return agent.chat(body.history, body.analysis, body.role)

    @app.post("/api/cases/from-chat")
    def case_from_chat(body: FromChatIn):
        s = agent.summarize_case(body.history, body.analysis)
        return cases.add(body.patient_name, s["urgency"], s, body.analysis, body.history)

    @app.get("/api/health")
    def health():
        return {"ok": True, "heads": sorted(analyzer.heads), "disclaimer": DISCLAIMER}

    static = ROOT / "app" / "static"
    if static.exists():
        app.mount("/", StaticFiles(directory=static, html=True), name="static")
    return app


def app_from_env():
    return create_app(os.environ.get("MODELS_DIR", ROOT / "models"), os.environ.get("SAMPLES_DIR", ROOT / "samples"),
                      os.environ.get("DB_PATH", "/tmp/cases.sqlite"))
