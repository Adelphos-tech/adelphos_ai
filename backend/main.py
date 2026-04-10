import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "voice-agent")

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/{full_path:path}")
async def serve(full_path: str):
    return FileResponse(os.path.join(FRONTEND, "index.html"))
