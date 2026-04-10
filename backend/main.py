import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

FRONTEND = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "voice-agent")
INDEX = os.path.join(FRONTEND, "index.html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
@app.get("/voice-agent")
@app.get("/voice-agent/")
async def index():
    return FileResponse(INDEX)

app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="static")
