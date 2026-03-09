import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.conversation import ToneCrafterGraph

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [FastAPI] Iniciando os motores da IA e carregando o RAG...")
    app_state["graph"] = ToneCrafterGraph()
    print("✅ [FastAPI] Sistema ToneCrafter pronto!")
    yield
    app_state.clear()

app = FastAPI(title="ToneCrafter AI API", lifespan=lifespan)

class TextQuery(BaseModel):
    query: str
    thread_id: str = "sessao_padrao"

@app.post("/api/chat/text")
async def chat_text(request: TextQuery):
    try:
        graph: ToneCrafterGraph = app_state["graph"]
        resposta = graph.process(text_input=request.query, thread_id=request.thread_id)
        return {"response": resposta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/audio")
async def chat_audio(
    file: UploadFile = File(...), 
    query: str = Form(""),
    thread_id: str = Form("sessao_padrao")
):
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        graph: ToneCrafterGraph = app_state["graph"]
        
        resposta = graph.process(text_input=query, audio_path=temp_file_path, thread_id=thread_id)
        
        os.remove(temp_file_path)
        return {"response": resposta}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

