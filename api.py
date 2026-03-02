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
    print("✅ [FastAPI] Sistema ToneCrafter pronto para receber requisições!")
    yield
    app_state.clear()


app = FastAPI(title="ToneCrafter AI API", lifespan=lifespan)

class TextQuery(BaseModel):
    query: str


@app.post("/api/chat/text")
async def chat_text(request: TextQuery):
    try:
        graph: ToneCrafterGraph = app_state["graph"]
        resposta = graph.process(user_input=request.query, is_audio=False)
        return {"response": resposta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/audio")
async def chat_audio(file: UploadFile = File(...)):
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        graph: ToneCrafterGraph = app_state["graph"]
        resposta = graph.process(user_input=temp_file_path, is_audio=True)
        
        os.remove(temp_file_path)
        
        return {"response": resposta}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

