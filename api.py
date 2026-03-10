import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
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
async def chat_text(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "")
        thread_id = data.get("thread_id", "sessao_padrao")
        
        graph: ToneCrafterGraph = app_state["graph"]
        
        return StreamingResponse(
            graph.process_stream(text_input=query, thread_id=thread_id), 
            media_type="application/x-ndjson"
        )
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
            
        graph = app_state["graph"]
        
        def stream_and_cleanup():
            try:
                for chunk in graph.process_stream(text_input=query, audio_path=temp_file_path, thread_id=thread_id):
                    yield chunk
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"🧹 Ficheiro temporário apagado: {temp_file_path}")

        return StreamingResponse(
            stream_and_cleanup(), 
            media_type="application/x-ndjson"
        )
        
    except Exception as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

