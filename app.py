import streamlit as st
import requests
import uuid
import json
import time

def stream_generator(texto):
    """Simula o efeito Typewriter na resposta final."""
    for palavra in texto.split(" "):
        yield palavra + " "
        time.sleep(0.03)



st.set_page_config(page_title="ToneCrafter AI", page_icon="🎸", layout="centered")

API_URL = "http://localhost:8000/api"

st.title("🎸 ToneCrafter AI")
st.markdown("Seu Guitar Tech particular.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ex: Som do Gilmour, ou anexe um áudio...", accept_file=True, file_type=["wav", "mp3"]):
    if isinstance(prompt, dict):
        texto_usuario = prompt.get("text", "")
        arquivos = prompt.get("files", [])
    else:
        texto_usuario = getattr(prompt, "text", prompt) if not isinstance(prompt, str) else prompt
        arquivos = getattr(prompt, "files", []) if not isinstance(prompt, str) else []
    
    has_audio = len(arquivos) > 0
    audio_file = arquivos[0] if has_audio else None
    
    if has_audio:
        mensagem_visual = f"📎 *Arquivo anexado: {audio_file.name}*\n\n{texto_usuario}"
    else:
        mensagem_visual = texto_usuario
        
    st.session_state.messages.append({"role": "user", "content": mensagem_visual})
    
    with st.chat_message("user"):
        st.markdown(mensagem_visual)

    with st.chat_message("assistant"):
        status_box = st.status("Iniciando ToneCrafter...", expanded=True)
        
        final_bot_response = ""
        
        if has_audio:
            files_payload = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
            data_payload = {"thread_id": st.session_state.session_id, "query": texto_usuario}
            response = requests.post(f"{API_URL}/chat/audio", files=files_payload, data=data_payload, stream=True)
        else:
            response = requests.post(
                f"{API_URL}/chat/text", 
                json={"query": texto_usuario, "thread_id": st.session_state.session_id},
                stream=True
            )
            
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    dados = json.loads(line.decode("utf-8"))
                    
                    if dados["type"] == "status":
                        status_box.update(label=dados["content"])
                        
                    elif dados["type"] == "final":
                        status_box.update(label="✅ Finalizado", state="complete", expanded=False)
                        final_bot_response = dados["content"]
            
            st.write_stream(stream_generator(final_bot_response))
            st.session_state.messages.append({"role": "assistant", "content": final_bot_response})
            
        else:
            status_box.update(label="❌ Erro na conexão", state="error")
            st.error(f"Erro na API: {response.text}")
