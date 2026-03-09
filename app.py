import streamlit as st
import requests
import uuid

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

if "staged_audio" not in st.session_state:
    st.session_state.staged_audio = None

if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

st.markdown("---")
gravacao = st.audio_input(
    "🎙️ Gravar Riff da Guitarra (O áudio ficará salvo esperando sua mensagem)", 
    key=f"mic_{st.session_state.audio_key}"
)

if gravacao:
    st.session_state.staged_audio = gravacao
    st.success("✅ Áudio capturado! Digite sua instrução na barra abaixo e aperte Enter para enviar os dois juntos.")

if prompt := st.chat_input("Ex: Qual pedal simula a distorção da gravação acima?"):
    audio_file = st.session_state.staged_audio
    has_audio = bool(audio_file)
    
    if has_audio:
        mensagem_visual = f"🎙️ *Áudio gravado anexado*\n\n{prompt}"
    else:
        mensagem_visual = prompt
        
    st.session_state.messages.append({"role": "user", "content": mensagem_visual})
    
    with st.chat_message("user"):
        st.markdown(mensagem_visual)

    with st.chat_message("assistant"):
        with st.spinner("Ouvindo o riff e analisando as frequências..." if has_audio else "Regulando os pedais..."):
            
            if has_audio:
                files_payload = {"file": ("gravacao.wav", audio_file.getvalue(), "audio/wav")}
                data_payload = {"thread_id": st.session_state.session_id, "query": prompt}
                response = requests.post(f"{API_URL}/chat/audio", files=files_payload, data=data_payload)
                
                st.session_state.staged_audio = None
                st.session_state.audio_key += 1
                st.rerun()
                
            else:
                response = requests.post(
                    f"{API_URL}/chat/text", 
                    json={"query": prompt, "thread_id": st.session_state.session_id}
                )
                
            if response.status_code == 200:
                bot_response = response.json()["response"]
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            else:
                st.error(f"Erro na API: {response.text}")
