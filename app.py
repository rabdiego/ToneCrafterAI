import streamlit as st
import requests

st.set_page_config(page_title="ToneCrafter AI", page_icon="🎸", layout="centered")

API_URL = "http://localhost:8000/api"

st.title("🎸 ToneCrafter AI")
st.markdown("Seu Guitar Tech particular. Descreva o som que você quer, peça o timbre de uma banda ou envie o anexo de um riff!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.expander("📎 Tem um riff gravado? Anexe o áudio aqui"):
    uploaded_file = st.file_uploader("Escolha um arquivo .wav ou .mp3", type=["wav", "mp3"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("🎧 Enviar e Analisar Áudio", use_container_width=True):
            mensagem_usuario = f"📁 *Áudio enviado para análise: {uploaded_file.name}*"
            st.session_state.messages.append({"role": "user", "content": mensagem_usuario})
            
            with st.chat_message("user"):
                st.markdown(mensagem_usuario)
                
            with st.chat_message("assistant"):
                with st.spinner("Ouvindo o áudio e montando o patch... (Isso pode levar alguns segundos)"):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/chat/audio", files=files)
                    
                    if response.status_code == 200:
                        bot_response = response.json()["response"]
                        st.markdown(bot_response)
                        # Salva a resposta no histórico para não sumir ao recarregar a página
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    else:
                        st.error(f"Erro na API: {response.text}")

if prompt := st.chat_input("Ex: Som denso pro metal ou Timbre do David Gilmour"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando e regulando os pedais..."):
            response = requests.post(
                f"{API_URL}/chat/text", 
                json={"query": prompt}
            )
            
            if response.status_code == 200:
                bot_response = response.json()["response"]
                st.markdown(bot_response)
                # Salva a resposta no histórico
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
            else:
                st.error(f"Erro na API: {response.text}")
