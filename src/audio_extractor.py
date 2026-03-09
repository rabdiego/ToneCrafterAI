import base64
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.settings import settings
from src.schemas import ToneBlueprint

class AudioExtractorAgent:
    def __init__(self):
        load_dotenv()
    
        self.llm = ChatGoogleGenerativeAI(
            model=settings.AUDIO_EXTRACTOR_LLM_MODEL_NAME, 
            temperature=0.1
        )
        
        self.structured_llm = self.llm.with_structured_output(ToneBlueprint)


    def _encode_audio_to_base64(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {file_path}")
            
        with open(file_path, "rb") as audio_file:
            audio_data = audio_file.read()
            return base64.b64encode(audio_data).decode("utf-8")


    def analyze_audio(self, audio_path: str, user_instructions: str = None) -> ToneBlueprint:
        print(f"🎧 [Audio Extractor] Analisando o arquivo: {audio_path}...")
        
        base64_audio = self._encode_audio_to_base64(audio_path)
        
        ext = audio_path.split('.')[-1].lower()
        mime_type = "audio/wav" if ext == "wav" else f"audio/{ext}"

        prompt_text = (
            "Você é um engenheiro de áudio mestre em timbres de guitarra elétrica. "
            "Ouça atentamente o clipe de áudio fornecido. "
            "Identifique QUAIS efeitos estão sendo usados para compor este timbre. "
            "Mapeie os efeitos encontrados APENAS para os campos fornecidos. "
            "Seja técnico e cirúrgico nas suas descrições e deduções de configurações (settings). "
            "Nas categorias de Preamp, temos efeitos como Boost, Autowah, etc. "
            "Nas categorias de Distortion, temos efeitos como Overdrive, Distorção, Fuzz, etc. "
            "As categorias amplifier e cabinet SEMPRE estarão ativas. "
            "Nas categorias de Modulation, teremos Chorus, Phaser, Pitch shifter, etc."
        )

        if user_instructions:
            prompt_text += f"\n\nATENÇÃO ÀS INSTRUÇÕES ESPECÍFICAS DO USUÁRIO:\n'{user_instructions}'"

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "media", "mime_type": mime_type, "data": base64_audio}
            ]
        )
        
        try:
            analysis_result = self.structured_llm.invoke([message])
            return analysis_result
        except Exception as e:
            print(f"⚠️ Erro ao extrair áudio: {e}")
            raise e

