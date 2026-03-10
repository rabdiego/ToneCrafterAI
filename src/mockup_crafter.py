from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.settings import settings
from src.schemas import ToneBlueprint

class MockupSetupCrafterAgent:
    def __init__(self):
        load_dotenv()

        self.llm = ChatGoogleGenerativeAI(
            model=settings.MOCKUP_CRAFTER_LLM_MODEL_NAME, 
            temperature=0.1
        )
        
        self.structured_llm = self.llm.with_structured_output(ToneBlueprint)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Você é um engenheiro de áudio mestre em timbres de guitarra. "
                "O usuário vai descrever um som de guitarra que ele tem na cabeça. "
                "Sua tarefa é traduzir essa descrição livre para uma receita de efeitos genéricos. "
                "Se o usuário pedir 'som limpo com eco', você deve ativar amplifier (clean) e delay (eco). "
                "Preencha as descrições e configurações (settings) de forma técnica para orientar a montagem posterior. "
                "As categorias amplifier e cabinet SEMPRE estarão presentes (is_active=True). "
                "Deixe is_active=False para efeitos que não foram solicitados."
                "Nas categorias de Preamp, temos efeitos como Boost, Autowah, etc."
                "Nas categorias de Distortion, temos efeitos como Overdrive, Distorção, Fuzz, etc."
                "As categorias Amplifier e Cabinet SEMPRE estarão presentes."
                "Nas categorias de Modulation, teremos Chorus, Phaser, Pitch shifter, etc."
            )),
            ("human", "{user_description}")
        ])
        
        self.chain = self.prompt | self.structured_llm

    def craft_mockup(
        self,
        user_description: str
    ) -> ToneBlueprint:
        try:
            result = self.chain.invoke({"user_description": user_description})
            return result
        except Exception as e:
            raise e

