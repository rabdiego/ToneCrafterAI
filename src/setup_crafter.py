from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.manual_rag import PedalboardRAG
from src.settings import settings
from src.schemas import ToneBlueprint

class PedalSetupCrafterAgent:
    def __init__(self):
        load_dotenv()
        
        self.llm = ChatGoogleGenerativeAI(model=settings.SETUP_CRAFTER_MODEL_NAME, temperature=0.1)
        self.structured_llm = self.llm.with_structured_output(ToneBlueprint)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Você é um técnico de guitarra profissional (Guitar Tech). "
                "Sua missão é pegar um projeto de timbre genérico e traduzi-lo EXATAMENTE para as opções disponíveis na pedaleira do usuário.\n\n"
                "REGRAS RÍGIDAS:\n"
                "1. Leia atentamente o 'Contexto do Manual' fornecido. Ele contém as tabelas dos efeitos reais que a pedaleira possui.\n"
                "2. Escolha o efeito da pedaleira que mais se aproxima da intenção genérica exigida.\n"
                "3. Na chave 'description_or_name', use APENAS nomes exatos que apareceram no 'Contexto do Manual'. Não invente modelos.\n"
                "4. Na chave 'settings', defina os valores respeitando os limites dos parâmetros do manual (ex: se o Range é 0~99, não coloque 100).\n"
                "5. Mantenha os mesmos blocos ligados (is_active=True) que foram solicitados no projeto original."
            )),
            ("human", (
                "Vibe Geral Desejada: {overall_vibe}\n\n"
                "Efeitos Genéricos Solicitados:\n{requested_effects}\n\n"
                "--- CONTEXTO DO MANUAL DA PEDALEIRA (RAG) ---\n{rag_context}"
            ))
        ])
        
        self.chain = self.prompt | self.structured_llm

    def craft_setup(
        self, 
        blueprint: ToneBlueprint,
        rag_system: PedalboardRAG
    ):
        rag_context_accumulated = ""
        requested_effects_str = ""
        
        blueprint_dict = blueprint.model_dump()
        
        for categoria, slot in blueprint_dict.items():
            if categoria == "overall_vibe":
                continue
            
            if slot["is_active"]:
                requested_effects_str += f"- {categoria.upper()}: {slot['description_or_name']} | Config: {slot['settings']}\n"
                
                query_para_rag = f"Categoria: {categoria}. Descrição: {slot['description_or_name']}."
                print(f"   🔍 Consultando o manual para opções de: {categoria.upper()}...")
                
                resultados = rag_system.search_effect_parameters(query=query_para_rag, k=3)
                rag_context_accumulated += f"\n>> OPÇÕES NO MANUAL PARA {categoria.upper()}:\n{resultados}\n"
        
        try:
            patch_final = self.chain.invoke({
                "overall_vibe": blueprint.overall_vibe,
                "requested_effects": requested_effects_str,
                "rag_context": rag_context_accumulated
            })
            return patch_final
            
        except Exception as e:
            raise e

