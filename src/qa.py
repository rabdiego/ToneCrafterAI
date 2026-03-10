from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.settings import settings
from src.manual_rag import PedalboardRAG
from src.audio_extractor import AudioExtractorAgent

class QAAgentWorker:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.CONVERSATION_LLM_MODEL_NAME,
            temperature=0.1
        )


    def process_qa(
        self,
        state: dict,
        rag_system: PedalboardRAG,
        audio_worker: AudioExtractorAgent
    ) -> dict:
        @tool
        def buscar_manual(query: str) -> str:
            """
            Busca efeitos no manual da pedaleira (RAG).
            REGRA CRÍTICA DE USO: A 'query' DEVE ser extremamente curta, contendo APENAS palavras-chave técnicas em inglês. NUNCA use frases completas.
            CERTO: 'Pitch Shifter', 'Noise Gate threshold', 'Tube Screamer distortion'.
            ERRADO: 'Como eu faço para mudar a afinação', 'Qual efeito simula o Tube Screamer'.
            """
            return rag_system.search_effect_parameters(
                query=query,
                k=3
            )
            
        tavily_tool = TavilySearch(
            max_results=3, 
            name="web_search", 
            description="""
            Descubra equipamentos de bandas na web.
            REGRA CRÍTICA DE USO: Use apenas palavras-chave diretas e nomes próprios. Nenhuma palavra de ligação.
            CERTO: 'Red Hot Chili Peppers Can't Stop guitar pedals'.
            ERRADO: 'Quais pedais o Red Hot Chili Peppers usa na música Can't Stop'.
            """
        )
        
        ferramentas_qa = [buscar_manual, tavily_tool]
        
        if state.get("audio_path"):
            @tool
            def ouvir_audio_anexado() -> str:
                """Use esta ferramenta para escutar e analisar o arquivo de áudio que o usuário anexou."""
                blueprint_extraido = audio_worker.analyze_audio(state["audio_path"])
                return f"A análise do áudio revelou os seguintes prováveis equipamentos:\n{blueprint_extraido.model_dump_json()}"
            
            ferramentas_qa.append(ouvir_audio_anexado)
        
        system_prompt = "Pesquise usando suas ferramentas e retorne os fatos brutos encontrados. Se o usuário perguntar sobre o áudio anexado, use a ferramenta de ouvir o áudio, ela retornará características que descrevem o conteúdo do áudio em texto."
        
        agente_qa = create_agent(self.llm, ferramentas_qa, system_prompt=system_prompt)
        resposta = agente_qa.invoke({"messages": [HumanMessage(content=state["clean_query"])]})
        
        return {"agent_context": resposta["messages"][-1].content}

