import os
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from src.audio_extractor import AudioExtractorAgent
from src.mockup_crafter import MockupSetupCrafterAgent
from src.web_searcher import WebSearcherAgent
from src.setup_crafter import PedalSetupCrafterAgent
from src.manual_rag import PedalboardRAG
from src.schemas import RouteDecision, RouteChoice
from src.settings import settings

class GraphState(TypedDict):
    user_input: str
    is_audio: bool
    route: Optional[str]
    clean_query: Optional[str]
    blueprint: Optional[any]
    patch: Optional[any]
    final_response: Optional[str]


class ToneCrafterGraph:
    def __init__(self):
        load_dotenv()
        
        self.audio_worker = AudioExtractorAgent()
        self.mockup_worker = MockupSetupCrafterAgent()
        self.web_worker = WebSearcherAgent()
        self.setup_crafter = PedalSetupCrafterAgent()
        self.rag_system = PedalboardRAG()
        
        self.llm_flash = ChatGoogleGenerativeAI(model=settings.CONVERSATION_LLM_MODEL_NAME, temperature=0.3)
        
        self.app = self._build_graph()


    def _build_graph(self):
        workflow = StateGraph(GraphState)

        def router_node(state: GraphState):
            print("🧭 [Nó: Roteador] Avaliando a intenção do usuário...")
            if state["is_audio"]:
                return {"route": "audio", "clean_query": state["user_input"]}
            
            router_prompt = ChatPromptTemplate.from_messages([
                ("system", "Responda apenas com 'web' ou 'mockup'. Use 'web' se o usuário citar o nome de uma BANDA, MÚSICA ou GUITARRISTA. Caso contrário, use 'mockup'."),
                ("human", "{user_input}") 
            ])
            
            route_str = (router_prompt | self.llm_flash).invoke({"user_input": state["user_input"]}).content.strip().lower()
            
            if "web" in route_str:
                route_str = "web"
            else:
                route_str = "mockup"
                
            return {"route": route_str, "clean_query": state["user_input"]}


        def web_node(state: GraphState):
            print("🌐 [Nó: Web Searcher] Acionado.")
            blueprint = self.web_worker.search_and_craft(state["clean_query"])
            return {"blueprint": blueprint}


        def mockup_node(state: GraphState):
            print("📝 [Nó: Mockup Crafter] Acionado.")
            blueprint = self.mockup_worker.craft_mockup(state["clean_query"])
            return {"blueprint": blueprint}


        def audio_node(state: GraphState):
            print("🎧 [Nó: Audio Extractor] Acionado.")
            blueprint = self.audio_worker.analyze_audio(state["clean_query"])
            return {"blueprint": blueprint}


        def crafter_node(state: GraphState):
            print("🧠 [Nó: Setup Crafter] Acionado.")
            if not state.get("blueprint"):
                print("⚠️ Blueprint vazio. Gerando patch em branco.")
                return {"patch": None}
                
            patch = self.setup_crafter.craft_setup(state["blueprint"], self.rag_system)
            return {"patch": patch}


        def synthesizer_node(state: GraphState):
            print("✍️ [Nó: Sintetizador] Acionado.")
            
            if not state.get("patch"):
                return {"final_response": "Poxa, não consegui identificar efeitos suficientes. Pode dar mais detalhes?"}

            synthesizer_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "Você é um produtor musical e técnico de guitarra amigável. "
                    "Você acabou de montar um patch em uma pedaleira para o usuário. "
                    "Sua resposta DEVE seguir EXATAMENTE esta estrutura:\n\n"
                    "1. Uma breve descrição da 'Vibe Geral' do timbre.\n"
                    "2. Uma lista de cada efeito ATIVADO na pedaleira.\n"
                    "3. Para cada efeito, forneça as 'Configurações (Settings)' e uma 'Justificativa' de POR QUE aquele pedal específico foi escolhido para compor o timbre, cruzando a intenção original com a limitação da pedaleira.\n\n"
                    "Não liste os efeitos que estão desligados (is_active=False)."
                    "Use uma linguagem simples e direta, com um tom amigável, mas não infantil."
                )),
                ("human", (
                    "--- INTENÇÃO ORIGINAL DO TIMBRE (BLUEPRINT) ---\n"
                    "{blueprint}\n\n"
                    "--- PATCH REAL GERADO NA PEDALEIRA ---\n"
                    "{patch}"
                ))
            ])
            
            response = (synthesizer_prompt | self.llm_flash).invoke({
                "blueprint": state["blueprint"].model_dump_json(),
                "patch": state["patch"].model_dump_json()
            })
            return {"final_response": response.content}


        workflow.add_node("router", router_node)
        workflow.add_node("web_worker", web_node)
        workflow.add_node("mockup_worker", mockup_node)
        workflow.add_node("audio_worker", audio_node)
        workflow.add_node("setup_crafter", crafter_node)
        workflow.add_node("synthesizer", synthesizer_node)

        workflow.set_entry_point("router")

        def route_condition(state: GraphState):
            return state["route"]

        workflow.add_conditional_edges(
            "router",
            route_condition,
            {
                "web": "web_worker",
                "mockup": "mockup_worker",
                "audio": "audio_worker"
            }
        )

        workflow.add_edge("web_worker", "setup_crafter")
        workflow.add_edge("mockup_worker", "setup_crafter")
        workflow.add_edge("audio_worker", "setup_crafter")
        
        workflow.add_edge("setup_crafter", "synthesizer")
        
        workflow.add_edge("synthesizer", END)

        return workflow.compile()


    def process(self, user_input: str, is_audio: bool = False) -> str:
        """Invoca o grafo com o estado inicial."""
        initial_state = {
            "user_input": user_input,
            "is_audio": is_audio,
            "route": None,
            "clean_query": None,
            "blueprint": None,
            "patch": None,
            "final_response": None
        }
        
        final_state = self.app.invoke(initial_state)
        return final_state["final_response"]


if __name__ == "__main__":
    app = ToneCrafterGraph()
    resposta = app.process("samples/test-gemini.mp3", is_audio=True)
    print("\n=== RESPOSTA FINAL ===\n")
    print(resposta)

