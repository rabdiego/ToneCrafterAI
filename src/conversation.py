import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from src.audio_extractor import AudioExtractorAgent
from src.mockup_crafter import MockupSetupCrafterAgent
from src.web_searcher import WebSearcherAgent
from src.setup_crafter import PedalSetupCrafterAgent
from src.router import SemanticRouterAgent
from src.qa import QAAgentWorker
from src.responder import ResponderAgent
from src.guardrails import GuardrailsAgent
from src.manual_rag import PedalboardRAG
from src.schemas import GraphState

class ToneCrafterGraph:
    def __init__(self):
        load_dotenv()
        
        self.audio_worker = AudioExtractorAgent()
        self.mockup_worker = MockupSetupCrafterAgent()
        self.web_worker = WebSearcherAgent()
        self.setup_crafter = PedalSetupCrafterAgent()
        self.rag_system = PedalboardRAG()
        self.router_worker = SemanticRouterAgent()
        self.qa_worker = QAAgentWorker()
        self.responder_agent = ResponderAgent()
        self.guardrail_worker = GuardrailsAgent()
        self.memory = MemorySaver()
        self.app = self._build_graph()


    def _build_graph(self):
        workflow = StateGraph(GraphState)

        def router_node(state: GraphState):
            print("🧭 [Nó: Roteador] Lendo histórico e reescrevendo a consulta...")
            return self.router_worker.route_request(state)


        def qa_node(state: GraphState):
            print(f"📚 [Nó: QA] Coletando dados brutos para: '{state['clean_query']}'...")
            return self.qa_worker.process_qa(state, self.rag_system, self.audio_worker)


        def web_node(state: GraphState):
            print("🌐 [Nó: Web Searcher] Acionado.")
            query = state.get("optimized_search_query") or state["clean_query"]
            return {"blueprint": self.web_worker.search_and_craft(query)}


        def mockup_node(state: GraphState):
            print("📝 [Nó: Mockup Crafter] Acionado.")
            return {"blueprint": self.mockup_worker.craft_mockup(state["clean_query"])}


        def audio_node(state: GraphState):
            print("🎧 [Nó: Audio Extractor] Acionado.")
            audio_path = state["audio_path"]
            if not audio_path:
                return {"blueprint": None}
            user_instructions = state.get("audio_instructions") or "Extraia o timbre de guitarra deste áudio."
            return {"blueprint": self.audio_worker.analyze_audio(audio_path, user_instructions)}


        def crafter_node(state: GraphState):
            print("🧠 [Nó: Setup Crafter] Acionado.")
            if not state.get("blueprint"):
                return {"patch": None}
            return {"patch": self.setup_crafter.craft_setup(state["blueprint"], self.rag_system)}


        def unified_responder_node(state: GraphState):
            print("🗣️ [Nó: Respondedor Unificado] Formatando a resposta final...")
            return self.responder_agent.generate_response(state)
        

        def guardrail_node(state: GraphState):
            print("🛡️ [Nó: Guardrail Multimodal] Avaliando a segurança da requisição...")
            return self.guardrail_worker.evaluate_request(state)


        workflow.add_node("guardrail", guardrail_node)
        workflow.add_node("router", router_node)
        workflow.add_node("web_worker", web_node)
        workflow.add_node("mockup_worker", mockup_node)
        workflow.add_node("audio_worker", audio_node)
        workflow.add_node("setup_crafter", crafter_node)
        workflow.add_node("qa_node", qa_node)
        workflow.add_node("unified_responder", unified_responder_node)

        workflow.set_entry_point("guardrail")

        workflow.add_conditional_edges(
            "guardrail",
            lambda state: state["route"],
            {
                "safe": "router",
                "blocked": END
            }
        )

        workflow.add_conditional_edges(
            "router",
            lambda state: state["route"],
            {
                "web": "web_worker",
                "mockup": "mockup_worker",
                "audio": "audio_worker",
                "qa": "qa_node",
                "chat": "unified_responder"
            }
        )

        workflow.add_edge("web_worker", "setup_crafter")
        workflow.add_edge("mockup_worker", "setup_crafter")
        workflow.add_edge("audio_worker", "setup_crafter")
        
        workflow.add_edge("setup_crafter", "unified_responder")
        workflow.add_edge("qa_node", "unified_responder")
        
        workflow.add_edge("unified_responder", END)

        return workflow.compile(checkpointer=self.memory)


    def process(self, text_input: str = "", audio_path: str = None, thread_id: str = "sessao_padrao") -> str:
        is_audio = bool(audio_path)
        
        if is_audio and text_input:
            msg_content = f"[Arquivo de Áudio Anexado] {audio_path}"
        elif is_audio:
            msg_content = "[Arquivo de Áudio Anexado] Gere um patch baseado neste áudio."
            text_input = "Gere um patch baseado neste áudio."
        else:
            msg_content = text_input
            
        initial_state = {
            "messages": [HumanMessage(content=msg_content)],
            "user_input": text_input,
            "audio_path": audio_path,
            "is_audio": is_audio
        }

        config = {'configurable': {'thread_id': thread_id}}
        final_state = self.app.invoke(initial_state, config=config)
        return final_state["final_response"]


if __name__ == "__main__":
    app = ToneCrafterGraph()
    print(app.process("Timbre de Master of Puppets do Metallica", thread_id="teste_local"))
    print("-" * 50)
    print(app.process("Mas eu queria apenas a distorção", thread_id="teste_local"))
