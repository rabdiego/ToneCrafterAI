import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

from src.audio_extractor import AudioExtractorAgent
from src.mockup_crafter import MockupSetupCrafterAgent
from src.web_searcher import WebSearcherAgent
from src.setup_crafter import PedalSetupCrafterAgent
from src.manual_rag import PedalboardRAG
from src.schemas import RouterDecision, GraphState
from src.settings import settings

class ToneCrafterGraph:
    def __init__(self):
        load_dotenv()
        
        self.audio_worker = AudioExtractorAgent()
        self.mockup_worker = MockupSetupCrafterAgent()
        self.web_worker = WebSearcherAgent()
        self.setup_crafter = PedalSetupCrafterAgent()
        self.rag_system = PedalboardRAG()
        
        self.llm_flash = ChatGoogleGenerativeAI(model=settings.CONVERSATION_LLM_MODEL_NAME, temperature=0.3)
        self.memory = MemorySaver()
        self.app = self._build_graph()


    def _build_graph(self):
        workflow = StateGraph(GraphState)
        router_llm = self.llm_flash.with_structured_output(RouterDecision)

        def router_node(state: GraphState):
            print("🧭 [Nó: Roteador] Lendo histórico e reescrevendo a consulta...")
            
            has_audio = bool(state.get("audio_path"))
            patch_atual = state.get("patch")
            user_text = state.get("user_input") or ""
            
            historico = ""
            if "messages" in state and len(state["messages"]) > 1:
                historico = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][-5:-1]])
            
            status_memoria = f"Patch Ativo: {patch_atual.overall_vibe}" if patch_atual else "NENHUM PATCH ATIVO."

            router_prompt = ChatPromptTemplate.from_messages([
                ("system", """Você é o Roteador do ToneCrafter AI. 
                
                1. REESCRITA (contextualized_query): Leia o Histórico e a Mensagem Atual. 
                Essa reescrita será passada para agentes sem contexto da conversa, então tente sintetizar o intuito do usuário
                de forma que um agente terceiro consiga entender perfeitamente, sem precisar ler a conversa.
                Por exemplo, se o usuário pedir para MUDAR O PATCH ATUAL, reescreva a query explicitando isso. 
                Exemplo de uma cadeira de inputs: Input: "Quero um timbre bem sujo e distorcido" --> Preset gerado --> "Tira a compressão, por favor?"
                A reescrita correta deve ser "Timbre bem sujo e distorcido, sem compressão."
                
                2. ROTEAMENTO (Intent):
                - CREATE: O usuário quer um timbre NOVO ou ALTERAR o patch ativo.
                - QA: Dúvidas conceituais ou sobre equipamentos ("Quais pedais o Metallica usa?").
                - CHAT: Conversa fiada ou saudações.
                
                3. SUB-ROTAS (Obrigatório se a intenção for CREATE):
                - web: O usuário citou o nome de uma BANDA, MÚSICA ou GUITARRISTA específico (ex: "Timbre do Gilmour", "Som do Nirvana").
                - mockup: O usuário descreveu características sonoras abstratas ("som limpo") OU pediu ajustes genéricos no patch atual ("aumente o ganho", "tire o delay").
                - audio: O usuário enviou um arquivo de áudio.
                - none: Para intenções QA ou CHAT.
                """),
                ("human", "Histórico:\n{historico}\n\nStatus: {status}\n\nMensagem Atual: '{user_text}'")
            ])
            
            decision = (router_prompt | router_llm).invoke({
                "historico": historico, "status": status_memoria, "has_audio": has_audio, "user_text": user_text
            })
            
            route_str = "audio" if (decision.intent.value == "create" and has_audio and decision.sub_route.value == "audio") else decision.sub_route.value if decision.intent.value == "create" else decision.intent.value
            query_limpa = decision.contextualized_query
            
            print(f"🔄 Consulta Traduzida: '{query_limpa}'")
            print(f"🔍 Busca Otimizada: '{decision.optimized_search_query}'")
            print(f"🎛️ Instrução de Áudio: '{decision.audio_instructions}'")
            
            return {
                "route": route_str, 
                "clean_query": query_limpa,
                "optimized_search_query": decision.optimized_search_query,
                "audio_instructions": decision.audio_instructions
            }


        def qa_node(state: GraphState):
            print(f"📚 [Nó: QA] Coletando dados brutos para: '{state['clean_query']}'...")
            
            @tool
            def buscar_manual(query: str) -> str:
                """
                Busca efeitos no manual da pedaleira (RAG).
                REGRA CRÍTICA DE USO: A 'query' DEVE ser extremamente curta, contendo APENAS palavras-chave técnicas em inglês. NUNCA use frases completas.
                CERTO: 'Pitch Shifter', 'Noise Gate threshold', 'Tube Screamer distortion'.
                ERRADO: 'Como eu faço para mudar a afinação', 'Qual efeito simula o Tube Screamer'.
                """
                return self.rag_system.search_effect_parameters(query=query, k=3)
                
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
                    print("🎧 [QA Agent] Escutando o áudio em background...")
                    blueprint_extraido = self.audio_worker.analyze_audio(state["audio_path"])
                    return f"A análise do áudio revelou os seguintes prováveis equipamentos:\n{blueprint_extraido.model_dump_json()}"
                
                ferramentas_qa.append(ouvir_audio_anexado)
            
            system_prompt = "Pesquise usando suas ferramentas e retorne os fatos brutos encontrados. Se o usuário perguntar sobre o áudio anexado, use a ferramenta de ouvir o áudio, ela retornará características que descrevem o conteúdo do áudio em texto."
            agente_qa = create_agent(self.llm_flash, ferramentas_qa, system_prompt=system_prompt)
            
            resposta = agente_qa.invoke({"messages": [HumanMessage(content=state["clean_query"])]})
            
            return {"agent_context": resposta["messages"][-1].content}


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
            
            route = state.get("route")
            
            system_base = (
                "Você é o ToneCrafter AI, um Guitar Tech virtual. "
                "Você receberá queries do usuário, e terá que respondê-lo de forma amigável e direta, sem ser infantil, mantendo um leve tom de seriedade."
                "Você receberá como contexto o histórico das mensagens, além do retorno de outros agentes especializados auxiliares para satisfazer o cliente."
                "Você é um agente conversacional, que pode além de conversar com o usuário, também responder dúvidas sobre guitar gear, além dos efeitos específicos da pedaleira do usuário."
                "Além disso, você também retornará, de forma limpa e concisa (usando markdown), as configurações para o preset da pedaleira do usuário, quando ele pedir para montar um."
                "No caso da montagem de presets, retorne uma lista com o nome do efeito na pedaleira, uma breve explicação de porque utilizá-lo, e as configurações do efeito."
                "Esses outros agentes auxiliares cuidarão de tarefas que não estão no seu alcance, como extrair informação de áudios, montar presets para a pedaleira buscando no manual de instruções do usuário, etc."
                "Utilize SEMPRE o retorno desses agentes auxiliares na sua resposta. Seu papel é manter a coerência entre as respostas para o usuário."
            )
            
            blueprint_str = state["blueprint"].model_dump_json() if state.get("blueprint") else "N/A"
            patch_str = state["patch"].model_dump_json() if state.get("patch") else "N/A"
            
            if route in ["web", "mockup", "audio"]:
                contexto_injetado = f"Você acabou de gerar um novo Patch na pedaleira do usuário.\nBlueprint:\n{blueprint_str}\nPatch Final:\n{patch_str}\nListe os efeitos ativados e justifique-os rapidamente."
            elif route == "qa":
                contexto_injetado = f"Seu agente de pesquisa encontrou estas informações nos manuais/internet: {state.get('agent_context')}. Apresente isso de forma clara e didática."
            else:
                contexto_injetado = "Apenas bata papo e responda à mensagem do usuário de forma natural e prestativa."

            responder_prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_base}\n\nCONTEXTO DA AÇÃO ATUAL:\n{contexto_injetado}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "Aja de acordo com o contexto e responda ao pedido original: {pedido}")
            ])

            historico = state["messages"][:-1] if "messages" in state else []
            
            response = (responder_prompt | self.llm_flash).invoke({
                "system_base": system_base,
                "contexto_injetado": contexto_injetado,
                "chat_history": historico,
                "pedido": state["clean_query"]
            })
            
            return {
                "final_response": response.content,
                "messages": [AIMessage(content=response.content)]
            }


        workflow.add_node("router", router_node)
        workflow.add_node("web_worker", web_node)
        workflow.add_node("mockup_worker", mockup_node)
        workflow.add_node("audio_worker", audio_node)
        workflow.add_node("setup_crafter", crafter_node)
        workflow.add_node("qa_node", qa_node)
        workflow.add_node("unified_responder", unified_responder_node)

        workflow.set_entry_point("router")

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
