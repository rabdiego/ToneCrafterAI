from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.schemas import RouterDecision
from src.settings import settings

class SemanticRouterAgent:
    def __init__(self):
        llm = ChatGoogleGenerativeAI(model=settings.CONVERSATION_LLM_MODEL_NAME, temperature=0.2)
        self.structured_llm = llm.with_structured_output(RouterDecision)
        
        self.prompt = ChatPromptTemplate.from_messages([
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


    def route_request(self, state: dict) -> dict:
        """Processa o estado atual do grafo e toma a decisão de roteamento."""
        has_audio = bool(state.get("audio_path"))
        patch_atual = state.get("patch")
        user_text = state.get("user_input") or ""
        
        historico = ""
        if "messages" in state and len(state["messages"]) > 1:
            historico = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][-5:-1]])
        
        status_memoria = f"Patch Ativo: {patch_atual.overall_vibe}" if patch_atual else "NENHUM PATCH ATIVO."

        decision = (self.prompt | self.structured_llm).invoke({
            "historico": historico, 
            "status": status_memoria, 
            "user_text": user_text
        })
        
        if decision.intent.value == "create":
            if decision.sub_route.value == "audio":
                route_str = "audio" if has_audio else "mockup"
            else:
                route_str = decision.sub_route.value
        else:
            route_str = decision.intent.value
            
        query_limpa = decision.contextualized_query
        
        print(f"🔄 Consulta Traduzida: '{query_limpa}'")
        print(f"🔍 Busca Otimizada: '{decision.optimized_search_query}'")
        print(f"🎛️ Instrução de Áudio: '{decision.audio_instructions}'")
        print(f"🔀 Rota Decidida: [{route_str.upper()}]")
        
        return {
            "route": route_str, 
            "clean_query": query_limpa,
            "optimized_search_query": decision.optimized_search_query,
            "audio_instructions": decision.audio_instructions
        }

