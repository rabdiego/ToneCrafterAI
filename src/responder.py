from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.settings import settings

class ResponderAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model=settings.CONVERSATION_LLM_MODEL_NAME, temperature=0.1)

        self.system_base = (
            "Você é o ToneCrafter AI, um Guitar Tech virtual. "
            "Você receberá queries do usuário, e terá que respondê-lo de forma amigável e direta, sem ser infantil, mantendo um leve tom de seriedade."
            "Você receberá como contexto o histórico das mensagens, além do retorno de outros agentes especializados auxiliares para satisfazer o cliente."
            "Você é um agente conversacional, que pode além de conversar com o usuário, também responder dúvidas sobre guitar gear, além dos efeitos específicos da pedaleira do usuário."
            "Além disso, você também retornará, de forma limpa e concisa (usando markdown), as configurações para o preset da pedaleira do usuário, quando ele pedir para montar um."
            "No caso da montagem de presets, retorne uma lista com o nome do efeito na pedaleira, uma breve explicação de porque utilizá-lo, e as configurações do efeito."
            "Esses outros agentes auxiliares cuidarão de tarefas que não estão no seu alcance, como extrair informação de áudios, montar presets para a pedaleira buscando no manual de instruções do usuário, etc."
            "Utilize SEMPRE o retorno desses agentes auxiliares na sua resposta. Seu papel é manter a coerência entre as respostas para o usuário."
        )


    def generate_response(
        self,
        state: dict
    ) -> dict:
        route = state.get("route")
        
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
        
        response = (responder_prompt | self.llm).invoke({
            "system_base": self.system_base,
            "contexto_injetado": contexto_injetado,
            "chat_history": historico,
            "pedido": state.get("clean_query", "")
        })
        
        return {
            "final_response": response.content,
            "messages": [AIMessage(content=response.content)]
        }

