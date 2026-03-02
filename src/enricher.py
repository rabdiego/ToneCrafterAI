import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from src.settings import settings

class PedalEnricherAgent:
    def __init__(self):
        load_dotenv()
        
        self.llm = ChatGoogleGenerativeAI(model=settings.ENRICHER_LLM_MODEL_NAME, temperature=0.1)
        
        self.search_tool = TavilySearch(
            max_results=3,
            name="web_search",
            description="Use para buscar as características sonoras (tone, EQ, ganho) de um pedal de guitarra."
        )
        self.tools = [self.search_tool]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Você é um engenheiro de áudio e especialista em timbres de guitarra. "
                "Sua tarefa é ler o nome e a descrição de um efeito de pedaleira, pesquisar na web se necessário, "
                "e retornar ESTRITAMENTE uma ou duas frases curtas descrevendo o seu perfil sonoro "
                "(ex: 'Fuzz agressivo com mid-scoop', 'Overdrive cremoso focado em médios'). "
                "Não inclua introduções, apenas o perfil sonoro."
            )),
            ("human", "Nome do Efeito: {pedal_name}\nDescrição no Manual: {description}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=False,
            max_iterations=3
        )

    
    async def aenrich_profile(self, pedal_name: str, description: str, category: str) -> str:
        utility_categories = ["Noise Reduction", "Equalizer", "Cabinet"]
        if category in utility_categories:
            return "Utility effect. Modifies dynamics or frequencies without adding organic saturation."

        print(f"🔍 [Enricher] Buscando perfil para: {pedal_name}...")
        
        try:
            response = await self.agent_executor.ainvoke({
                "pedal_name": pedal_name,
                "description": description
            })
            return response["output"]
        except Exception as e:
            print(f"⚠️ Erro ao enriquecer {pedal_name}: {e}")
            return "Perfil sonoro genérico baseado no manual."
    

    def enrich_profile(self, pedal_name: str, description: str, category: str) -> str:
        utility_categories = ["Noise Reduction", "Equalizer", "Cabinet"]
        if category in utility_categories:
            return "Utility effect. Modifies dynamics or frequencies without adding organic saturation."

        print(f"🔍 [Enricher] Buscando perfil para: {pedal_name}...")
        
        try:
            response = self.agent_executor.invoke({
                "pedal_name": pedal_name,
                "description": description
            })
            return response["output"]
        except Exception as e:
            print(f"⚠️ Erro ao enriquecer {pedal_name}: {e}")
            return "Perfil sonoro genérico baseado no manual."

