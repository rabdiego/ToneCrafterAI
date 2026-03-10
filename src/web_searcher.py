from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from src.settings import settings
from src.schemas import ToneBlueprint
from src.enricher import PedalEnricherAgent

class WebSearcherAgent:
    def __init__(self):
        load_dotenv()
        
        self.llm = ChatGoogleGenerativeAI(model=settings.WEB_SEARCHER_LLM_MODEL_NAME, temperature=0.1)
        
        self.search_tool = TavilySearch(
            max_results=5,
            name="web_search",
            description="Use para pesquisar artigos, fóruns e entrevistas sobre quais pedais e amplificadores um guitarrista usou."
        )
        
        researcher_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Você é um pesquisador de equipamentos musicais. O usuário vai pedir o timbre de uma música ou guitarrista. "
                "Use a web para descobrir EXATAMENTE quais guitarras, amplificadores e pedais foram usados na gravação ou turnê. "
                "Retorne um resumo detalhado contendo os modelos reais dos equipamentos e como estavam configurados."
                "Nota: Para os efeitos de Amplifier e Cabinet, esse SEMRE serão utilizados. Então certifique de saber os modelos que o guitarrista utilizou."
            )),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_tool_calling_agent(self.llm, [self.search_tool], researcher_prompt)
        self.researcher_executor = AgentExecutor(
            agent=agent, 
            tools=[self.search_tool], 
            verbose=False,
            max_iterations=5
        )
        
        formatter_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "Você é um engenheiro de áudio. Leia a pesquisa abaixo sobre os equipamentos usados por um guitarrista. "
                "Sua tarefa é extrair esses equipamentos e mapeá-los estritamente para as categorias permitidas do nosso sistema. "
                "Na chave 'description', cite o modelo real do pedal/amp encontrado na pesquisa (ex: 'Marshall Plexi 1959', 'Boss CE-2')."
                "Nas categorias de Preamp, temos efeitos como Boost, Autowah, etc."
                "Nas categorias de Distortion, temos efeitos como Overdrive, Distorção, Fuzz, etc."
                "As categorias Amplifier e Cabinet SEMPRE estarão presentes."
                "Nas categorias de Modulation, teremos Chorus, Phaser, Pitch shifter, etc."
                "Nota: SEMPRE retorne os efeitos de Amplifier e Cabinet."
            )),
            ("human", "Pesquisa bruta:\n{research_text}")
        ])
        
        self.formatter_chain = formatter_prompt | self.llm.with_structured_output(ToneBlueprint)


    def search_and_craft(
        self,
        query: str
    ) -> ToneBlueprint:
        try:
            research_result = self.researcher_executor.invoke({"query": query})
            raw_research = research_result["output"]
            blueprint = self.formatter_chain.invoke({"research_text": raw_research})
            enricher = PedalEnricherAgent()
            
            for nome_campo, slot in blueprint:
                if nome_campo == "overall_vibe":
                    continue
                    
                if slot.is_active:
                    if nome_campo in ["noise_reduction", "equalizer"]:
                        continue
                        
                    print(f"   -> Buscando perfil semântico para: {slot.description_or_name}...")
                    
                    perfil_acustico = enricher.enrich_profile(
                        pedal_name=slot.description_or_name, 
                        description=slot.settings,
                        category=nome_campo.upper() 
                    )
                    
                    slot.description_or_name = f"{slot.description_or_name}. Perfil Sonoro: {perfil_acustico}"
                
            return blueprint
            
        except Exception as e:
            raise e

