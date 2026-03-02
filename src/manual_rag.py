import os
import re
import asyncio
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.enricher import PedalEnricherAgent
from src.settings import settings

effects_dict = {
    'PRE' : 'Preamp',
    'DST' : 'Distortion',
    'AMP' : 'Amplifier',
    'NR'  : 'Noise Reduction',
    'CAB' : 'Cabinet',
    'EQ'  : 'Equalizer',
    'MOD' : 'Modulation',
    'DELAY' : 'Delay',
    'REVERB' : 'Reverb'
}

class PedalboardRAG:
    def __init__(self, persist_directory: str = "chroma_db"):
        load_dotenv()
        
        print("Loading local embeddings model (HuggingFace)...")
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL,
            model_kwargs={'device':'cpu'}
        )
        
        self.persist_directory = persist_directory
        self.collection_name = "pedal_manuals"
        
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings_model,
            persist_directory=self.persist_directory
        )

    
    async def _aparse_markdown_tables(self, markdown_text: str) -> list[Document]:
        enricher = PedalEnricherAgent()
        extracted_items = []
        current_category = "Unknown_Category"
        
        lines = markdown_text.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('# '):
                raw_category = line[2:].strip()
                current_category = effects_dict.get(raw_category, raw_category)
                continue
            
            if line.startswith('|') and line.endswith('|'):
                if re.match(r'^\|[-|\s]+\|$', line) or 'FX Title' in line:
                    continue
                
                parts = [p.strip() for p in line.split('|')[1:-1]]
                if len(parts) >= 3:
                    extracted_items.append({
                        "category": current_category,
                        "fx_title": parts[0],
                        "description": parts[1],
                        "parameters": parts[2]
                    })

        semaphore = asyncio.Semaphore(3)
        
        async def process_item(item: dict) -> Document:
            async with semaphore:
                await asyncio.sleep(1) 
                
                acoustic_profile = await enricher.aenrich_profile(
                    item["fx_title"], 
                    item["description"], 
                    item["category"]
                )
                
                page_content = (
                    f"Effect Category: {item['category']}\n"
                    f"Effect name: {item['fx_title']}\n"
                    f"Manual Description: {item['description']}\n"
                    f"Acoustic Profile: {acoustic_profile}\n"
                    f"Available parameters: {item['parameters']}"
                )
                
                return Document(
                    page_content=page_content,
                    metadata={
                        "Category": item['category'],
                        "Effect": item['fx_title']
                    }
                )

        print(f"🚀 Iniciando enriquecimento assíncrono de {len(extracted_items)} efeitos...")
        
        tasks = [process_item(item) for item in extracted_items]
        documents = await asyncio.gather(*tasks)
        
        return documents


    async def aingest_markdown_manual(self, markdown_path: str) -> int:
        print(f"Loading and unfolding tables from {markdown_path}...")
        
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_document = f.read()

        chunks = await self._aparse_markdown_tables(markdown_document)
        
        if not chunks:
            print("No individual effect found. Check file's structure.")
            return 0

        print(f"💾 Ingesting {len(chunks)} individual effects into Vector Database...")
        self.vector_store.add_documents(documents=chunks)
        
        return len(chunks)


    def search_effect_parameters(self, query: str, k: int = 5) -> str:
        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            return "No relevant information found in the manual on this effects."
        
        context_str = "=== EFFECTS FOUND (ORDERED BY RELEVANCE) ===\n\n"
        for i, doc in enumerate(results):
            context_str += f"[{i+1} MOST SIMILAR] {doc.metadata['Category']} -> {doc.metadata['Effect']}\n"
            context_str += f"{doc.page_content}\n"
            context_str += "-" * 50 + "\n"
            
        return context_str


if __name__ == "__main__":
    async def main():
        rag_system = PedalboardRAG()
        print(rag_system.search_effect_parameters('Efeito de auto wah.'))

    asyncio.run(main())

