# 🎸 ToneCrafter AI

> A multi-agent system that translates tone descriptions, song names, or audio files into exact settings for your real-world pedalboard.

**ToneCrafter AI** is an application based on *Agentic AI* and *Retrieval-Augmented Generation* (RAG). It understands the user's sonic intent (whether it's "A dense and spatial fuzz" or "The solo from Comfortably Numb") and maps it mathematically to the physical limitations of the user's equipment, returning the exact effect blocks and knob settings.

## Architecture and Flow

The project adopts a microservices architecture (FastAPI + Streamlit) and uses **LangGraph** as a state machine to orchestrate a squad of specialist agents:

1. **Router Agent:** Evaluates the user's intent and directs the flow (Native LLM).
2. **Audio Extractor:** Analyzes `.wav`/`.mp3` files, extracting the spectral profile and deducing the generic signal chain.
3. **Web Searcher:** Uses the Tavily API to scour the internet for the actual gear used in specific studio recordings.
4. **Mockup Crafter:** Translates abstract ideas and audio concepts into a signal chain draft (Blueprint).
5. **Setup Crafter (The Brain):** Executes RAG on the user's equipment manual (via ChromaDB + HuggingFace Embeddings), cross-referencing the generic intent with the actual available effects.
6. **Synthesizer:** Drafts the final humanized response, justifying the choices made for the generated patch.

## Tech Stack

* **Agent Orchestration:** LangChain & LangGraph
* **Native Models:** Google Gemini 2.5 Pro & Flash
* **Web Search:** Tavily Search API
* **Vector Database & Embeddings:** ChromaDB & HuggingFace 
* **Backend:** FastAPI, Uvicorn (Asynchronous REST Architecture)
* **Frontend:** Streamlit

## How to Run Locally

### Prerequisites
* Python 3.10+
* Package manager `uv` (recommended) or `pip`
* API keys configured (Google Gemini and Tavily)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/rabdiego/ToneCrafterAI.git
cd ToneCrafterAI

```

2. Install dependencies:

```bash
uv sync

```

3. Configure environment variables:
Create a `.env` file in the project root following the `.env.example` structure:

```env
GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
# See the settings.py file for other model and path configurations

```

4. Run the application (Backend + Frontend):
Use the built-in `Makefile` to spin up the entire infrastructure with a single command:

```bash
make run

```

*The interface will be available at `http://localhost:8501` and the API documentation at `http://localhost:8000/docs`.*
