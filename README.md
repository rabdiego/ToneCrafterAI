# 🎸 ToneCrafter AI

> A production-grade multimodal agentic system that translates abstract sonic ideas, song references, or raw audio files into exact physical pedalboard configurations.

**ToneCrafter AI** goes beyond traditional LLM wrappers by implementing an advanced **Multi-Agent Architecture** and **Retrieval-Augmented Generation (RAG)**. It understands a user's sonic intent—whether it's "A dense spatial fuzz", a request for "Gilmour's Comfortably Numb solo", or an uploaded `.wav` file of a guitar riff—and mathematically maps it to the physical constraints of the user's specific gear, returning exact effect blocks and knob parameters.

## Key Technical Differentiators

This project tackles the most common pitfalls of GenAI in production through advanced software engineering patterns:

* **Multimodal Guardrails (Fail-Fast Design):** Prevents hallucinations and saves LLM tokens by short-circuiting off-topic text and voice-only audio requests right at the entry point, keeping the system strictly focused on music gear.
* **Router-Orchestrated Multi-Agent Pipeline:** Decouples intent routing, specialized processing, and response generation. The flow expands for parallel task execution and funnels back into a single "Unified Responder" to ensure state consistency and maintain a cohesive Guitar Tech persona.
* **QA Agent with Parallel Tool Calling:** Intelligently handles complex, multi-subject queries (e.g., comparing two different pedals) by executing parallel vector DB searches and web queries simultaneously, drastically reducing latency and avoiding semantic dilution in the RAG pipeline.
* **Semantic Query Rewriting:** Performs on-the-fly keyword optimization, stripping conversational filler from user prompts to maximize cosine similarity and retrieval accuracy in ChromaDB and Tavily search.
* **Sliding Window Memory:** Optimizes context window limits and costs by providing specialized agents with only the immediate conversational context, while the final synthesizer reads the broader history for narrative coherence.

## Architecture and Flow

The application is built on an asynchronous microservices architecture (FastAPI + Streamlit), utilizing **LangGraph** as a state machine to orchestrate isolated specialist agents:

1. **Guardrails Agent:** Validates text and decodes base64 audio to ensure the input is musically relevant.
2. **Semantic Router:** Evaluates the user's intent, rewrites queries for downstream tools, and dynamically directs the graph's edges.
3. **Specialized Workers (The Engine):**
* **Audio Extractor:** Multimodal agent that analyzes `.wav`/`.mp3` files, extracting spectral profiles to deduce the signal chain.
* **QA Agent:** Navigates the user's pedalboard manual (RAG) and the web autonomously.
* **Web Searcher:** Scours the internet for studio gear used in specific commercial tracks.
* **Mockup Crafter:** Translates abstract ideas into a logical signal chain draft.


4. **Setup Crafter:** Cross-references the drafted signal chain with the actual available effects in the local Vector Database, locking in exact parameters.
5. **Unified Responder:** Drafts the final Markdown response, synthesizing raw data into a friendly, professional format using conversational memory.

## Tech Stack

* **Agent Orchestration:** LangGraph, LangChain
* **Native Models:** Google Gemini 1.5 Pro & Flash (Structured Outputs & Multimodal capabilities)
* **Web Search:** Tavily Search API
* **Vector Database & Embeddings:** ChromaDB & HuggingFace (`all-MiniLM-L6-v2`)
* **Backend:** FastAPI, Uvicorn, Pydantic (Asynchronous REST & SSE Architecture)
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
