.PHONY: run

.ONESHELL:
run:
	@echo "🚀 Iniciando o Backend (FastAPI)..."
	@uv run uvicorn api:app --host 0.0.0.0 --port 8000 & \
	API_PID=$$!; \
	trap "echo '\n🛑 Desligando a API e o Streamlit...'; kill $$API_PID; exit 0" SIGINT SIGTERM EXIT; \
	echo "⏳ Aguardando a API (e o HuggingFace) carregar na memória..."; \
	while ! curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs | grep -q "200"; do \
		sleep 2; \
	done; \
	echo "✅ API 100% pronta! Subindo a interface visual..."; \
	uv run streamlit run app.py --server.headless true