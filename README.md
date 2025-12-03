# Ollama LLM Chat App

A hybrid LLM application featuring a premium chat interface (Open WebUI) and a custom FastAPI backend for advanced data control and RAG capabilities.

## Architecture

-   **Frontend**: [Open WebUI](https://openwebui.com/) (Port 3000) - A feature-rich, self-hosted web interface.
-   **LLM Engine**: [Ollama](https://ollama.com/) (Local) - Runs the LLM models.
-   **Custom Backend**: FastAPI (Port 8000) - Your custom API for RAG, Agents, and data processing.
-   **Database**: MongoDB (Port 27017) - Stores data for the custom backend.

## Prerequisites

-   [Docker](https://www.docker.com/) and Docker Compose
-   [Ollama](https://ollama.com/) running locally (default port 11434)

## Quick Start

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd ollama_llm_chat_app
    ```

2.  **Start the application**:
    ```bash
    docker-compose up --build -d
    ```

3.  **Access the Services**:
    -   **Chat UI**: [http://localhost:3000](http://localhost:3000) (Create an admin account to start).
    -   **Custom API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs).

## Configuration

The application uses environment variables for configuration. These are set in `docker-compose.yml`:

-   `OLLAMA_BASE_URL`: URL for Open WebUI to connect to Ollama (default: `http://host.docker.internal:11434`).
-   `MONGO_URI`: Connection string for the Custom Backend (default: `mongodb://mongodb:27017/personalAI`).

## Development

To develop your custom backend (`fast_api.py`):

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run FastAPI locally**:
    ```bash
    uvicorn fast_api:app --reload
    ```

## License

[MIT](LICENSE)