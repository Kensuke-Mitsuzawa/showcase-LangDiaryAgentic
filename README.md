LangDiary: Agentic RAG Platform for Mixed-Language Learning
An offline-first, privacy-preserving AI pipeline enabling "Code-Switching" for language acquisition.

<div align="center"> <a href="https://kensuke-mitsuzawa.github.io/portfolio/LangDiaryAgentic.mp4"> <img src="https://private-user-images.githubusercontent.com/1772712/538293801-8148441b-be67-4b2f-a8ae-1d3d8f31e1ac.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg5NTU2ODUsIm5iZiI6MTc2ODk1NTM4NSwicGF0aCI6Ii8xNzcyNzEyLzUzODI5MzgwMS04MTQ4NDQxYi1iZTY3LTRiMmYtYThhZS0xZDNkOGYzMWUxYWMucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDEyMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjAxMjFUMDAyOTQ1WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NjA1YThhNjAyOGQ5MjQ5MzA4YzkzZGJlYzZjOWQzNTFmODU3MzFjYTE4NzQ3MjQxMTc4MjMyZDQ5YWM1YzE1ZSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.E-41PqJkcozbSIFflRGzUFK4yeXzBszK7_FEcXs3sys" alt="Watch the Demo" width="800"/> </a> <p><em>Click the image to watch the full architecture demo.</em></p> </div>

---

# 🚀 The Engineering Challenge
Intermediate language learners often hit a plateau where they cannot express complex thoughts in their target language, leading to frustration and abandonment of practice. Traditional tools (DeepL, ChatGPT) offer simple translation but lack contextual awareness of the user's past mistakes.

**LangDiary** solves this by engineering a **Code-Switching Interface**. It allows users to write naturally by mixing their native language (L1) into the target language (L2) syntax.


### Example Input (English $\rightarrow$ French):

> "Je suis [a girl]. J'ai [13 years old]."

### System Output:

> "Je suis une fille. J'ai 13 ans."

(Context Retrieved: User previously confused 'être' and 'avoir' for age. System highlights correction.)

# 🏗 System Architecture

This project moves beyond simple LLM wrapping by implementing a Multi-Agent RAG Pipeline using **LangGraph**. 
The system is designed for local deployment on consumer-grade GPUs, ensuring total data privacy.


```mermaid

graph TD

%% 1. The User Interface Layer

subgraph "Frontend (Flask)"

UI_Input[User Input: Mixed-Lang Draft]

UI_Output[Display: Corrected Text + Personalized Tips]

end

  

%% 2. DBs

subgraph "Memory (Chroma Vector DB)"

VDB[(Error Embeddings)]

end

subgraph "DuckDB"

DDB[(Diary DB)]

end

%% 3. The Agentic Pipeline (LangGraph)

subgraph "Agentic Pipeline (LangGraph)"

%% Step 1: Retrieve

Node_Retriever[<b>Node 1: Context Retriever</b><br/>Analyze draft topic<br/>Fetch relevant past errors]

%% Step 2: Translation

Node_Translator[<b>Node 2: Translator</b><br/>Inputs: placeholder<br/>Action: Translate]

%% Step 3: Error analysis & Store

Node_Archivist[<b>Node 3: Archivist</b><br/>Identify <i>new</i> errors<br/>Format for storage]

%% Step 4: Rewriting
Node_Rewriter[<b>Node 4: Rewriter</b><br/>Rewriting the draft<br/>]

%% Step 5: Review
Node_Reviewer[<b>Node 5: Reviewer</b><br/>Review the draft<br/>]

end


%% The Flow

UI_Input -->|1. Send Draft| Node_Retriever

%% The RAG Loop (The Critical Part)

Node_Retriever -->|"2. Query (e.g., 'Gender errors')"| VDB

VDB -->|"3. Return Context (e.g., 'User struggles with household items')"| Node_Retriever
Node_Retriever -->|4. Pass State: Draft + <b>Context</b>| Node_Translator

%% Processing

Node_Translator -->|5. Pass Corrected Text| Node_Archivist
Node_Archivist -->|6. Save New Errors| Node_Rewriter
Node_Rewriter --> Node_Reviewer

%% Closing the Loops
Node_Reviewer -->DDB
Node_Reviewer -->|6. Save New Errors| VDB
Node_Reviewer -->|7. Final Response| UI_Output

  

%% Styling

style VDB fill:#bbf,stroke:#333,stroke-width:2px,color:black
```

## Key Technical Features
- Agentic State Machine: Uses LangGraph to orchestrate a 5-step loop (Retrieve $\to$ Translate $\to$ Archive $\to$ Rewrite $\to$ Review), ensuring higher accuracy than zero-shot prompting.
- Hybrid Memory Architecture:
    - Short-term: Context window management.
    - Long-term: ChromaDB stores embeddings of specific user error patterns (e.g., gender mismatches, wrong prepositions).
    - Analytical: DuckDB enables fast SQL querying of diary metadata.
- Privacy-First: Fully compatible with local weights (Mistral, Llama 3) via HuggingFace transformers.

# 🛠 Tech Stack

| Layer | Technology | Reason for Choice |
| ----- | ---------- | ----------------- |
| Orchestration | LangChain / LangGraph | For building cyclic graph-based agent workflows. |
| LLM Backend | HuggingFace Transformers | To run open-weights models locally without API costs.|
| API Layer | FastAPI | High-performance async backend. |
| Frontend | Flask | Rapid prototyping of interactive data/chat interfaces.|
| Vector Store | ChromaDB | Lightweight, local vector search for RAG.|
|Analytics DB | DuckDB | In-process SQL OLAP for diary analysis. |
| Package Mgmt | uv | Extremely fast Python dependency management (Rust-based). |


# ⚡ Installation & Setup

## Prerequisites

- OS: Linux (Recommended)
- Python: 3.12+
- Hardware: NVIDIA GPU (Recommended: 24GB+ VRAM for comfortable local inference). Verified on: 2x NVIDIA V100S (32GB).
- Package Manager: uv (for fast setup).

## 1. Environment Setup

This project uses `uv` for modern, reliable dependency management.

```
# Install uv (if not installed)
pip install uv

# Sync dependencies and create virtual environment
uv sync

# Activate the environment
source .venv/bin/activate
```

## 2. Configuration

`.env` controls the variables.

Please copy the `env_base` and make `.env`.


# 🚀 Deployment

The system supports dual-mode deployment via `configs.py`.
Please jump to "Deployment with Docker" when you prefer to use the docker container.


## A. Local Mode (Standalone)

Ideal for single-user privacy. No complex setup required other than GPU drivers.

## B. Server Mode (API)

Decouples the heavy inference engine from the UI.

### Launch the Inference API:

```
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Launch the Frontend Interfaces:
 
```
% at ./my_diary_app directory
python app.py
```

## 🚀 Deployment with Docker

The LLM server container is launched with the command (i.e. on the GPU server),

#### Container on server

```bash
docker-compose --profile server up --build
```

#### Container on client

Please edit the `.env` (create it if not exists).
The file is 

```
Server_API_Endpoint="http://0.0.0.0:8000"
```


The client web app is launched with command (i.e. on the laptop).


```bash
docker-compose --profile client up --build
```



# 🛡 Security Note

This architecture allows for 100% data sovereignty. 
When running in Local Mode, no diary entries or user data ever leave the machine, making it suitable for sensitive personal journals or enterprise training environments.

