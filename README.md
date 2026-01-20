# LangDiaryAgentic


# Prerequisites

- Python: 3.12+
- Package Manager: uv
- Hardware: NVIDIA GPU with CUDA support (Recommended: [e.g., 24GB+] VRAM). Confirmed to work with CUDA 12.9 on 2 V100S-PCIE-32GB GPUs.
- OS: Linux.


# Installation & Setup

```
1. Environment Setup
This project uses uv for fast dependency management.

# Install uv if not already installed
pip install uv

# Sync dependencies and create virtual environment (default: .venv)
uv sync

# Activate the environment
source .venv/bin/activate
```

##  API Keys & LLM Access

These descriptions below are applied in the followig cases,
1. want to use the LLMs that requires the accepting "terms" on the HuggingFace (such as Mistral or Llama),
2. want to change the storage path where the `transformers` package saves the model file.


Create an Access Token (Read permission) in your HF settings. Create a .env file in the project root:

```
HF_TOKEN='[your_huggingface_token_here]'

# Optional: Custom model cache location (default is ~/.cache)
# HF_HOME='/abs/path/to/large/storage'
```

# System Architecture


```mermaid
graph TD
    %% 1. The User Interface Layer
    subgraph "Frontend (Streamlit)"
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
        
        %% Step 1: Analyze & Retrieve
        Node_Retriever[<b>Node 1: Context Retriever</b><br/>Analyze draft topic<br/>Fetch relevant past errors]
        
        %% Step 2: Augment & Generate
        Node_Processor[<b>Node 2: Augmented Coach</b><br/>Inputs: Draft + <i>Retrieved History</i><br/>Action: Translate & Correct]
        
        %% Step 3: Extract & Store
        Node_Archivist[<b>Node 3: Archivist</b><br/>Identify <i>new</i> errors<br/>Format for storage]
    end

    %% The Flow
    UI_Input -->|1. Send Draft| Node_Retriever
    
    %% The RAG Loop (The Critical Part)
    Node_Retriever -->|"2. Query (e.g., 'Gender errors')"| VDB
    VDB -->|"3. Return Context (e.g., 'User struggles with household items')"| Node_Retriever
    
    Node_Archivist -->DDB
    
    Node_Retriever -->|4. Pass State: Draft + <b>Context</b>| Node_Processor
    
    %% Processing
    Node_Processor -->|5. Pass Corrected Text| Node_Archivist
    
    %% Closing the Loops
    Node_Archivist -->|6. Save New Errors| VDB
    Node_Archivist -->|7. Final Response| UI_Output

    %% Styling
    style Node_Processor fill:#f9f,stroke:#333,stroke-width:2px,color:black
    style VDB fill:#bbf,stroke:#333,stroke-width:2px,color:black
```



# Web App

A Web GUI to write the diary entries.

```
streamlit run ui/data_input_viewer.py
```

A Web GUI to view the diary entries.

```
streamlit run ui/data_viewer.py
```

