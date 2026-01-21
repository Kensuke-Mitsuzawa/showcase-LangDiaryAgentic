# LangDiaryAgentic

This project provides AI language learning assistance.

[![Watch the Demo](https://private-user-images.githubusercontent.com/1772712/538293801-8148441b-be67-4b2f-a8ae-1d3d8f31e1ac.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg5NTU2ODUsIm5iZiI6MTc2ODk1NTM4NSwicGF0aCI6Ii8xNzcyNzEyLzUzODI5MzgwMS04MTQ4NDQxYi1iZTY3LTRiMmYtYThhZS0xZDNkOGYzMWUxYWMucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI2MDEyMSUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNjAxMjFUMDAyOTQ1WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9NjA1YThhNjAyOGQ5MjQ5MzA4YzkzZGJlYzZjOWQzNTFmODU3MzFjYTE4NzQ3MjQxMTc4MjMyZDQ5YWM1YzE1ZSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.E-41PqJkcozbSIFflRGzUFK4yeXzBszK7_FEcXs3sys)](https://kensuke-mitsuzawa.github.io/portfolio/LangDiaryAgentic.mp4)


## Background

Writing diaries is supposed to be an effective methods to learn a language, especially the mid-level to the advanced language learner.

The language learner often encounter troubles in finding expressions and vocabularies that they want to express.
Looking up correct expressions are time-consuming and painful works.
This can be one of reasons that language learners give up writing diaries.

## Solution

This project help writing diaries with an expression in the language that the learner is familiar with.

For example, "Je suis [a girl]. J'ai [13 years old]". for french-learner whose native language is English.

# Main features

- AI suggestions of improving the vocabularies and expressions. 
- Keeping the diary and errors in the past.


# Technical description

Agentic LLM system is in charge of translating, grammar checker, and rewriting.
To reflect user's errors in past, RAG is used to suggest grammatical reviews.

System deployment can be done all locally.
No concern of security matters.

## Tech Stack

- AIs: LangChain, HuggingFace
- WepAPI: fastapi, streamlit
- DBs: ChromaDB, DuckDB



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

## Configuration

The app configurations are found at `lang_diary_agentic/configs.py`.


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


# Deployments

## LLM service

There are two choices: `local` or `server`.
Please switch the mode at `lang_diary_agentic/configs.py`.

### Local mode

Nothing special configuration is necessary.
Yet, the local machine must be equipped with GPUs.

### Server mode

1. `git clone` this project on the machine equipped with GPU devices.
2. setting up this project using `nv` command.
3. launching the API service with the following command,

```
# suppose at `./servers/
uvicorn server:app --host 0.0.0.0 --port 8000
```


## Web App

### Configurations

Check the configurations `lang_diary_agentic/configs.py`.

### Web App service

A Web GUI to write the diary entries.

```
streamlit run ui/data_input_viewer.py
```

A Web GUI to view the diary entries.

```
streamlit run ui/data_viewer.py
```

