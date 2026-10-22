# Session 1 Assignment — Text Summarization with Ollama

## Overview
This project uses a locally hosted Large Language Model (LLM) to summarize text using Python and Ollama.

## Technologies Used
- Python
- Ollama
- Llama 3.2 (3B)
- uv (Python package manager)

## How It Works
1. The user provides text to summarize.
2. Python sends the text to the locally running Ollama model.
3. Llama 3.2 generates a concise summary.
4. The summary is displayed in the terminal.

## Setup Instructions

1. Install Ollama.
2. Download the model:

   `ollama pull llama3.2:3b`

3. Install the project dependencies:

   `uv sync`

4. Run the application:

   `uv run python ollama_summary.py`

## Files
- `ollama_summary.py` — Text summarization application
- `.env.example` — Example environment variables
- `.gitignore` — Files excluded from Git
- `README.md` — Project documentation

## Notes
Ollama runs locally and does not require an API key for local inference.