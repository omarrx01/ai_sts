# NovaTech AI Assistant — RAG Chatbot

An AI-powered customer support chatbot for a fictional electronics company, built using Retrieval-Augmented Generation (RAG), OpenRouter, and Streamlit.

## Overview

The assistant answers questions about NovaTech products, prices, specifications, delivery, and return policies using information retrieved from a PDF knowledge base.

The project demonstrates an end-to-end RAG pipeline: PDF extraction, text chunking, embedding generation, semantic retrieval, and AI-generated answers.

**Note:** NovaTech is a fictional company. All products, prices, and policies are examples.

## Technologies

- Python 3.11+
- uv — dependency management
- PyMuPDF — PDF text extraction
- NumPy — cosine similarity
- OpenRouter — embeddings and LLM API
- Streamlit — chatbot interface
- python-dotenv — environment variable management

## Project Structure

```text
AI_STS/
├── novatech_rag/
│   ├── data/
│   │   ├── NovaTech_Electronics_Knowledge_Base_v3.0_25pages_Detailed.pdf
│   │   ├── text_chunks.json
│   │   └── embedding_vectors.json
│   ├── extract.py
│   ├── chunk.py
│   ├── embed.py
│   ├── retrieve.py
│   ├── rag_engine.py
│   ├── app.py
│   └── test_rag.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/omarxr01/ai_sts.git
cd ai_sts
```

### 2. Install uv

Follow the installation instructions at https://docs.astral.sh/uv/getting-started/installation/

### 3. Install dependencies

```bash
uv sync --locked
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Never commit your actual API key to GitHub.

## Running the Application

From the project root:

```bash
uv run streamlit run novatech_rag/app.py
```

Open the local URL shown in the terminal, typically:

http://localhost:8501

## How the RAG Pipeline Works

1. **Extraction:** Extract text from the NovaTech PDF using PyMuPDF.
2. **Chunking:** Split extracted text into smaller, overlapping passages.
3. **Embeddings:** Convert text chunks into numerical vectors using `baai/bge-large-en-v1.5`.
4. **Retrieval:** Embed the user's question and identify relevant passages using cosine similarity and product-aware reranking.
5. **Generation:** Send the retrieved context to `openai/gpt-4o-mini` through OpenRouter to generate an answer.
6. **Interface:** Display answers and supporting source pages in Streamlit.

## Testing

Run the basic end-to-end RAG test:

```bash
uv run python novatech_rag/test_rag.py
```

Example questions:

- What is the price of NovaPhone S?
- What is the return policy?
- How long does delivery take?
- What are the specifications of NovaBook Air 14?

## Limitations

- Requires an internet connection and a valid OpenRouter API key.
- Answers depend on the quality of retrieved PDF passages.
- Chat history is displayed in the interface, but previous messages are not yet included in the LLM prompt.
- This is an educational demonstration, not a production customer support system.

## Author

Omar Haitham — Summit Academy AI/Big Data Training Project
