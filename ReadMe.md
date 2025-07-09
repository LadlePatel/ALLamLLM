# ALLamLLM: Private Multilingual Chatbot API with Knowledge Base & Redis

A robust, private FastAPI-powered chatbot for Arabic and English, powered by ALLaM LLM, Redis, and HuggingFace. Supports semantic caching, session-based multi-user chat history, and a flexible Knowledge Base (manual entry or file upload).

---

## 🚀 Features

- **Arabic & English Q&A**: Natural chat, with Knowledge Base augmentation.
- **Knowledge Base**: Add information or upload files (PDF/TXT) as context.
- **Semantic Caching**: Faster, smarter, and avoids repeating answers.
- **Session Support**: Each user/session has their own saved chat history.
- **API-first Architecture**: Easily integrate with your own UI or tools.
- **Runs Locally**: Your data stays private.

---

## 🖥️ 1. System Requirements

- Ubuntu/Debian Linux server (tested on Ubuntu 20.04+)
- Python 3.8+
- 12-16GB RAM recommended for 7B models
- Redis server (local)
- [Optional] HuggingFace account (for model download)

---

## 🛠️ 2. Setup Instructions

### A. Install Prerequisites

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip redis-server -y
```

### B. Create and Activate a Virtual Environment

```bash
python3 -m venv allamllm-env
source allamllm-env/bin/activate
pip install --upgrade pip
```

### C. Install Python Dependencies

```bash
pip install torch transformers fastapi uvicorn langchain langchain-community sentencepiece redis tiktoken chromadb fpdf PyPDF2 accelerate sentence-transformers huggingface_hub
```

### D. Download the ALLaM LLM Model

1. **(Recommended)** Log in to HuggingFace and download:

    ```bash
    huggingface-cli login   # Only needed the first time
    huggingface-cli download ALLaM-AI/ALLaM-7B-Instruct-preview --local-dir allam-model --local-dir-use-symlinks False
    ```

2. Or, download and unzip the model files manually into `./allam-model`.

---

### E. Start Redis Server

```bash
redis-server
```
*(Usually runs in the background by default on `localhost:6379`.)*

---

### F. Add Your Project Files

- Place your main FastAPI script (`main.py`) in the project root directory.
- (Optional) Place this `ReadMe.md` in the project root for reference.

---

### G. Run the FastAPI Application

```bash
uvicorn main:app --reload
```

- The API documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)
- You can interact with endpoints for chat, knowledge base management, and session history.

---

## 📚 API Overview

Key endpoints (see `/docs` for full details):

- `POST /chat` — Chat with the model (with semantic cache and KB support)
- `GET /languages` — List supported languages
- `POST /detect-language` — Detect language of input text
- `GET /sessions/{session_id}/messages` — Retrieve chat history for a session
- `DELETE /sessions/{session_id}` — Clear chat history for a session
- `POST /knowledge-base/add` — Manually add a KB entry
- `POST /knowledge-base/upload` — Upload PDF/TXT files to the KB
- `GET /health` — Health check endpoint

---

## ⚡ One-liner: All Steps

Copy-paste to go from zero to ready in minutes:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip redis-server -y
python3 -m venv allamllm-env
source allamllm-env/bin/activate
pip install --upgrade pip
pip install torch transformers fastapi uvicorn langchain langchain-community sentencepiece redis tiktoken chromadb fpdf PyPDF2 sentence-transformers huggingface_hub
huggingface-cli login   # if needed
huggingface-cli download ALLaM-AI/ALLaM-7B-Instruct-preview --local-dir allam-model --local-dir-use-symlinks False
redis-server
uvicorn main:app --reload
```

---

## 📚 Dependencies

- torch
- transformers
- fastapi
- uvicorn
- langchain
- langchain-community
- sentencepiece
- redis
- tiktoken
- chromadb
- fpdf
- PyPDF2
- accelerate
- sentence-transformers
- huggingface_hub

---

## 📝 Notes

- For GPU acceleration, ensure your `torch` install matches your CUDA version.
- Everything runs locally; no cloud needed, your data stays private.
- Want to clear all chat history? Stop the app, flush Redis, and restart.
- Make sure your model is in `./allam-model`.

---

## 💬 Support

For issues or questions, open a GitHub issue or contact the maintainer.

---

**Enjoy your private ALLamLLM API chatbot!**
