# Allam Chatbot: Private Arabic Chatbot with Knowledge Base & Redis

A full-featured, offline, mobile-friendly chatbot for Arabic and English, powered by ALLaM LLM, Redis, and Streamlit.
Supports semantic caching, multi-user history, and your own Knowledge Base via manual entry or file upload.

---

## 🚀 Features

* **Arabic & English Q\&A**: Natural chat, supports Knowledge Base augmentation.
* **Knowledge Base**: Add information or upload files (PDF/TXT) as context.
* **Semantic Caching**: Faster, smarter, and avoids repeating answers.
* **Multi-user Support**: Each user/session has their own saved chat history.
* **Mobile-Ready**: UI adapts for smartphones and desktop.


---

## 🖥️ 1. System Requirements

* Ubuntu/Debian Linux server (tested on Ubuntu 20.04+)
* Python 3.8+
* 12-16GB RAM recommended for 7B models
* Redis server (local)
* \[Optional] A HuggingFace account (for model download)

---

## 🛠️ 2. Setup Instructions

### **A. Install Prerequisites**

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip redis-server -y
```

### **B. Create and Activate a Virtual Environment**

```bash
python3 -m venv allam-chatbot
source allam-chatbot/bin/activate
pip install --upgrade pip
```

### **C. Install All Python Dependencies**

```bash
pip install torch transformers streamlit langchain langchain-community sentencepiece redis tiktoken chromadb fpdf PyPDF2 accelerate sentence-transformers huggingface_hub
```

### **D. Download the ALLaM LLM Model**

1. **(Recommended)** Log in to HuggingFace and download:

   ```bash
   huggingface-cli login   # Only needed the first time
   huggingface-cli download ALLaM-AI/ALLaM-7B-Instruct-preview --local-dir allam-model --local-dir-use-symlinks False
   ```

2. Or, download and unzip the model files manually into `./allam-model`.

---

### **E. Start Redis Server**

```bash
redis-server
```

*(This runs in the background by default on `localhost:6379`.)*

---

### **F. Add Your Project Files**

* Put your chatbot script (e.g., `allam_chatbot.py`) in the root project directory.
* Place your logos (`redis_logo.png`, `anb_logo.png`) in the same directory.
* (Optional) Place this `README.md` in the project root for reference.

---

### **G. Run the Chatbot App**

```bash
streamlit run allam_chatbot.py
```

* Open `http://localhost:8501` on your desktop **or phone** (on same network).
* Select or add a user. Type questions in Arabic or English.
* Add Knowledge Base entries via text or file upload for improved answers.
* The UI shows response time, KB context, and if an answer was served from cache.

---

## 🎨 Branding / Logos

To use custom logos (e.g., Redis and ANB), add PNG files named `redis_logo.png` and `anb_logo.png` in the project directory.
You can customize this in the script with:

```python
st.markdown(
    '''
    <div style="text-align: left;">
        <img src="redis_logo.png" width="80" style="vertical-align: middle;"/>
        <span style="display: inline-block; font-weight: bold; font-size: 22px; vertical-align: middle; margin: 0 15px;">
            Allam Chatbot
        </span>
        <img src="anb_logo.png" width="80" style="vertical-align: middle;"/>
    </div>
    '''
    , unsafe_allow_html=True
)
```

---

## ⚡ **One-liner: All Steps**

Copy-paste to go from zero to ready in minutes:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip redis-server -y
python3 -m venv allam-chatbot
source allam-chatbot/bin/activate
pip install --upgrade pip
pip install torch transformers streamlit langchain langchain-community sentencepiece redis tiktoken chromadb fpdf PyPDF2 sentence-transformers huggingface_hub
huggingface-cli login   # if needed
huggingface-cli download ALLaM-AI/ALLaM-7B-Instruct-preview --local-dir allam-model --local-dir-use-symlinks False
redis-server
streamlit run allam_chatbot.py
```

---

## 📚 **Dependencies**

* torch
* transformers
* streamlit
* langchain
* langchain-community
* sentencepiece
* redis
* tiktoken
* chromadb
* fpdf
* PyPDF2
* sentence-transformers
* huggingface\_hub

---

## 📝 **Notes**

* For GPU acceleration, ensure your `torch` install matches your CUDA version.
* Everything runs locally; no cloud needed, and your data stays private.
* Want to clear all chat history? Stop the app, flush Redis, and restart.
* Make sure your model is in `./allam-model`.

---

## 💬 **Support**

For issues or questions, open a GitHub issue or contact the maintainer.

---

**Enjoy your private Allam Chatbot!**
