from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import time
import redis
import hashlib
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models import LLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import re
import io
from pydantic import Field
from typing import Any

app = FastAPI(title="ALLaM Chatbot API", version="1.0.0")

# CORS middleware for React/Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://your-frontend-domain.com"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    question: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    kb_used: str
    question_asked: str
    final_answer: str
    cached: bool
    distance: Optional[float] = None
    response_time: float

class KBEntry(BaseModel):
    id: str
    content: str

class KBResponse(BaseModel):
    success: bool
    message: str

class SessionResponse(BaseModel):
    messages: List[dict]

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int

# Copy the same classes from your Streamlit app
def parse_arabic_qa_output(output):
    kb_match = re.search(r"\[معلومة من قاعدة المعرفة\]:(.*?)سؤال:", output, re.DOTALL)
    question_match = re.search(r"سؤال:(.*?)جواب:", output, re.DOTALL)
    answer_match = re.search(r"جواب:(.*)", output, re.DOTALL)

    kb = kb_match.group(1).strip() if kb_match else ""
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else output.strip()
    answer = answer.replace("أنت مساعد ذكاء اصطناعي ذكي ودقيق.", "").strip()
    return kb, question, answer

class ALLaMLLM(LLM):
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    
    def __init__(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        super().__init__(tokenizer=tokenizer, model=model)
    
    def _call(self, prompt: str, stop=None, run_manager=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    @property
    def _llm_type(self):
        return "custom_allam"

class SemanticCache:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def get(self, query, threshold=0.8):
        query_embedding = self.embedder.encode(query)
        cached = self.redis.hgetall("semantic_cache")
        for key, value in cached.items():
            entry = json.loads(value)
            cached_embedding = np.array(entry['embedding'])
            distance = np.dot(query_embedding, cached_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding))
            if distance >= threshold:
                return entry['answer'], distance
        return None, None
    
    def set(self, query, answer):
        embedding = self.embedder.encode(query).tolist()
        key = hashlib.sha256(query.encode()).hexdigest()
        self.redis.hset("semantic_cache", key, json.dumps({"question": query, "answer": answer, "embedding": embedding}))

class KnowledgeBase:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.kb_key = "knowledge_base"
    
    def add_entry(self, id, content):
        embedding = self.embedder.encode(content).tolist()
        self.redis.hset(self.kb_key, id, json.dumps({"content": content, "embedding": embedding}))
    
    def get_best_entry(self, query, threshold=0.7):
        query_embedding = self.embedder.encode(query)
        best_entry = None
        best_score = threshold
        for key, value in self.redis.hgetall(self.kb_key).items():
            entry = json.loads(value)
            kb_embedding = np.array(entry["embedding"])
            score = np.dot(query_embedding, kb_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(kb_embedding))
            if score > best_score:
                best_score = score
                best_entry = entry["content"]
        return best_entry

def get_redis_history(session_id):
    redis_url = "redis://localhost:6379"
    return RedisChatMessageHistory(url=redis_url, session_id=session_id)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Initialize the chatbot components
llm = ALLaMLLM(model_path="./allam-model")
cache = SemanticCache()
kb = KnowledgeBase()

# API Endpoints

@app.get("/")
async def root():
    return {"message": "ALLaM Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        start_time = time.time()
        
        # Get chat history
        redis_history = get_redis_history(request.session_id)
        
        # Add user message to history
        user_msg = HumanMessage(content=request.question)
        redis_history.add_message(user_msg)
        
        # Check cache first
        cached_answer, distance = cache.get(request.question)
        kb_context = kb.get_best_entry(request.question)
        
        # Prepare prompt
        if kb_context:
            prompt = f"""أنت مساعد ذكاء اصطناعي ذكي ودقيق.
[معلومة من قاعدة المعرفة]: {kb_context}
سؤال: {request.question}
جواب:"""
        else:
            prompt = f"""أنت مساعد ذكاء اصطناعي ذكي ودقيق.
سؤال: {request.question}
جواب:"""
        
        if cached_answer:
            # Use cached answer
            kb_used, question_asked, final_answer = parse_arabic_qa_output(cached_answer)
            response = cached_answer
            is_cached = True
        else:
            # Generate new answer
            full_output = llm(prompt)
            response = full_output.strip()
            kb_used, question_asked, final_answer = parse_arabic_qa_output(response)
            is_cached = False
            
            # Cache the response if valid
            invalid_phrases = [
                "لا أعرف",
                "I do not know",
                request.question.strip(),
                "سؤال:",
                "جواب:"
            ]
            is_valid = True
            for phrase in invalid_phrases:
                if final_answer.strip() == phrase or final_answer.startswith(phrase):
                    is_valid = False
                    break
            
            if response != "" and is_valid:
                cache.set(request.question, response)
        
        # Add AI message to history
        ai_msg = AIMessage(content=response)
        redis_history.add_message(ai_msg)
        
        response_time = time.time() - start_time
        
        return ChatResponse(
            answer=response,
            kb_used=kb_used if kb_used else 'لا توجد معلومة من قاعدة المعرفة.',
            question_asked=question_asked if question_asked else request.question,
            final_answer=final_answer,
            cached=is_cached,
            distance=distance,
            response_time=response_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/messages", response_model=SessionResponse)
async def get_session_messages(session_id: str):
    """Get chat history for a session"""
    try:
        redis_history = get_redis_history(session_id)
        messages = []
        
        for msg in redis_history.messages:
            messages.append({
                "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                "content": msg.content
            })
        
        return SessionResponse(messages=messages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/add", response_model=KBResponse)
async def add_kb_entry(entry: KBEntry):
    """Add a manual entry to the knowledge base"""
    try:
        kb.add_entry(entry.id, entry.content)
        return KBResponse(
            success=True,
            message=f"Entry '{entry.id}' added successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/upload", response_model=FileUploadResponse)
async def upload_file_to_kb(
    file: UploadFile = File(...),
    chunk_size: int = 700
):
    """Upload a file (TXT or PDF) to the knowledge base"""
    try:
        if file.content_type not in ["text/plain", "application/pdf"]:
            raise HTTPException(status_code=400, detail="Only TXT and PDF files are supported")
        
        # Read file content
        file_content = await file.read()
        
        if file.content_type == "application/pdf":
            # Handle PDF
            pdf_file = io.BytesIO(file_content)
            text = extract_text_from_pdf(pdf_file)
        else:
            # Handle TXT
            text = file_content.decode("utf-8")
        
        # Split into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        added_count = 0
        
        for idx, chunk in enumerate(chunks):
            clean_chunk = re.sub(r'[^\w\s\u0600-\u06FF]', '', chunk.strip())
            if clean_chunk:
                kb_id = f"{file.filename}_chunk_{idx}"
                kb.add_entry(kb_id, clean_chunk)
                added_count += 1
        
        return FileUploadResponse(
            success=True,
            message=f"File processed successfully",
            chunks_added=added_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session"""
    try:
        redis_history = get_redis_history(session_id)
        redis_history.clear()
        return {"success": True, "message": f"Session '{session_id}' cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
