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
from enum import Enum

app = FastAPI(title="ALLaM Multilingual Chatbot API", version="1.0.0")

# CORS middleware for React/Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Language enum for supported languages
class Language(str, Enum):
    ARABIC = "ar"
    ENGLISH = "en"

# Language-specific prompts and responses
LANGUAGE_PROMPTS = {
    Language.ARABIC: {
        "system_prompt": "أنت مساعد ذكاء اصطناعي ذكي ودقيق. يرجى الإجابة باللغة العربية.",
        "kb_prefix": "[معلومة من قاعدة المعرفة]:",
        "question_prefix": "سؤال:",
        "answer_prefix": "جواب:",
        "no_kb_message": "لا توجد معلومة من قاعدة المعرفة.",
        "thinking_prefix": "دعني أفكر في هذا السؤال..."
    },
    Language.ENGLISH: {
        "system_prompt": "You are an intelligent and accurate AI assistant. Please respond in English.",
        "kb_prefix": "[Knowledge Base Information]:",
        "question_prefix": "Question:",
        "answer_prefix": "Answer:",
        "no_kb_message": "No information found in knowledge base.",
        "thinking_prefix": "Let me think about this question..."
    }
}

# Updated Pydantic models
class ChatRequest(BaseModel):
    question: str
    session_id: str
    language: Language = Language.ARABIC  # Default to Arabic

class ChatResponse(BaseModel):
    answer: str
    kb_used: str
    question_asked: str
    final_answer: str
    cached: bool
    language: Language
    distance: Optional[float] = None
    response_time: float

class KBEntry(BaseModel):
    id: str
    content: str
    language: Optional[Language] = Language.ARABIC

class KBResponse(BaseModel):
    success: bool
    message: str

class SessionResponse(BaseModel):
    messages: List[dict]

class FileUploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int

class LanguageDetectionResponse(BaseModel):
    detected_language: Language
    confidence: float

# Fixed parsing function for multiple languages
def parse_qa_output(output: str, language: Language):
    lang_config = LANGUAGE_PROMPTS[language]
    
    # Use literal string matching instead of regex to avoid escaping issues
    kb_prefix = lang_config['kb_prefix']
    question_prefix = lang_config['question_prefix']
    answer_prefix = lang_config['answer_prefix']
    
    kb = ""
    question = ""
    answer = output.strip()
    
    # Find KB information
    if kb_prefix in output:
        kb_start = output.find(kb_prefix) + len(kb_prefix)
        if question_prefix in output:
            kb_end = output.find(question_prefix)
            if kb_end > kb_start:
                kb = output[kb_start:kb_end].strip()
    
    # Find question
    if question_prefix in output:
        q_start = output.find(question_prefix) + len(question_prefix)
        if answer_prefix in output:
            q_end = output.find(answer_prefix)
            if q_end > q_start:
                question = output[q_start:q_end].strip()
    
    # Find answer
    if answer_prefix in output:
        a_start = output.find(answer_prefix) + len(answer_prefix)
        answer = output[a_start:].strip()
    
    # Clean up system prompt from answer
    answer = answer.replace(lang_config["system_prompt"], "").strip()
    
    return kb, question, answer

class ALLaMLLM(LLM):
    tokenizer: Any = Field(default=None, exclude=True)
    model: Any = Field(default=None, exclude=True)
    
    def __init__(self, model_path: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            super().__init__(tokenizer=tokenizer, model=model)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback: create a mock model for testing
            super().__init__(tokenizer=None, model=None)
    
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs):
        # Check if model is loaded
        if self.model is None or self.tokenizer is None:
            # Return a mock response for testing
            return f"Mock response for: {prompt[:50]}..."
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=kwargs.get('max_tokens', 200),
                temperature=kwargs.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            print(f"Error in model generation: {e}")
            return f"Error generating response: {str(e)}"
    
    @property
    def _llm_type(self):
        return "custom_allam"

class SemanticCache:
    def __init__(self, redis_host='localhost', redis_port=6379):
        try:
            self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            # Test Redis connection
            self.redis.ping()
        except Exception as e:
            print(f"Warning: Redis connection failed. Using mock cache: {e}")
            self.redis = None
            self.embedder = None
    
    def get(self, query, language: Language, threshold=0.8):
        if self.redis is None or self.embedder is None:
            return None, None
        
        try:
            cache_key = f"semantic_cache_{language.value}"
            query_embedding = self.embedder.encode(query)
            cached = self.redis.hgetall(cache_key)
            
            for key, value in cached.items():
                entry = json.loads(value)
                cached_embedding = np.array(entry['embedding'])
                distance = np.dot(query_embedding, cached_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                )
                if distance >= threshold:
                    return entry['answer'], distance
            return None, None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None, None
    
    def set(self, query, answer, language: Language):
        if self.redis is None or self.embedder is None:
            return
        
        try:
            embedding = self.embedder.encode(query).tolist()
            key = hashlib.sha256(query.encode()).hexdigest()
            cache_key = f"semantic_cache_{language.value}"
            self.redis.hset(
                cache_key, 
                key, 
                json.dumps({
                    "question": query, 
                    "answer": answer, 
                    "embedding": embedding,
                    "language": language.value
                })
            )
        except Exception as e:
            print(f"Cache set error: {e}")

class KnowledgeBase:
    def __init__(self, redis_host='localhost', redis_port=6379):
        try:
            self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.kb_key = "knowledge_base"
            # Test Redis connection
            self.redis.ping()
        except Exception as e:
            print(f"Warning: Redis connection failed. Using mock KB: {e}")
            self.redis = None
            self.embedder = None
    
    def add_entry(self, id, content, language: Language = Language.ARABIC):
        if self.redis is None or self.embedder is None:
            return
        
        try:
            embedding = self.embedder.encode(content).tolist()
            self.redis.hset(
                self.kb_key, 
                id, 
                json.dumps({
                    "content": content, 
                    "embedding": embedding,
                    "language": language.value
                })
            )
        except Exception as e:
            print(f"KB add error: {e}")
    
    def get_best_entry(self, query, language: Language, threshold=0.7):
        if self.redis is None or self.embedder is None:
            return None
        
        try:
            query_embedding = self.embedder.encode(query)
            best_entry = None
            best_score = threshold
            
            for key, value in self.redis.hgetall(self.kb_key).items():
                entry = json.loads(value)
                kb_embedding = np.array(entry["embedding"])
                score = np.dot(query_embedding, kb_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(kb_embedding)
                )
                
                # Boost score if language matches
                if entry.get("language") == language.value:
                    score *= 1.1
                    
                if score > best_score:
                    best_score = score
                    best_entry = entry["content"]
                    
            return best_entry
        except Exception as e:
            print(f"KB get error: {e}")
            return None

def get_redis_history(session_id):
    try:
        redis_url = "redis://localhost:6379"
        return RedisChatMessageHistory(url=redis_url, session_id=session_id)
    except Exception as e:
        print(f"Redis history error: {e}")
        # Return a mock history object
        class MockHistory:
            def __init__(self):
                self.messages = []
            def add_message(self, msg):
                self.messages.append(msg)
            def clear(self):
                self.messages = []
        return MockHistory()

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""

def detect_language(text: str) -> Language:
    """Simple language detection for Arabic and English"""
    try:
        # Arabic script detection
        if re.search(r'[\u0600-\u06FF]', text):
            return Language.ARABIC
        
        # Default to English
        return Language.ENGLISH
    except Exception:
        return Language.ENGLISH

# Initialize the chatbot components with error handling
try:
    llm = ALLaMLLM(model_path="./allam-model")
except Exception as e:
    print(f"LLM initialization error: {e}")
    llm = ALLaMLLM(model_path="")  # Will use mock mode

cache = SemanticCache()
kb = KnowledgeBase()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "ALLaM Multilingual Chatbot API is running"}

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "languages": [
            {"code": lang.value, "name": lang.name} 
            for lang in Language
        ]
    }

@app.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_text_language(text: str):
    """Detect language of input text"""
    try:
        detected = detect_language(text)
        return LanguageDetectionResponse(
            detected_language=detected,
            confidence=0.8
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Language detection error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with language support"""
    try:
        start_time = time.time()
        
        # Get language configuration
        lang_config = LANGUAGE_PROMPTS[request.language]
        
        # Get chat history
        redis_history = get_redis_history(request.session_id)
        
        # Add user message to history
        user_msg = HumanMessage(content=request.question)
        redis_history.add_message(user_msg)
        
        # Check cache first (language-specific)
        cached_answer, distance = cache.get(request.question, request.language)
        kb_context = kb.get_best_entry(request.question, request.language)
        
        # Prepare language-specific prompt
        if kb_context:
            prompt = f"""{lang_config['system_prompt']}
{lang_config['kb_prefix']} {kb_context}
{lang_config['question_prefix']} {request.question}
{lang_config['answer_prefix']}"""
        else:
            prompt = f"""{lang_config['system_prompt']}
{lang_config['question_prefix']} {request.question}
{lang_config['answer_prefix']}"""
        
        if cached_answer:
            # Use cached answer
            kb_used, question_asked, final_answer = parse_qa_output(cached_answer, request.language)
            response = cached_answer
            is_cached = True
        else:
            # Generate new answer
            full_output = llm(prompt)
            response = full_output.strip()
            kb_used, question_asked, final_answer = parse_qa_output(response, request.language)
            is_cached = False
            
            # Cache the response if valid
            invalid_phrases = [
                "لا أعرف", "I do not know",
                request.question.strip()
            ]
            is_valid = True
            for phrase in invalid_phrases:
                if final_answer.strip() == phrase or final_answer.startswith(phrase):
                    is_valid = False
                    break
            
            if response != "" and is_valid:
                cache.set(request.question, response, request.language)
        
        # Add AI message to history
        ai_msg = AIMessage(content=response)
        redis_history.add_message(ai_msg)
        
        response_time = time.time() - start_time
        
        return ChatResponse(
            answer=response,
            kb_used=kb_used if kb_used else lang_config['no_kb_message'],
            question_asked=question_asked if question_asked else request.question,
            final_answer=final_answer if final_answer else response,
            cached=is_cached,
            language=request.language,
            distance=distance,
            response_time=response_time
        )
        
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")

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
        raise HTTPException(status_code=500, detail=f"Session retrieval error: {str(e)}")

@app.post("/knowledge-base/add", response_model=KBResponse)
async def add_kb_entry(entry: KBEntry):
    """Add a manual entry to the knowledge base"""
    try:
        kb.add_entry(entry.id, entry.content, entry.language)
        return KBResponse(
            success=True,
            message=f"Entry '{entry.id}' added successfully in {entry.language.value}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KB add error: {str(e)}")

@app.post("/knowledge-base/upload", response_model=FileUploadResponse)
async def upload_file_to_kb(
    file: UploadFile = File(...),
    language: Language = Language.ARABIC,
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
        
        # Auto-detect language if not specified
        if language == Language.ARABIC:  # Default case
            detected_lang = detect_language(text)
            language = detected_lang
        
        # Split into chunks
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        added_count = 0
        
        for idx, chunk in enumerate(chunks):
            clean_chunk = re.sub(r'[^\w\s\u0600-\u06FF\u0900-\u097F]', '', chunk.strip())
            if clean_chunk:
                kb_id = f"{file.filename}_{language.value}_chunk_{idx}"
                kb.add_entry(kb_id, clean_chunk, language)
                added_count += 1
        
        return FileUploadResponse(
            success=True,
            message=f"File processed successfully in {language.value}",
            chunks_added=added_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session"""
    try:
        redis_history = get_redis_history(session_id)
        redis_history.clear()
        return {"success": True, "message": f"Session '{session_id}' cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session clear error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
