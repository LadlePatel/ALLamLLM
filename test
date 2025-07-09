import streamlit as st
import time
import redis
import hashlib
import json
from typing import Any
from pydantic import Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.language_models import LLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from sentence_transformers import SentenceTransformer
import numpy as np
from PyPDF2 import PdfReader
import re

st.set_page_config(page_title="Allam Chatbot", layout="centered")
st.image("https://upload.wikimedia.org/wikipedia/commons/e/ee/Redis_logo.svg", width=150)

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

DEMO_USERS = [f"demo_user_{i+1}" for i in range(5)]
def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = DEMO_USERS[0]
    return st.session_state["session_id"]
def set_session_id(new_id):
    st.session_state["session_id"] = new_id
def get_redis_history(session_id):
    redis_url = "redis://localhost:6379"
    return RedisChatMessageHistory(url=redis_url, session_id=session_id)
@st.cache_resource
def load_chatbot():
    llm = ALLaMLLM(model_path="./allam-model")
    kb = KnowledgeBase()
    cache = SemanticCache()
    return llm, cache, kb

# ---- User/session selection - main view, not sidebar
with st.expander("👤 المستخدم / الجلسة", expanded=True):
    user_options = DEMO_USERS + ["مستخدم مخصص ..."]
    current_id = st.session_state.get("session_id", DEMO_USERS[0])
    selected_option = st.selectbox(
        "اختر مستخدم/جلسة",
        user_options,
        index=user_options.index(current_id) if current_id in user_options else len(user_options)-1,
        key="session_selector_main"
    )
    custom_id = st.text_input("أو أدخل معرف جلسة مخصص", value="" if current_id in DEMO_USERS else current_id, key="custom_session_id_main")
    if st.button("تغيير المستخدم/الجلسة"):
        if selected_option == "مستخدم مخصص ..." and custom_id.strip():
            set_session_id(custom_id.strip())
        else:
            set_session_id(selected_option)
        st.experimental_rerun()
    st.markdown(f"**معرف الجلسة الحالي:** `{get_session_id()}`")

if ("messages" not in st.session_state or
    st.session_state.get("last_session_id") != get_session_id()):
    redis_history = get_redis_history(get_session_id())
    st.session_state["messages"] = redis_history.messages
    st.session_state["last_session_id"] = get_session_id()

llm, cache, kb = load_chatbot()

st.markdown("<h2 style='text-align:center;'>💬 شات ALLaM</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.1rem;'>اسأل أي سؤال، ويمكن للروبوت استخدام قاعدة المعرفة كمصدر.</p>", unsafe_allow_html=True)

# ---- KB Manual Entry - main view
with st.expander("➕ أضف معلومة إلى قاعدة المعرفة", expanded=False):
    kb_id = st.text_input("مفتاح فريد للمعلومة", key="kb_id_mobile")
    kb_content = st.text_area("محتوى المعلومة", key="kb_content_mobile")
    if st.button("إضافة إلى قاعدة المعرفة"):
        if kb_id and kb_content:
            kb.add_entry(kb_id, kb_content)
            st.success(f"تمت الإضافة: {kb_id}")
        else:
            st.warning("يرجى إدخال كل من المفتاح والمحتوى.")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ---- KB File Upload - main view
with st.expander("📄 رفع ملف كقاعدة معرفة", expanded=False):
    uploaded_file = st.file_uploader("اختر ملف TXT أو PDF", type=["txt", "pdf"], key="kb_file_upload_mobile")
    chunk_size = st.number_input("حجم المقطع (أحرف)", min_value=100, max_value=2000, value=700)
    if uploaded_file is not None:
        if st.button("إضافة الملف إلى قاعدة المعرفة"):
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8")
            st.text_area("معاينة النص", value=text[:2000])
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            added_count = 0
            for idx, chunk in enumerate(chunks):
                clean_chunk = re.sub(r'[^\w\s\u0600-\u06FF]', '', chunk.strip())
                if clean_chunk:
                    kb_id = f"{uploaded_file.name}_chunk_{idx}"
                    kb.add_entry(kb_id, clean_chunk)
                    added_count += 1
            st.success(f"تمت إضافة {added_count} مقطع من {uploaded_file.name} إلى قاعدة المعرفة.")

# ---- Conversation Display - single column
for msg in st.session_state["messages"]:
    role = "user" if isinstance(msg, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(f"<span style='font-size:1.1rem'>{msg.content}</span>", unsafe_allow_html=True)

# ---- Main Chat Logic ----
if question := st.chat_input("اسأل سؤالاً ..."):
    redis_history = get_redis_history(get_session_id())

    msg = HumanMessage(content=question)
    redis_history.add_message(msg)
    st.session_state["messages"].append(msg)
    with st.chat_message("user"):
        st.markdown(f"<span style='font-size:1.15rem'>{question}</span>", unsafe_allow_html=True)

    with st.chat_message("ai"):
        start_time = time.time()
        cached_answer, distance = cache.get(question)
        kb_context = kb.get_best_entry(question)
        if kb_context:
            prompt = f"""أنت مساعد ذكاء اصطناعي ذكي ودقيق.
[معلومة من قاعدة المعرفة]: {kb_context}
سؤال: {question}
جواب:"""
        else:
            prompt = f"""أنت مساعد ذكاء اصطناعي ذكي ودقيق.
سؤال: {question}
جواب:"""

        if cached_answer:
            kb_used, question_asked, final_answer = parse_arabic_qa_output(cached_answer)
            st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)
            st.markdown("**📚 المعرفة المستخدمة:**")
            st.markdown(f"<div style='background-color:#f9f9f9; border-radius:7px; color:#000; padding:7px; margin-bottom:7px; direction:rtl;'>{kb_used if kb_used else 'لا توجد معلومة من قاعدة المعرفة.'}</div>", unsafe_allow_html=True)
            st.markdown("**❓ السؤال المرسل للنموذج:**")
            st.markdown(f"<div style='background-color:#f4f7fa; border-radius:7px; color:#000; padding:7px; margin-bottom:7px; direction:rtl;'>{question_asked if question_asked else question}</div>", unsafe_allow_html=True)
            st.markdown("**💡 الجواب:**")
            st.markdown(f"<div style='background-color:#d1ffd6; border-radius:7px; color:#000; padding:10px; direction:rtl; font-size:1.15rem;'>{final_answer}</div>", unsafe_allow_html=True)
            st.caption(
                f'<span style="color:green;">🔍</span> من الكاش (cosine: {distance:.2f})<br>'
                f'⏱️ الزمن: {time.time() - start_time:.2f} ثانية',
                unsafe_allow_html=True,
            )
            response = cached_answer
        else:
            full_output = llm(prompt)
            response = full_output.strip()
            kb_used, question_asked, final_answer = parse_arabic_qa_output(response)
            st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)
            st.markdown("**📚 المعرفة المستخدمة:**")
            st.markdown(f"<div style='background-color:#f9f9f9; border-radius:7px; color:#000; padding:7px; margin-bottom:7px; direction:rtl;'>{kb_used if kb_used else 'لا توجد معلومة من قاعدة المعرفة.'}</div>", unsafe_allow_html=True)
            st.markdown("**❓ السؤال المرسل للنموذج:**")
            st.markdown(f"<div style='background-color:#f4f7fa; border-radius:7px; color:#000; padding:7px; margin-bottom:7px; direction:rtl;'>{question_asked if question_asked else question}</div>", unsafe_allow_html=True)
            st.markdown("**💡 الجواب:**")
            st.markdown(f"<div style='background-color:#eaffea; border-radius:7px; color:#000; padding:10px; direction:rtl; font-size:1.15rem;'>{final_answer}</div>", unsafe_allow_html=True)
            st.caption(
                f"⏱️ الزمن: {time.time() - start_time:.2f} ثانية",
                unsafe_allow_html=True,
            )
            invalid_phrases = [
                "لا أعرف",
                "I do not know",
                question.strip(),
                "سؤال:",
                "جواب:"
            ]
            is_valid = True
            for phrase in invalid_phrases:
                if final_answer.strip() == phrase or final_answer.startswith(phrase):
                    is_valid = False
                    break
            if response != "" and is_valid:
                cache.set(question, response)
        ai_msg = AIMessage(content=response)
        redis_history.add_message(ai_msg)
        st.session_state["messages"].append(ai_msg)
