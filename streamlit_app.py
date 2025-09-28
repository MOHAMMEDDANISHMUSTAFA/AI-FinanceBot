"""
AI-FinanceBot - Enhanced (env vars, ML fraud detection, Chroma RAG, audit logging)

Requirements (install in your environment):
pip install streamlit python-dotenv scikit-learn joblib langdetect googletrans==4.0.0-rc1 requests chromadb sentence-transformers

Notes:
- Set environment variables (or create a .env file):
    PLAID_CLIENT_ID, PLAID_SECRET, PLAID_ACCESS_TOKEN
    CHROMA_PARENT_DIR (optional, for local chroma persistence)
    FRAUD_MODEL_PATH (optional, path to a pre-trained fraud model .pkl)
- If chromadb isn't running as a service, Chroma will run in-process using the default persistence dir.
"""

import os
import json
import random
import pickle
import math
import requests
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# NLP / embeddings / RAG
from langdetect import detect
from googletrans import Translator

# ML fraud
import joblib
import numpy as np

# Optional vector DB: chromadb + sentence-transformers
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# ----------------------------
# Load environment and config
# ----------------------------
load_dotenv()  # loads .env into environment variables (if present)

PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
PLAID_ACCESS_TOKEN = os.getenv("PLAID_ACCESS_TOKEN")

FRAUD_MODEL_PATH = os.getenv("FRAUD_MODEL_PATH", "models/fraud_model.pkl")
CHROMA_PARENT_DIR = os.getenv("CHROMA_PARENT_DIR", "chromadb_store")
AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit.log")

# ----------------------------
# Audit logging (JSON lines)
# ----------------------------
logger = logging.getLogger("ai_financebot_audit")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(AUDIT_LOG_PATH)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

def audit_log(username, role, page, query, response, sources=None):
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "username": username if username else "anonymous",
        "role": role if role else "unknown",
        "page": page,
        "query": query,
        "response": response,
        "sources": sources or []
    }
    logger.info(json.dumps(entry))

# ----------------------------
# User DB (kept simple, still demo)
# ----------------------------
USER_CREDENTIALS = {
    "admin": {"password": "pass123", "role": "admin"},
    "danish": {"password": "finance2025", "role": "user"},
    "guest": {"password": "welcome", "role": "guest"}
}

# ----------------------------
# Session state: login
# ----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_ui():
    st.title("🔐 Login to AI-FinanceBot (Secure Demo)")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = USER_CREDENTIALS.get(username)
        if user and user["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
            st.success(f"Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

# Logout handler
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# ----------------------------
# Translator & language utils
# ----------------------------
translator = Translator()

def detect_lang_safe(text):
    try:
        return detect(text)
    except Exception:
        return "en"

# ----------------------------
# Load classic ML-based chatbot (your existing pickles) safely
# ----------------------------
MODEL_DIR = Path("chatbot/model")
model = None
vectorizer = None
classes = None
intents_data = {}

try:
    model_path = MODEL_DIR / "chatbot_model.pkl"
    vec_path = MODEL_DIR / "vectorizer.pkl"
    classes_path = MODEL_DIR / "classes.pkl"
    intents_path = Path("chatbot/intents.json")

    if model_path.exists():
        model = pickle.load(open(model_path, "rb"))
    if vec_path.exists():
        vectorizer = pickle.load(open(vec_path, "rb"))
    if classes_path.exists():
        classes = pickle.load(open(classes_path, "rb"))
    if intents_path.exists():
        with open(intents_path, "r") as f:
            intents_data = json.load(f)
except Exception as e:
    st.warning("Could not load prebuilt model/vectorizer/classes; chatbot fallback will still run.")
    # continue; get_response will handle missing model gracefully

# ----------------------------
# Chroma RAG setup (policies/doc retrieval)
# ----------------------------
chroma_client = None
chroma_collection = None
embed_model = None

if CHROMA_AVAILABLE:
    try:
        # Use local persistent chroma (directory)
        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_PARENT_DIR)
        chroma_client = chromadb.Client(settings)
        # collection name
        chroma_collection = chroma_client.get_or_create_collection(name="bank_policies")
        # embedding model (sentence-transformers)
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # compact, fast
    except Exception as e:
        CHROMA_AVAILABLE = False
        st.warning(f"Chroma setup failed: {e}. RAG feature will fallback to internal docs.")

# Seed the RAG collection with small policy docs if empty (one-time)
def seed_rag_collection():
    if not CHROMA_AVAILABLE:
        return
    try:
        if chroma_collection.count() == 0:
            docs = [
                {"id": "kyc", "text": "All customers must complete KYC using valid ID and proof of address.", "meta": {"source": "KYC Policy"}},
                {"id": "loan", "text": "Loan approval requires credit score above 600 and income proof.", "meta": {"source": "Lending Guidelines"}},
                {"id": "compliance", "text": "Employees must report suspicious activity within 24 hours.", "meta": {"source": "Compliance Manual"}},
            ]
            texts = [d["text"] for d in docs]
            ids = [d["id"] for d in docs]
            metadatas = [d["meta"] for d in docs]
            embeddings = embed_model.encode(texts).tolist()
            chroma_collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
            chroma_client.persist()
    except Exception as e:
        st.error(f"Failed to seed RAG collection: {e}")

if CHROMA_AVAILABLE:
    seed_rag_collection()

# RAG retrieval
def rag_retrieve(query, top_k=3):
    """
    Returns list of dicts: [{"id":..., "document":..., "source":...}, ...]
    """
    if not CHROMA_AVAILABLE:
        return []
    emb = embed_model.encode([query]).tolist()
    results = chroma_collection.query(query_embeddings=emb, n_results=top_k, include=["documents", "metadatas", "ids"])
    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown")
        })
    return docs

# ----------------------------
# Pluggable fraud detection
# ----------------------------
# Keyword fallback
SUSPICIOUS_KEYWORDS = ["transfer", "lottery", "urgent", "refund", "crypto", "gift", "unknown"]

# Try to load a pre-trained fraud classifier (sklearn pipeline)
fraud_model = None
if Path(FRAUD_MODEL_PATH).exists():
    try:
        fraud_model = joblib.load(FRAUD_MODEL_PATH)
    except Exception as e:
        st.warning(f"Could not load FRAUD_MODEL_PATH: {e}")

def detect_fraud(description):
    """
    If fraud_model exists, it should accept raw text and return prediction probability or label.
    Otherwise fallback to keyword heuristic.
    """
    if not description or description.strip() == "":
        return "⚠️ No transaction description provided."
    if fraud_model:
        try:
            # model expected to implement predict_proba or predict
            if hasattr(fraud_model, "predict_proba"):
                proba = fraud_model.predict_proba([description])[0][1]  # probability of fraud
                if proba > 0.7:
                    return f"🚨 High-risk transaction detected (fraud score: {proba:.2f})."
                elif proba > 0.4:
                    return f"⚠️ Potential risk (fraud score: {proba:.2f}). Manual review recommended."
                else:
                    return f"✅ Low risk (fraud score: {proba:.2f})."
            else:
                pred = fraud_model.predict([description])[0]
                return "🚨 Alert: This transaction may be suspicious." if pred == 1 else "✅ No fraud detected."
        except Exception as e:
            # fallback
            pass

    # Fallback keyword approach (works as demo)
    if any(word in description.lower() for word in SUSPICIOUS_KEYWORDS):
        return "🚨 Alert: This transaction may be suspicious (keyword-based)."
    return "✅ No fraud detected."

# ----------------------------
# EMI, loan, account recommendation (unchanged logic but cleaned)
# ----------------------------
def calculate_emi(amount, rate, years):
    try:
        monthly_rate = rate / (12 * 100)
        months = years * 12
        if monthly_rate == 0:
            emi = amount / months
        else:
            emi = (amount * monthly_rate * math.pow(1 + monthly_rate, months)) / (math.pow(1 + monthly_rate, months) - 1)
        return round(emi, 2)
    except Exception:
        return None

def check_loan_eligibility(age, income, credit_score):
    if age < 18:
        return "❌ Not eligible: Must be at least 18 years old."
    if income < 1000:
        return "❌ Not eligible: Monthly income too low."
    if credit_score < 600:
        return "⚠️ Risky: Credit score below recommended threshold."
    return "✅ Eligible for loan consideration!"

def recommend_account(age, income, purpose):
    if age < 25 and purpose == "Education":
        return "🎓 Student Account"
    elif income > 5000 and purpose == "Business":
        return "🏢 Business Account"
    elif purpose == "Savings":
        return "💰 High-Interest Savings Account"
    else:
        return "🏦 Standard Checking Account"

# ----------------------------
# Masking & guardrails
# ----------------------------
def mask_sensitive(text):
    if not text:
        return text
    # simple masking rules; expand for production
    masked = text.replace("account", "****").replace("iban", "****").replace("name", "****")
    return masked

def is_source_backed(response, sources):
    # Consider it source-backed if sources present and non-empty
    return bool(sources)

# ----------------------------
# Chatbot: hybrid model + RAG usage
# ----------------------------
def get_response(user_input):
    lang = detect_lang_safe(user_input)
    translated = translator.translate(user_input, dest="en").text if lang != "en" else user_input
    # 1) If RAG relevant (policy keywords), use RAG retrieval
    rag_docs = rag_retrieve(translated, top_k=3) if CHROMA_AVAILABLE else []
    sources = [d["source"] for d in rag_docs] if rag_docs else []
    rag_answer = None
    if rag_docs:
        # Simple synth of retrieved docs + LLM placeholder (we'll concatenate)
        combined = " ".join([d["document"] for d in rag_docs])
        # We don't call an LLM here (avoid external calls in demo); we return combined as answer
        rag_answer = f"{combined} (source: {', '.join(sources)})"
        final_answer = translator.translate(rag_answer, dest=lang).text if lang != "en" else rag_answer
        return final_answer, sources

    # 2) Else fallback to classic ML-intent-based chatbot (if available)
    if model and vectorizer:
        try:
            X = vectorizer.transform([translated.lower()])
            prediction = model.predict(X)[0]
            for intent in intents_data.get("intents", []):
                if intent.get("tag") == prediction:
                    reply = random.choice(intent.get("responses", ["Sorry, I don't know."]))
                    final = translator.translate(reply, dest=lang).text if lang != "en" else reply
                    return final, []
        except Exception:
            pass

    # 3) Generic fallback
    fallback = "Sorry, I don't have the exact answer right now. Please refine your question or contact support."
    return fallback, []

# ----------------------------
# Banking API (Plaid sandbox) - secure pattern
# ----------------------------
def get_account_data():
    url = "https://sandbox.plaid.com/accounts/get"
    headers = {"Content-Type": "application/json"}
    payload = {
        "client_id": PLAID_CLIENT_ID or "REDACTED",
        "secret": PLAID_SECRET or "REDACTED",
        "access_token": PLAID_ACCESS_TOKEN or "REDACTED"
    }
    try:
        # If keys missing, return friendly error
        if not all([PLAID_CLIENT_ID, PLAID_SECRET, PLAID_ACCESS_TOKEN]):
            return [{"error": "Plaid credentials not configured (use env vars)."}]
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("accounts", [])
    except Exception as e:
        return [{"error": str(e)}]

# ----------------------------
# Internal assistant (now uses RAG)
# ----------------------------
def internal_assistant(query):
    # Use RAG retrieval
    docs = rag_retrieve(query, top_k=3) if CHROMA_AVAILABLE else []
    if docs:
        response = " ".join([d["document"] for d in docs])
        sources = [d["source"] for d in docs]
        if not is_source_backed(response, sources):
            return "⚠️ Cannot answer without verified source.", sources
        return mask_sensitive(response), sources
    # fallback to simple doc lookup (old behavior)
    docs_map = {
        "kyc": ("All customers must complete KYC using valid ID and proof of address.", "KYC Policy"),
        "loan": ("Loan approval requires credit score above 600 and income proof.", "Lending Guidelines"),
        "compliance": ("Employees must report suspicious activity within 24 hours.", "Compliance Manual")
    }
    for key in docs_map:
        if key in query.lower():
            response, src = docs_map[key]
            return mask_sensitive(response), [src]
    return "⚠️ No matching policy found. Please refine your query.", []

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI-FinanceBot", page_icon="💼", layout="wide")

if not st.session_state.logged_in:
    login_ui()
else:
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", [
        "Chatbot", "Loan Checker", "Account Recommendation",
        "Fraud Alert", "EMI Calculator", "Banking Dashboard",
        "Internal Assistant", "Admin"
    ])

    username = st.session_state.get("username", "anonymous")
    role = st.session_state.get("role", "guest")

    if page == "Chatbot":
        st.title("💬 AI-FinanceBot (Chat)")
        st.markdown("Ask me anything about banking, loans, KYC, or interest rates! (RAG-enabled)")
        user_input = st.text_input("You:", placeholder="e.g. What are the KYC requirements?")
        if user_input:
            response, sources = get_response(user_input)
            st.markdown(f"**Bot:** {response}")
            audit_log(username, role, "Chatbot", user_input, response, sources)

    elif page == "Loan Checker":
        st.title("📊 Loan Eligibility Checker")
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        income = st.number_input("Monthly Income (€)", min_value=0, value=2000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        if st.button("Check Eligibility"):
            result = check_loan_eligibility(age, income, credit_score)
            st.markdown(f"**Result:** {result}")
            audit_log(username, role, "Loan Checker", f"age={age},income={income},score={credit_score}", result)

    elif page == "Account Recommendation":
        st.title("🧭 Account Recommendation Engine")
        age = st.number_input("Your Age", min_value=0, max_value=100, value=30)
        income = st.number_input("Monthly Income (€)", min_value=0, value=3000)
        purpose = st.selectbox("Primary Purpose", ["Savings", "Business", "Education", "Daily Use"])
        if st.button("Recommend Account"):
            result = recommend_account(age, income, purpose)
            st.markdown(f"**Recommended Account:** {result}")
            audit_log(username, role, "Account Recommendation", f"age={age},income={income},purpose={purpose}", result)

    elif page == "Fraud Alert":
        st.title("🚨 Fraud Alert Assistant (ML-enhanced)")
        description = st.text_input("Transaction Description", placeholder="e.g. urgent crypto refund")
        if st.button("Check for Fraud"):
            result = detect_fraud(description)
            st.markdown(f"**Result:** {result}")
            audit_log(username, role, "Fraud Alert", description, result)

    elif page == "EMI Calculator":
        st.title("💸 Loan EMI Calculator")
        amount = st.number_input("Loan Amount (€)", min_value=0, value=10000)
        rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=7.5)
        years = st.number_input("Loan Duration (Years)", min_value=1, value=5)
        if st.button("Calculate EMI"):
            emi = calculate_emi(amount, rate, years)
            st.markdown(f"**Monthly EMI:** €{emi}")
            audit_log(username, role, "EMI Calculator", f"amount={amount},rate={rate},years={years}", f"emi={emi}")

    elif page == "Banking Dashboard":
        st.title("🏦 Banking Dashboard")
        st.markdown("Fetching account data from Plaid sandbox...")
        accounts = get_account_data()
        for acc in accounts:
            st.markdown(f"**{acc.get('name', 'Account')}**")
            st.write(f"Type: {acc.get('type', 'N/A')}")
            st.write(f"Subtype: {acc.get('subtype', 'N/A')}")
            st.write(f"Balance: €{acc.get('balances', {}).get('current', 'N/A')}")
            st.write("---")
        audit_log(username, role, "Banking Dashboard", "fetch_accounts", json.dumps(accounts))

    elif page == "Internal Assistant":
        st.title("🧠 Internal AI Assistant (RAG)")
        if role != "admin":
            st.warning("Access restricted to admin users.")
        query = st.text_input("Ask about policies, compliance, or legal guidelines:")
        if st.button("Get Answer"):
            answer, sources = internal_assistant(query)
            st.markdown(f"**Answer:** {answer}")
            if sources:
                st.markdown(f"**Sources:** {', '.join(sources)}")
            audit_log(username, role, "Internal Assistant", query, answer, sources)

    elif page == "Admin":
        st.title("⚙️ Admin & Diagnostics")
        st.markdown("Environment & diagnostics (sensitive values hidden):")
        st.write({
            "PLAID_CONFIGURED": bool(PLAID_CLIENT_ID and PLAID_SECRET and PLAID_ACCESS_TOKEN),
            "CHROMA_AVAILABLE": CHROMA_AVAILABLE,
            "FRAUD_MODEL_LOADED": bool(fraud_model),
            "AUDIT_LOG_PATH": AUDIT_LOG_PATH
        })
        if st.button("Show last 10 audit logs"):
            if Path(AUDIT_LOG_PATH).exists():
                with open(AUDIT_LOG_PATH, "r") as f:
                    lines = f.readlines()[-10:]
                    for line in lines:
                        try:
                            obj = json.loads(line)
                            st.json(obj)
                        except:
                            st.write(line)
            else:
                st.info("No audit log file found yet.")
