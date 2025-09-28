import streamlit as st
import json
import random
import pickle
import math
import requests
import os
from langdetect import detect
from googletrans import Translator
from datetime import datetime

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# User database with roles
USER_CREDENTIALS = {
    "admin": {"password": "pass123", "role": "admin"},
    "danish": {"password": "finance2025", "role": "user"},
    "guest": {"password": "welcome", "role": "guest"}
}

# Login function
def login():
    st.title("🔐 Login to AI-FinanceBot")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = USER_CREDENTIALS.get(username)
        if user and user["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid credentials")

# Logout
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

# Load model and data
model = pickle.load(open("chatbot/model/chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("chatbot/model/vectorizer.pkl", "rb"))
classes = pickle.load(open("chatbot/model/classes.pkl", "rb"))
with open("chatbot/intents.json", "r") as file:
    data = json.load(file)

# Translator
translator = Translator()

# GDPR masking
def mask_sensitive(text):
    return text.replace("account", "****").replace("iban", "****").replace("name", "****")

# Guardrails
def is_source_backed(response):
    return "source:" in response.lower()

# Audit logging
def log_interaction(user, role, query, response):
    with open("chat_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | {user} ({role}) | Q: {query} | A: {response}\n")

# Multilingual chatbot (merged with internal assistant)
def get_response(user_input):
    lang = detect(user_input)
    translated = translator.translate(user_input, dest="en").text.lower()
    
    # RAG-lite docs
    docs = {
        "kyc": "All customers must complete KYC using valid ID and proof of address. Source: KYC Policy",
        "loan": "Loan approval requires credit score above 600 and income proof. Source: Lending Guidelines",
        "compliance": "Employees must report suspicious activity within 24 hours. Source: Compliance Manual"
    }
    for key in docs:
        if key in translated:
            response = docs[key]
            if not is_source_backed(response):
                response = "⚠️ Cannot answer without verified source."
            response = mask_sensitive(response)
            final = translator.translate(response, dest=lang).text
            log_interaction(st.session_state.username, st.session_state.role, user_input, final)
            return final

    # Intent-based response
    X = vectorizer.transform([translated])
    prediction = model.predict(X)[0]
    for intent in data["intents"]:
        if intent["tag"] == prediction:
            reply = random.choice(intent["responses"])
            final = translator.translate(reply, dest=lang).text
            log_interaction(st.session_state.username, st.session_state.role, user_input, final)
            return final

    # Fallback
    final = translator.translate("⚠️ Sorry, I don't understand that.", dest=lang).text
    log_interaction(st.session_state.username, st.session_state.role, user_input, final)
    return final

# Loan eligibility
def check_loan_eligibility(age, income, credit_score):
    if age < 18:
        return "❌ Not eligible: Must be at least 18 years old."
    if income < 1000:
        return "❌ Not eligible: Monthly income too low."
    if credit_score < 600:
        return "⚠️ Risky: Credit score below recommended threshold."
    return "✅ Eligible for loan consideration!"

# Account recommendation
def recommend_account(age, income, purpose):
    if age < 25 and purpose == "Education":
        return "🎓 Student Account"
    elif income > 5000 and purpose == "Business":
        return "🏢 Business Account"
    elif purpose == "Savings":
        return "💰 High-Interest Savings Account"
    else:
        return "🏦 Standard Checking Account"

# Fraud detection
def detect_fraud(description):
    suspicious_keywords = ["transfer", "lottery", "urgent", "refund", "crypto", "gift", "unknown"]
    if any(word in description.lower() for word in suspicious_keywords):
        return "🚨 Alert: This transaction may be suspicious."
    return "✅ No fraud detected."

# EMI calculator
def calculate_emi(amount, rate, years):
    monthly_rate = rate / (12 * 100)
    months = years * 12
    emi = (amount * monthly_rate * math.pow(1 + monthly_rate, months)) / (math.pow(1 + monthly_rate, months) - 1)
    return round(emi, 2)

# Banking API (Plaid sandbox)
def get_account_data():
    url = "https://sandbox.plaid.com/accounts/get"
    headers = {"Content-Type": "application/json"}
    payload = {
        "client_id": os.getenv("PLAID_CLIENT_ID", "demo_client"),
        "secret": os.getenv("PLAID_SECRET", "demo_secret"),
        "access_token": os.getenv("PLAID_ACCESS_TOKEN", "demo_token")
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        return data.get("accounts", [])
    except Exception as e:
        return [{"error": str(e)}]

# Internal assistant (admin only, but RAG moved to chatbot too)
def internal_assistant(query):
    return get_response(query)

# UI
st.set_page_config(page_title="AI-FinanceBot", page_icon="💼")

if not st.session_state.logged_in:
    login()
else:
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("Go to", [
        "Chatbot", "Loan Checker", "Account Recommendation",
        "Fraud Alert", "EMI Calculator", "Banking Dashboard",
        "Internal Assistant"
    ])

    if page == "Chatbot":
        st.title("💬 AI-FinanceBot")
        st.markdown("Ask me anything about banking, loans, KYC, or interest rates!")
        user_input = st.text_input("You:", placeholder="e.g. What are the current interest rates?")
        if user_input:
            response = get_response(user_input)
            st.markdown(f"**Bot:** {response}")

    elif page == "Loan Checker":
        st.title("📊 Loan Eligibility Checker")
        age = st.number_input("Age", min_value=0, max_value=100, value=25)
        income = st.number_input("Monthly Income (€)", min_value=0, value=2000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        if st.button("Check Eligibility"):
            result = check_loan_eligibility(age, income, credit_score)
            st.markdown(f"**Result:** {result}")

    elif page == "Account Recommendation":
        st.title("🧭 Account Recommendation Engine")
        age = st.number_input("Your Age", min_value=0, max_value=100, value=30)
        income = st.number_input("Monthly Income (€)", min_value=0, value=3000)
        purpose = st.selectbox("Primary Purpose", ["Savings", "Business", "Education", "Daily Use"])
        if st.button("Recommend Account"):
            result = recommend_account(age, income, purpose)
            st.markdown(f"**Recommended Account:** {result}")

    elif page == "Fraud Alert":
        st.title("🚨 Fraud Alert Assistant")
        description = st.text_input("Transaction Description", placeholder="e.g. urgent crypto refund")
        if st.button("Check for Fraud"):
            result = detect_fraud(description)
            st.markdown(f"**Result:** {result}")

    elif page == "EMI Calculator":
        st.title("💸 Loan EMI Calculator")
        amount = st.number_input("Loan Amount (€)", min_value=0, value=10000)
        rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, value=7.5)
        years = st.number_input("Loan Duration (Years)", min_value=1, value=5)
        if st.button("Calculate EMI"):
            emi = calculate_emi(amount, rate, years)
            st.markdown(f"**Monthly EMI:** €{emi}")

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

    elif page == "Internal Assistant":
        if st.session_state.role != "admin":
            st.warning("Access restricted to admin users.")
        else:
            st.title("🧠 Internal AI Assistant")
            query = st.text_input("Ask about policies, compliance, or legal guidelines:")
            if st.button("Get Answer"):
                result = internal_assistant(query)
                st.markdown(f"**Answer:** {result}")
