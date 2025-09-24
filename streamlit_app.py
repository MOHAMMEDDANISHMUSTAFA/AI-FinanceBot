import streamlit as st
import json
import random
import pickle
import math

# Load model and data
model = pickle.load(open("chatbot/model/chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("chatbot/model/vectorizer.pkl", "rb"))
classes = pickle.load(open("chatbot/model/classes.pkl", "rb"))

with open("chatbot/intents.json", "r") as file:
    data = json.load(file)

def get_response(user_input):
    X = vectorizer.transform([user_input.lower()])
    prediction = model.predict(X)[0]
    
    for intent in data["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])

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

def detect_fraud(description):
    suspicious_keywords = ["transfer", "lottery", "urgent", "refund", "crypto", "gift", "unknown"]
    if any(word in description.lower() for word in suspicious_keywords):
        return "🚨 Alert: This transaction may be suspicious."
    return "✅ No fraud detected."

def calculate_emi(amount, rate, years):
    monthly_rate = rate / (12 * 100)
    months = years * 12
    emi = (amount * monthly_rate * math.pow(1 + monthly_rate, months)) / (math.pow(1 + monthly_rate, months) - 1)
    return round(emi, 2)

# Streamlit UI
st.set_page_config(page_title="AI-FinanceBot", page_icon="💼")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Loan Checker", "Account Recommendation", "Fraud Alert", "EMI Calculator"])

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
