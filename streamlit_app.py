import streamlit as st
import json
import random
import pickle

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

# Streamlit UI
st.set_page_config(page_title="AI-FinanceBot", page_icon="💬")
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["Chatbot", "Loan Checker"])

if page == "Chatbot":
    st.title("💬 AI-FinanceBot")
    st.markdown("Ask me anything about banking, loans, KYC, or interest rates!")

    user_input = st.text_input("You:", placeholder="e.g. What are the current interest rates?")
    if user_input:
        response = get_response(user_input)
        st.markdown(f"**Bot:** {response}")

elif page == "Loan Checker":
    st.title("📊 Loan Eligibility Checker")
    st.markdown("Enter your details to check if you're eligible for a loan.")

    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    income = st.number_input("Monthly Income (€)", min_value=0, value=2000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

    if st.button("Check Eligibility"):
        result = check_loan_eligibility(age, income, credit_score)
        st.markdown(f"**Result:** {result}")
