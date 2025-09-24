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

# Streamlit UI
st.set_page_config(page_title="AI-FinanceBot", page_icon="💬")
st.title("💬 AI-FinanceBot")
st.markdown("Ask me anything about banking, loans, KYC, or interest rates!")

user_input = st.text_input("You:", placeholder="e.g. What are the current interest rates?")
if user_input:
    response = get_response(user_input)
    st.markdown(f"**Bot:** {response}")
