import json
import random
import pickle
import os

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

# Chat loop
print("🤖 AI-FinanceBot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Bot: Goodbye! Stay financially smart 💼")
        break
    response = get_response(user_input)
    print("Bot:", response)
