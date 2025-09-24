import pickle
import numpy as np

# Load model, vectorizer, and classes
model = pickle.load(open("chatbot/model/chatbot_model.pkl", "rb"))
vectorizer = pickle.load(open("chatbot/model/vectorizer.pkl", "rb"))
classes = pickle.load(open("chatbot/model/classes.pkl", "rb"))

def predict_class(text):
    """Predict the intent class for a given text"""
    text = text.lower().split()  # same preprocessing as training
    text = " ".join(text)
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred

def get_response(intent_tag, data):
    """Get a random response for the predicted intent"""
    for intent in data["intents"]:
        if intent["tag"] == intent_tag:
            return np.random.choice(intent["responses"])
    return "I don't understand."

if __name__ == "__main__":
    import json
    with open("chatbot/intents.json", "r") as file:
        data = json.load(file)

    print("🤖 Chatbot is ready! Type 'quit' to exit.")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        intent_tag = predict_class(message)
        response = get_response(intent_tag, data)
        print(f"Chatbot: {response}")
