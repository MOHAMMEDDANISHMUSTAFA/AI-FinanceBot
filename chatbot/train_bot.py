import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
with open("chatbot/intents.json", "r") as file:
    data = json.load(file)

# Prepare training data
corpus = []
labels = []
classes = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = pattern.lower().split()
        tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
        corpus.append(" ".join(tokens))
        labels.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
y = labels

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
os.makedirs("chatbot/model", exist_ok=True)
pickle.dump(model, open("chatbot/model/chatbot_model.pkl", "wb"))
pickle.dump(vectorizer, open("chatbot/model/vectorizer.pkl", "wb"))
pickle.dump(classes, open("chatbot/model/classes.pkl", "wb"))

print("✅ Chatbot model trained and saved.")
