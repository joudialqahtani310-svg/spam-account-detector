import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Spam & Bot Account Detector", page_icon="🤖", layout="wide")

@st.cache_resource
def train_and_get_model():
    data = {
        'text': [
            "Free follower boost click here now!", "Win cash prize instantly", "Follow back for back",
            "Get rich quick with crypto investments", "Buy followers cheap price", "Click link to claim free money",
            "Good morning everyone", "Deep learning is fascinating", "Just shared a new blog post",
            "Working on my graduation project today", "Great weather outside", "Data science and machine learning"
        ],
        'is_bot': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    model = RandomForestClassifier(random_state=42)
    model.fit(X, df['is_bot'])
    return model, vectorizer

model, vectorizer = train_and_get_model()

st.title("🤖 Multilingual Spam & Bot Account Detector")
st.write("Analyze text content or bio description to detect automated bot accounts and spam messages.")

user_text = st.text_area("Enter Tweet text or Account Bio to inspect:", "")

if st.button("Detect Spam / Bot"):
    if user_text.strip() == "":
        st.warning("Please enter text to analyze.")
    else:
        text_vector = vectorizer.transform([user_text])
        prediction = model.predict(text_vector)[0]
        proba = model.predict_proba(text_vector)[0][1]
        
        st.subheader("Analysis Results:")
        if prediction == 1 or proba > 0.5:
            st.error(f"🚨 Warning: This account/text is flagged as **Bot / Spam** (Risk Score: {proba*100:.1f}%)")
        else:
            st.success(f"✅ Safe: This account/text appears to be **Authentic** (Risk Score: {proba*100:.1f}%)")
