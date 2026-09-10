import streamlit as st
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

st.set_page_config(page_title="Multilingual Bot & Spam Account Detector", page_icon="🤖", layout="wide")

@st.cache_resource
def load_assets():
    data = {
        'text': [
            "Click the link now and WIN !!", "Free followers instantly", "Buy cheap crypto now",
            "اشتري المتابعين واللايكات بأرخص الأسعار", "ربح سريع اضغط الرابط",
            "Working on my graduation project today", "Great research paper published", "Enjoying the weather",
            "مشروع الذكاء الاصطناعي مكتمل بنجاح", "صباح الخير جميعاً"
        ],
        'followers_count': [40, 10, 5, 12, 8, 800, 1200, 450, 950, 1500],
        'following_count': [3000, 4500, 2900, 4900, 5000, 310, 400, 290, 510, 600],
        'tweet_count': [12145, 8000, 15000, 6000, 9500, 320, 210, 150, 430, 890],
        'account_age': [31, 15, 10, 20, 5, 365, 730, 500, 400, 600],
        'is_bot': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    vectorizer = TfidfVectorizer(max_features=100)
    text_features = vectorizer.fit_transform(df['text']).toarray()
    
    meta_features = df[['followers_count', 'following_count', 'tweet_count', 'account_age']].values
    X = np.hstack((text_features, meta_features))
    y = df['is_bot']
    
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X, y)
    
    return model, vectorizer

model, vectorizer = load_assets()

# Header
st.title("🤖 Multilingual Bot & Spam Account Detector")
st.write("Real-time behavioral and text-pattern detection system.")

st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Profile Metadata")
    followers = st.number_input("Follower Count", min_value=0, value=40)
    following = st.number_input("Following Count", min_value=0, value=3000)
    tweets = st.number_input("Total Tweet Count", min_value=0, value=12145)
    account_age = st.number_input("Account Age (Days)", min_value=1, value=31)
    
    analyze_btn = st.button("Analyze Account", type="primary")

with col2:
    st.subheader("2. Recent Post Content")
    user_text = st.text_area("Post Text (Arabic or English)", "Click the link now and WIN !!", height=150)

if analyze_btn:
    st.markdown("---")
    st.subheader("Analysis Output")
    
    # Feature Calculation
    text_vec = vectorizer.transform([user_text]).toarray()
    meta_vec = np.array([[followers, following, tweets, account_age]])
    input_features = np.hstack((text_vec, meta_vec))
    
    proba = model.predict_proba(input_features)[0][1]
    
    # Rule Checkers for Risk Indicators
    indicators = []
    
    # Metadata Checks
    if following > 0 and (followers / following) < 0.2:
        indicators.append("❌ **Metadata:** Suspiciously low Follower-to-Following ratio.")
    
    daily_posts = tweets / max(account_age, 1)
    if daily_posts > 50:
        indicators.append("❌ **Metadata:** Abnormally high daily post volume.")
        
    # Text Checks
    spam_words = ['click', 'link', 'win', 'free', 'buy', 'اربح', 'رابط', 'مجانا', 'اشتري']
    if any(word in user_text.lower() for word in spam_words):
        indicators.append("❌ **Post Text:** High-risk promotional keywords detected (click, link, win).")
        
    if re.search(r'[\!\?\$]{2,}', user_text):
        indicators.append("❌ **Post Text:** Uses repetitive punctuation or spam symbols ($ / !).")

    # Display Result Bar
    if proba > 0.5:
        st.error(f"🚨 **HIGH RISK BOT:** {proba*100:.1f}% Risk Probability Score")
    else:
        st.success(f"✅ **HUMAN ACCOUNT:** {(1-proba)*100:.1f}% Confidence Score")
        
    # Behavioral Indicators Section
    st.subheader("Behavioral Risk Indicators")
    if indicators:
        for ind in indicators:
            st.markdown(ind)
    else:
        st.write("✅ No explicit high-risk behavioral indicators triggered.")
