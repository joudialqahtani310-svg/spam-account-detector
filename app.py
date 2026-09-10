import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(page_title="Bot Detector Dashboard", page_icon="🤖", layout="wide")

st.title("🤖 Multilingual Bot & Spam Account Detector")
st.write("Real-time behavioral and text-pattern detection system.")

@st.cache_resource
def load_assets():
    model = joblib.load('bot_detector_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' urltoken ', text)
    text = re.sub(r'@\w+', ' mentiontoken ', text)
    text = re.sub(r'#\w+', ' hashtagtoken ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\u0600-\u06FF]', ' ', text)
    return text.strip().lower()

def analyze_phrase_patterns(text):
    text_lower = text.lower()
    pattern_score = 0
    text_indicators = []

    spam_keywords = [
        "ربح", "مجانا", "جائزة", "اضغط", "فرصة", "سريعة", "ارباح", "تابع", "متابعين", "كسب",
        "free", "click", "link", "win", "money", "claim", "bitcoin", "crypto", "earn", "instant", "booster", "gift"
    ]
    found_keywords = [word for word in spam_keywords if word in text_lower]
    if found_keywords:
        pattern_score += len(found_keywords) * 30
        text_indicators.append(f"❌ **Post Text:** High-risk promotional keywords detected ({', '.join(found_keywords)}).")

    if re.search(r'http\S+|www\S+|\.com|\.net|\.org', text_lower):
        pattern_score += 40
        text_indicators.append("❌ **Post Text:** Includes suspicious external link / URL.")

    if len(re.findall(r'[!$?]{2,}', text)) > 0:
        pattern_score += 25
        text_indicators.append("❌ **Post Text:** Uses repetitive punctuation or spam symbols ($ / !).")

    return pattern_score, text_indicators

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Profile Metadata")
    follower_count = st.number_input("Follower Count", min_value=0, value=300)
    following_count = st.number_input("Following Count", min_value=0, value=5057)
    tweet_count = st.number_input("Total Tweet Count", min_value=0, value=12166)
    account_age_days = st.number_input("Account Age (Days)", min_value=1, value=30)

with col2:
    st.subheader("2. Recent Post Content")
    tweet_text = st.text_area("Post Text (Arabic or English)", value="CLICK THIS LINK NOW and enjoy the prize!! don't miss it")

if st.button("Analyze Account"):
    ratio = follower_count / (following_count + 1)
    posting_freq = tweet_count / (account_age_days + 1)
    
    pattern_score, text_indicators = analyze_phrase_patterns(tweet_text)
    
    meta_score = 0
    if ratio < 0.2:
        meta_score += 35
    if posting_freq > 80:
        meta_score += 35
    if account_age_days < 30:
        meta_score += 30

    cleaned_txt = clean_text(tweet_text)
    text_feat = vectorizer.transform([cleaned_txt]).toarray()
    
    # Check ML text features activation
    ml_text_active = np.sum(text_feat) > 0
    
    meta_feat = np.array([[follower_count, following_count, tweet_count, account_age_days, ratio, posting_freq, len(cleaned_txt)]])
    X_input = np.hstack((meta_feat, text_feat))
    
    try:
        raw_ml_prob = model.predict_proba(X_input)[0][1] * 100
    except:
        raw_ml_prob = 50.0

    # Ensure rule override if metadata or text patterns strongly indicate bot
    calculated_risk = (meta_score * 0.5) + (pattern_score * 0.5)
    if ml_text_active:
        calculated_risk += 30

    final_bot_score = max(raw_ml_prob, calculated_risk)
    final_bot_score = min(final_bot_score, 98.5)

    st.markdown("---")
    st.subheader("Analysis Output")

    if final_bot_score >= 50:
        st.error(f"🚨 **HIGH RISK BOT**: {final_bot_score:.1f}% Risk Probability Score")
    elif final_bot_score >= 30:
        st.warning(f"⚠️ **SUSPICIOUS ACTIVITY**: {final_bot_score:.1f}% Risk Probability Score")
    else:
        st.success(f"✅ **HUMAN ACCOUNT**: {100 - final_bot_score:.1f}% Confidence Score")

    st.subheader("Behavioral Risk Indicators")
    all_indicators = []

    if ratio < 0.2:
        all_indicators.append("❌ **Metadata:** Suspiciously low Follower-to-Following ratio.")
    if posting_freq > 80:
        all_indicators.append("❌ **Metadata:** Abnormally high daily post volume.")
    if account_age_days < 30:
        all_indicators.append("⚠️ **Metadata:** Account created very recently.")

    all_indicators.extend(text_indicators)

    if all_indicators:
        for ind in all_indicators:
            st.write(ind)
    else:
        st.write("✅ **Post Text & Metadata:** Activity matches organic human behavior.")
