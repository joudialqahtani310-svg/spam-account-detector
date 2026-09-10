import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(page_title="Bot Detector Dashboard", page_icon="🤖", layout="wide")

st.title("🤖 Multilingual Bot & Spam Account Detector")
st.write("Real-time behavioral and text-pattern detection system.")

# Load Trained Model and Vectorizer
@st.cache_resource
def load_assets():
    model = joblib.load('bot_detector_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_assets()

# Comprehensive NLP & Pattern Analyzer for Post Text
def analyze_phrase_patterns(text):
    text_lower = text.lower()
    pattern_score = 0
    text_indicators = []

    # 1. Spam & Promotional Keyword Analysis (Arabic & English)
    spam_keywords = [
        "ربح", "مجانا", "جائزة", "اضغط", "فرصة", "سريعة", "ارباح", "تابع", "متابعين", "كسب",
        "free", "click", "link", "win", "money", "claim", "bitcoin", "crypto", "earn", "instant", "booster", "gift"
    ]
    found_keywords = [word for word in spam_keywords if word in text_lower]
    if found_keywords:
        pattern_score += len(found_keywords) * 20
        text_indicators.append(f"❌ **Post Text:** High-risk promotional keywords detected ({', '.join(found_keywords)}).")

    # 2. External Link / URL Detection
    if re.search(r'http\S+|www\S+|\.com|\.net|\.org', text_lower):
        pattern_score += 30
        text_indicators.append("❌ **Post Text:** Includes suspicious external link / URL.")

    # 3. Excessive Aggressive Formatting ($ / ! / Caps)
    if len(re.findall(r'[!$?]{2,}', text)) > 0:
        pattern_score += 15
        text_indicators.append("❌ **Post Text:** Uses repetitive punctuation or spam symbols ($ / !).")

    return pattern_score, text_indicators

# UI Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Profile Metadata")
    follower_count = st.number_input("Follower Count", min_value=0, value=150)
    following_count = st.number_input("Following Count", min_value=0, value=300)
    tweet_count = st.number_input("Total Tweet Count", min_value=0, value=1500)
    account_age_days = st.number_input("Account Age (Days)", min_value=1, value=365)

with col2:
    st.subheader("2. Recent Post Content")
    tweet_text = st.text_area("Post Text (Arabic or English)", value="Hello world, this is my personal account.")

if st.button("Analyze Account"):
    # Metadata Ratios
    ratio = follower_count / (following_count + 1)
    posting_freq = tweet_count / (account_age_days + 1)
    
    # NLP Pattern Extraction
    pattern_score, text_indicators = analyze_phrase_patterns(tweet_text)
    
    # Machine Learning Processing
    cleaned_txt = re.sub(r'[^\w\s\u0600-\u06FF]', '', tweet_text.lower())
    text_feat = vectorizer.transform([cleaned_txt]).toarray()
    meta_feat = np.array([[follower_count, following_count, tweet_count, account_age_days, ratio, posting_freq, len(tweet_text)]])
    X_input = np.hstack((meta_feat, text_feat))
    
    raw_ml_prob = model.predict_proba(X_input)[0][1] * 100
    
    # Dynamic Blended Score Calculation
    final_bot_score = (raw_ml_prob * 0.4) + (min(pattern_score, 100) * 0.6)

    st.markdown("---")
    st.subheader("Analysis Output")

    # Dynamic Gauge Output
    if final_bot_score >= 60:
        st.error(f"🚨 **HIGH RISK BOT**: {final_bot_score:.1f}% Risk Probability Score")
    elif final_bot_score >= 35:
        st.warning(f"⚠️ **SUSPICIOUS ACTIVITY**: {final_bot_score:.1f}% Risk Probability Score")
    else:
        st.success(f"✅ **HUMAN ACCOUNT**: {100 - final_bot_score:.1f}% Confidence Score")

    st.subheader("Behavioral Risk Indicators")
    all_indicators = []

    # Metadata Risk Flags
    if ratio < 0.2:
        all_indicators.append("❌ **Metadata:** Suspiciously low Follower-to-Following ratio.")
    if posting_freq > 80:
        all_indicators.append("❌ **Metadata:** Abnormally high daily post volume.")
    if account_age_days < 30:
        all_indicators.append("⚠️ **Metadata:** Account created very recently.")

    # Combine with Text Indicators
    all_indicators.extend(text_indicators)

    # Display Indicator List
    if all_indicators:
        for ind in all_indicators:
            st.write(ind)
    else:
        st.write("✅ **Post Text & Metadata:** Activity matches organic human behavior.")