import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

st.set_page_config(page_title="Multilingual Bot & Spam Account Detector", page_icon="🤖", layout="wide")

# إعادة بناء وتشغيل النموذج الأصلي بنفس المتغيرات
@st.cache_resource
def load_original_model():
    # بيانات تدريب نموذجية تحتوي على النص وخصائص الحساب (Followers, Following, Tweets count)
    data = {
        'text': [
            "Get free followers now click here", "Win instant cash prize", "Follow back immediately",
            "اشتري المتابعين واللايكات الآن", "ربح سريع اضغط الرابط", "فرصة استثمارية مضمونة",
            "Working on my machine learning project", "Great research paper published today", "Enjoying the weekend with family",
            "مشروع الذكاء الاصطناعي اليوم ممتاز", "بحث علمي جديد في معالجة اللغة", "مساء الخير جميعاً"
        ],
        'followers_count': [10, 5, 2, 12, 8, 3, 450, 1200, 380, 890, 1500, 420],
        'following_count': [3000, 4500, 2900, 4900, 5000, 3200, 310, 400, 290, 510, 600, 380],
        'tweet_count': [500, 800, 1200, 600, 950, 400, 150, 320, 210, 430, 890, 180],
        'is_bot': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    vectorizer = TfidfVectorizer(max_features=100)
    text_features = vectorizer.fit_transform(df['text']).toarray()
    
    # دمج الخصائص النصية مع خصائص الحساب (Metadata)
    meta_features = df[['followers_count', 'following_count', 'tweet_count']].values
    X = np.hstack((text_features, meta_features))
    y = df['is_bot']
    
    # تدريب نموذج XGBoost الاصلي
    model = XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X, y)
    
    return model, vectorizer

model, vectorizer = load_original_model()

# واجهة المستخدم الأصلية
st.title("🤖 Multilingual Bot & Spam Account Detector")
st.write("Detect automated bot profiles and spam messages using XGBoost and NLP analysis.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Text Content / Bio")
    user_text = st.text_area("Enter Tweet text or Account Bio:", "Get free followers now click here")

with col2:
    st.subheader("2. Account Metadata")
    followers = st.number_input("Followers Count:", min_value=0, value=10)
    following = st.number_input("Following Count:", min_value=0, value=3000)
    tweets = st.number_input("Total Tweets Count:", min_value=0, value=500)

if st.button("Run Bot & Spam Detection", type="primary"):
    text_vec = vectorizer.transform([user_text]).toarray()
    meta_vec = np.array([[followers, following, tweets]])
    input_features = np.hstack((text_vec, meta_vec))
    
    prediction = model.predict(input_features)[0]
    proba = model.predict_proba(input_features)[0][1]
    
    st.markdown("---")
    st.subheader("Prediction Results:")
    
    if prediction == 1 or proba > 0.5:
        st.error(f"🚨 **Bot / Spam Account Detected!** (Risk Score: {proba*100:.1f}%)")
    else:
        st.success(f"✅ **Authentic / Human Account** (Risk Score: {proba*100:.1f}%)")
