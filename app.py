import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re

st.set_page_config(page_title="Spam & Bot Account Detector", page_icon="🤖", layout="wide")

@st.cache_resource
def train_and_get_model():
    # بيانات تدريب سريعة ومختصرة لضمان عمل التطبيق مباشرة
    data = {
        'text': [
            "Free follower boost click here now!", "Win cash prize instantly", "Follow back for back",
            "اشتري المتابعين واللايكات بأرخص الأسعار", "ربح سريع اضغط الرابط", "فرصة ذهبية استثمر الآن",
            "Good morning everyone", "Deep learning is fascinating", "Just shared a new blog post",
            "صباح الخير جميعاً", "مشروع الذكاء الاصطناعي مكتمل بنجاح", "دراسة جديدة عن تحليل البيانات"
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

st.title("🤖 Spam & Bot Account Detector")
st.write("أدخل النص أو تفاصيل الحساب للتحقق مما إذا كان حقيقياً أم حساماً وهمياً/سبام.")

user_text = st.text_area("أدخل نص التغريدة أو وصف الحساب (Bio):", "")

if st.button("فحص الآن"):
    if user_text.strip() == "":
        st.warning("الرجاء إدخال نص للفحص.")
    else:
        text_vector = vectorizer.transform([user_text])
        prediction = model.predict(text_vector)[0]
        proba = model.predict_proba(text_vector)[0][1]
        
        st.subheader("نتيجة الفحص:")
        if prediction == 1 or proba > 0.5:
            st.error(f"🚨 تنبيه: هذا الحساب/النص صُنف كـ **بوت / سبام** (نسبة الشك: {proba*100:.1f}%)")
        else:
            st.success(f"✅ هذا الحساب/النص يبدو **حقيقياً** (نسبة الشك: {proba*100:.1f}%)")
