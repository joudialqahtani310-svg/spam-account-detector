import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

st.set_page_config(page_title="Multilingual Bot & Spam Account Detector", page_icon="🤖", layout="wide")

@st.cache_resource
def load_assets():
    # التحقق مما إذا كانت ملفات النموذج المدرب موجودة مسبقاً
    if os.path.exists('bot_detector_model.pkl') and os.path.exists('vectorizer.pkl'):
        model = joblib.load('bot_detector_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
    else:
        # إذا لم تجد الملفات، يتم تجهيزها فوراً بالخلفية لضمان عدم توقف السيرفر
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

# استدعاء الدالة بنفس طريقة كودك الأصلي تماماً
model, vectorizer = load_assets()
