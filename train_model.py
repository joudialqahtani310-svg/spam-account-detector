import re
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' urltoken ', text)
    text = re.sub(r'@\w+', ' mentiontoken ', text)
    text = re.sub(r'#\w+', ' hashtagtoken ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\u0600-\u06FF]', ' ', text)
    return text.strip().lower()

def generate_robust_dataset(n_samples=2000):
    np.random.seed(42)
    half = n_samples // 2

    # Human Data Patterns
    h_followers = np.random.randint(100, 10000, half)
    h_following = np.random.randint(50, 1500, half)
    h_tweets = np.random.randint(100, 5000, half)
    h_age = np.random.randint(100, 2000, half)
    
    human_phrases = [
        "صباح الخير جميعاً أتمنى لكم يوماً سعيداً",
        "Just finished working on my project, feeling great!",
        "ما هو رأيكم في التحديثات الجديدة للذكاء الاصطناعي؟",
        "Had a wonderful weekend with my family and friends.",
        "الطقس اليوم جميل جداً في المدينة",
        "Learning Python and machine learning is really interesting.",
        "شكراً لك على المساعدة القيمة أتمنى لك التوفيق",
        "Great article! Thanks for sharing these valuable insights."
    ]
    h_texts = np.random.choice(human_phrases, half)

    # Bot Data Patterns (Contains spam tokens, links, calls to action)
    b_followers = np.random.randint(0, 50, half)
    b_following = np.random.randint(1500, 8000, half)
    b_tweets = np.random.randint(10000, 80000, half)
    b_age = np.random.randint(1, 60, half)
    
    bot_phrases = [
        "عاجل ربح مال مجاني اضغط على الرابط الان http://spamlink.com",
        "CLICK HERE TO WIN FREE BITCOIN NOW! GO TO http://crypto.com",
        "احصل على متابعين مجاناً بسرعة وبدون مجهود اضغط هنا",
        "CLAIM YOUR FREE $500 GIFT CARD INSTANTLY CLICK LINK",
        "فرصة خيالية لاستثمار الأموال أرباح مضمونة 100%",
        "FAST FOLLOWERS BOOSTER GET 10K FOLLOWERS NOW FREE",
        "اشترك الان للحصول على الجائزة الكبرى مجانا",
        "MAKE MONEY FROM HOME FAST EASY CLICK HERE NOW"
    ]
    b_texts = np.random.choice(bot_phrases, half)

    df_h = pd.DataFrame({
        'follower_count': h_followers, 'following_count': h_following,
        'tweet_count': h_tweets, 'account_age_days': h_age,
        'tweet_text': h_texts, 'label': 0
    })

    df_b = pd.DataFrame({
        'follower_count': b_followers, 'following_count': b_following,
        'tweet_count': b_tweets, 'account_age_days': b_age,
        'tweet_text': b_texts, 'label': 1
    })

    return pd.concat([df_h, df_b], ignore_index=True)

if __name__ == "__main__":
    print("Training robust model...")
    df = generate_robust_dataset(2000)

    df['ratio'] = df['follower_count'] / (df['following_count'] + 1)
    df['freq'] = df['tweet_count'] / (df['account_age_days'] + 1)
    df['clean_txt'] = df['tweet_text'].apply(clean_text)
    df['txt_len'] = df['clean_txt'].apply(len)

    # Word & Phrase Vectorizer (Unigrams + Bigrams)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=300)
    text_feat = vectorizer.fit_transform(df['clean_txt']).toarray()

    meta_feat = df[['follower_count', 'following_count', 'tweet_count', 
                    'account_age_days', 'ratio', 'freq', 'txt_len']].values

    X = np.hstack((meta_feat, text_feat))
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    print(f"Model Trained Successfully! Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

    joblib.dump(model, 'bot_detector_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Saved bot_detector_model.pkl and tfidf_vectorizer.pkl successfully.")