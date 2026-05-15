import joblib

vec = joblib.load("models/word_vectorizer.pkl")

print(vec.get_feature_names_out()[:20])