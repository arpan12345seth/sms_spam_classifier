import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK requirements
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Text preprocessing


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- PAGE CONFIG ---
st.set_page_config(page_title="Spam Classifier",
                   page_icon="✉️", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        background-color: #fff;
        border: 1px solid #ccc;
        border-radius: 0.5rem;
    }
    .title {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .result {
        font-size: 2rem;
        text-align: center;
        font-weight: 600;
    }
    .spam {
        color: #d62728;
    }
    .not-spam {
        color: #2ca02c;
    }
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- INTERFACE ---
st.markdown('<div class="title">📩 Email / SMS Spam Classifier</div>',
            unsafe_allow_html=True)

input_sms = st.text_area("✍️ Enter your message below:", height=150)

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Preprocess input
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]

        # Display Result
        if result == 1:
            st.markdown(
                '<div class="result spam">🚫 Spam Message</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result not-spam">✅ Not a Spam Message</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Created by Arpan Seth</div>',
            unsafe_allow_html=True)
