
# 📩 SMS Spam Classifier

🔗 **[Live Demo](https://spam-messege-detection.onrender.com)** – Explore the Live Web App

A compact yet powerful machine learning project to **detect spam messages** using **Naive Bayes** and **Natural Language Processing (NLP)** techniques. This classifier is trained on SMS data to distinguish between **ham** (legitimate) and **spam** (unwanted) messages with a **precision of 100%**.

---

## ✨ Features & Functional Overview

### 🧠 Model
- **Algorithm**: Multinomial Naive Bayes – ideal for text classification tasks due to its probabilistic nature and efficiency on word frequency data.
- **Training Data**: Cleaned and tokenized SMS message dataset (`spam.csv`), transformed using TF-IDF Vectorization.

### 🧹 Text Preprocessing
- Lowercasing  
- Removal of punctuation and non-alphanumeric characters  
- Stopword removal  
- Stemming using `PorterStemmer`  
- Tokenization via `nltk.word_tokenize`

These steps ensure the model focuses on **meaningful textual features** and avoids noise in input data.

---

## 📊 Evaluation Metrics

| Metric         | Value         |
|----------------|---------------|
| **Accuracy**   | 97.10%        |
| **Precision**  | 100.00%       |

> ⚠️ While precision is **perfect**, indicating that while **no ham is misclassified**, a few spam messages may be missed.

---

## 📉 Confusion Matrix

|                | Predicted Ham | Predicted Spam |
|----------------|---------------|----------------|
| **Actual Ham** |      896      |       0        |
| **Actual Spam**|      30       |      108       |

- **True Positives (TP)**: 108  
- **True Negatives (TN)**: 896  
- **False Positives (FP)**: 0  
- **False Negatives (FN)**: 30  

### 🔍 Analysis Summary

- 🔒 **High Precision**: Every message labeled as spam truly is spam. This ensures users never lose real messages due to misclassification.   
- ✅ **Perfect Specificity**: No ham messages are flagged as spam (zero false positives).

---

## 🗂️ Project Structure

```
sms_spam_classifier/
├── app.py                       # Streamlit web app
├── data/
│   └── spam.csv                 # Original dataset
├── model/
│   ├── model.pkl                # Trained Naive Bayes model
│   └── vectorizer.pkl           # TF-IDF vectorizer
├── notebooks/
│   └── spam_message_detection.ipynb
├── requirements.txt             # Dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/arpan12345seth/sms_spam_classifier.git
cd sms_spam_classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Jupyter Notebook (Optional Training)
```bash
jupyter notebook notebooks/spam_message_detection.ipynb
```

### 4. Launch the App
```bash
streamlit run app.py
```

---

## 🔮 Future Enhancements

- 🧠 Incorporate other models (SVM, Logistic Regression, Random Forest) for comparison  
- 🔁 Use SMOTE or under-sampling to balance the dataset and improve recall  
- 🌐 Deploy REST API using Flask or FastAPI  
- 🌍 Add support for other languages and Unicode text  
- 📈 Integrate dashboard metrics using Plotly or Matplotlib  

---

## 👨‍💻 Author

**Arpan Seth**  
🔗 GitHub: [@arpan12345seth](https://github.com/arpan12345seth)
