# 🚀 Query Classification using Sentence Transformers + Streamlit  

## 📝 Project Overview  
This project demonstrates a **query classification pipeline** built using the pretrained **SentenceTransformer paraphrase-MiniLM-L6-v2** model.  

We encode user queries into embeddings, classify them into categories, and provide an **interactive Streamlit app** for real-time inference.  

The deployed app allows users to:  
- 📂 Upload a CSV file of queries  
- 🤖 Run inference using the pretrained MiniLM model  
- 📊 View results in a DataFrame  
- 💾 Download predictions as CSV  

---

## 🛠️ Approach  

### 🔍 1. Model Selection  
- We used **`sentence-transformers/paraphrase-MiniLM-L6-v2`**  
- Reason:  
  - Lightweight (~80 MB) → fast inference  
  - Strong performance for semantic similarity & classification tasks  

### 🧹 2. Data Preprocessing  
- Lowercasing text  
- Removing extra spaces, punctuation  
- Tokenization (handled internally by transformer tokenizer)  

### ⚡ 3. Embedding & Classification  
- Convert queries → embeddings using MiniLM  
- Feed embeddings into a trained classifier (e.g., Logistic Regression / XGBoost)  
- Output predicted category  

### 🎛️ 4. Deployment with Streamlit  
- Created `app.py` to:  
  - Upload CSV → encode queries → predict categories  
  - Display results in an interactive table  
  - Provide download option for predictions  

---

## 🧰 Tools & Libraries  

- 🐍 Python 3.10  
- 🤗 [Sentence Transformers](https://www.sbert.net/) (`paraphrase-MiniLM-L6-v2`)  
- 📊 Pandas, NumPy  
- 🔢 Scikit-learn (for classifier & metrics)  
- ⚡ Streamlit (frontend demo platform)  

