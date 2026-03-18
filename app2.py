import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

st.title("User Query Mapping")

# Path to your locally saved model folder
LOCAL_MODEL_PATH = "paraphrase" # Change this to your folder name if different

@st.cache_resource(show_spinner=True)
def get_model():
    # Load the model from the local directory
    return SentenceTransformer(LOCAL_MODEL_PATH)

model = get_model()

def calculate_product_score(product_name, tag_sentence, model=model):
    if not isinstance(product_name, str) or not isinstance(tag_sentence, str) or not tag_sentence.strip():
        return np.nan
    product_embedding = model.encode(product_name, convert_to_tensor=True)
    tag_embedding = model.encode(tag_sentence, convert_to_tensor=True)
    similarity = util.cos_sim(product_embedding, tag_embedding).item()
    return similarity

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Sample of uploaded data:", df.head())

    # Drop NA values
    df = df.dropna()

    # Clean category_path (remove commas if present)
    if 'category_path' in df.columns:
        df['category_path'] = df['category_path'].str.replace(",", " ", regex=False)
    else:
        st.error("Input CSV must contain 'category_path' column.")
        st.stop()
    if 'origin_query' not in df.columns:
        st.error("Input CSV must contain 'origin_query' column.")
        st.stop()

    # Compute scores
    with st.spinner("Calculating similarity scores..."):
        df['score'] = [
            calculate_product_score(name, tags)
            for name, tags in zip(df['origin_query'], df['category_path'])
        ]

    # Predict result column (user adjustable threshold)
    threshold = 0.22
    df['result'] = [0 if scr < threshold else 1 for scr in df['score']]

    st.write("Preview of prediction results:", df[['origin_query', 'category_path', 'score', 'result']].head())

    # Download result file
    out_csv = df['result'].to_csv(index=False)
    st.download_button(
        "Download predictions as CSV",
        out_csv,
        file_name="predictions.csv",
        mime="text/csv"
    )