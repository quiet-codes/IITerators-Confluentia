
import hashlib
import io

import numpy as np
import pandas as pd
import streamlit as st
# from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Semantic Query Classifier",
    page_icon="🔎",
    layout="wide",
)

MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.20


# --------------------------------------------------
# 2. Cached model: load only when predictions are run
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)
# Cache embeddings for repeated text inputs.
# The model name is included in the cache key.
@st.cache_data(show_spinner=False, max_entries=8)
def encode_texts(texts, model_name):
    model = get_model()
    return model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )


def calculate_scores(queries, categories):
    query_embeddings = encode_texts(tuple(queries), MODEL_NAME)
    category_embeddings = encode_texts(tuple(categories), MODEL_NAME)

    # Embeddings are normalized, so dot product = cosine similarity.
    return np.sum(query_embeddings * category_embeddings, axis=1)


# --------------------------------------------------
# 3. Interface
# --------------------------------------------------
st.title("🔎 Semantic Query Classifier")
st.markdown(
    "Compare search queries with category paths using "
    "MiniLM embeddings and cosine similarity."
)

st.info(
    "Upload a CSV with `origin_query` and `category_path`. "
    "The app will preserve the input columns and add `score` "
    "and `result`."
)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload your CSV",
        type=["csv"],
        help="Your CSV must contain origin_query and category_path.",
    )

with col2:
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
    )

st.caption(
    f"Decision rule: score ≥ {threshold:.2f} → result 1; "
    f"score < {threshold:.2f} → result 0."
)

# --------------------------------------------------
# 4. Read and validate input
# --------------------------------------------------
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    result_key = (file_hash, threshold)

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        st.stop()

    required_columns = ["origin_query", "category_path"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        st.error(
            "Missing required column(s): "
            + ", ".join(missing_columns)
        )
        st.stop()

    if df.empty:
        st.warning("The CSV has no data rows.")
        st.stop()

    # Preserve all original columns.
    df = df.copy()

    # Convert required text columns safely.
    for column in required_columns:
        df[column] = df[column].astype("string").str.strip()

    # Keep the original category cleanup behavior.
    df["category_path"] = df["category_path"].str.replace(
        ",", " ", regex=False
    )

    valid_rows = (
        df["origin_query"].notna()
        & df["category_path"].notna()
        & df["origin_query"].ne("")
        & df["category_path"].ne("")
    ).fillna(False).to_numpy(dtype=bool)

    st.subheader("Input preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    if not valid_rows.all():
        st.warning(
            f"{int((~valid_rows).sum())} row(s) have missing or "
            "empty query/category text. These rows will have "
            "a blank score and result 0."
        )

    run_clicked = st.button(
        "🚀 Run predictions",
        type="primary",
        disabled=not valid_rows.any(),
        use_container_width=True,
    )

    if run_clicked:
        try:
            queries = df.loc[valid_rows, "origin_query"].tolist()
            categories = df.loc[valid_rows, "category_path"].tolist()

            with st.spinner(
                "Loading the AI model if needed and calculating scores..."
            ):
                scores = calculate_scores(queries, categories)

            result_df = df.copy()
            result_df["score"] = np.nan
            result_df["result"] = 0

            result_df.loc[valid_rows, "score"] = scores
            result_df.loc[valid_rows, "result"] = (
                scores >= threshold
            ).astype(int)

            # Store results so they remain visible after Streamlit reruns.
            st.session_state["prediction_df"] = result_df
            st.session_state["prediction_key"] = result_key

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    # Show results only for the current file and threshold.
    if (
        st.session_state.get("prediction_key") == result_key
        and "prediction_df" in st.session_state
    ):
        result_df = st.session_state["prediction_df"]

        st.divider()
        st.subheader("📊 Prediction summary")

        total_rows = len(result_df)
        relevant_count = int(result_df["result"].sum())
        not_relevant_count = total_rows - relevant_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Total rows", total_rows)
        m2.metric("Relevant (1)", relevant_count)
        m3.metric("Not relevant (0)", not_relevant_count)

        st.subheader("Complete prediction results")
        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

        # Download the COMPLETE dataframe, not just the result column.
        csv_output = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download complete results CSV",
            data=csv_output,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.caption(
            "The CSV includes the original input columns, score, "
            "and result. A result of 1 indicates that the score "
            "meets the selected threshold; it does not guarantee "
            "a correct classification."
        )

else:
    st.subheader("How to use")
    st.markdown(
        """
        1. Upload your CSV file.
        2. Choose the similarity threshold.
        3. Click **Run predictions**.
        4. Inspect the complete table and download the results.
        """
    )
