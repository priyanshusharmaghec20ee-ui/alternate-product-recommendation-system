import streamlit as st
import pandas as pd
import numpy as np
import faiss
from gensim.models import Word2Vec
import nltk
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Alternate Product Recommendation System using AI",
    page_icon="🛍️",
    layout="wide"
)

# ---------------- DOWNLOAD NLTK ----------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data_and_models():
    df_master_loaded = pd.read_pickle("df_master.pkl")
    faiss_index_loaded = faiss.read_index("faiss_index.bin")
    with open("word2vec_model.pkl", "rb") as f:
        word2vec_model_loaded = pickle.load(f)
    return df_master_loaded, faiss_index_loaded, word2vec_model_loaded

df_master, index, word2vec_model = load_data_and_models()

# ---------------- PREPROCESSING ----------------
s_w = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [i for i in words if i not in s_w]
    words = [lem.lemmatize(i) for i in words]
    return words

def get_product_embedding(words):
    vectors = [
        word2vec_model.wv[word]
        for word in words
        if word in word2vec_model.wv.key_to_index
    ]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def recommend_faiss(query, n=100):
    tokenized_query = preprocess_text(query)
    query_embedding = get_product_embedding(tokenized_query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')

    distances, indices = index.search(query_embedding, n)
    similar_indices = indices.flatten()
    recommended_products = df_master.iloc[similar_indices].copy()

    recommended_products["similarity_score"] = 1 / (1 + distances.flatten())

    return recommended_products.to_dict(orient="records")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = 1
if "results" not in st.session_state:
    st.session_state.results = []

# ---------------- UI DESIGN ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.stApp {
    background: transparent;
}
.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #00e0ff;
    margin-bottom: 25px;
    text-shadow: 0 0 20px rgba(0,224,255,0.7);
}
.card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(15px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    height: 100%;
    transition: 0.4s ease;
    box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    color: white;
}
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 15px 40px rgba(0,224,255,0.6);
}
.price {
    color: #00ff9d;
    font-weight: bold;
    font-size: 18px;
}
.platform-logo {
    height: 24px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">Alternate Product Recommendation System using AI</div>',
    unsafe_allow_html=True
)

# ---------------- SEARCH ----------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    query = st.text_input(
        "Search",
        placeholder="Search for a product...",
        label_visibility="collapsed"
    )
    search_clicked = st.button("🔍 Search")

# ---------------- FILTERS ----------------
st.markdown("### 🔎 Filters")
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    sort_option = st.selectbox(
        "Sort By",
        ["Relevance", "Price Low to High", "Price High to Low", "Rating High to Low"]
    )
    platform_filter = st.selectbox(
        "Platform",
        ["All", "Amazon", "Flipkart"]
    )

with filter_col2:
    min_price = st.number_input("Minimum Price", min_value=0, value=0)
    max_price = st.number_input("Maximum Price", min_value=0, value=100000)

st.markdown("---")

# ---------------- SEARCH ACTION ----------------
if search_clicked:
    st.session_state.page = 1
    if query.strip() == "":
        st.warning("Please enter something to search.")
    else:
        with st.spinner("Finding best matches..."):
            st.session_state.results = recommend_faiss(query)

results = st.session_state.results

# ---------------- DISPLAY RESULTS ----------------
if results:
    filtered = []

    for r in results:
        price = float(r.get("price", 0))
        if min_price <= price <= max_price:
            if platform_filter == "All" or r.get("platform") == platform_filter:
                filtered.append(r)

    if sort_option == "Price Low to High":
        filtered = sorted(filtered, key=lambda x: float(x.get("price", 0)))
    elif sort_option == "Price High to Low":
        filtered = sorted(filtered, key=lambda x: float(x.get("price", 0)), reverse=True)
    elif sort_option == "Rating High to Low":
        filtered = sorted(filtered, key=lambda x: float(x.get("rating", 0)), reverse=True)

    items_per_page = 16
    total_pages = max(1, (len(filtered) - 1) // items_per_page + 1)

    start = (st.session_state.page - 1) * items_per_page
    end = start + items_per_page
    page_items = filtered[start:end]

    st.markdown("## 🔥 Recommended Products")

    columns_count = 4
    for row in range(0, len(page_items), columns_count):
        cols = st.columns(columns_count)
        for i in range(columns_count):
            if row + i < len(page_items):
                product = page_items[row + i]
                with cols[i]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)

                    if product.get("image_url"):
                        st.image(product["image_url"])

                    st.markdown(
                        f"<h4>{product.get('title','No Title')}</h4>",
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f"<div class='price'>₹ {product.get('price','N/A')}</div>",
                        unsafe_allow_html=True
                    )

                    if product.get("product_url"):
                        st.markdown(f"[🛒 View Product]({product['product_url']})")

                    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- PAGINATION ----------------
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        if st.button("⬅ Previous") and st.session_state.page > 1:
            st.session_state.page -= 1

    with col3:
        if st.button("Next ➡") and st.session_state.page < total_pages:
            st.session_state.page += 1

    st.markdown(
        f"<center style='color:white;'>Page {st.session_state.page} of {total_pages}</center>",
        unsafe_allow_html=True
    )
