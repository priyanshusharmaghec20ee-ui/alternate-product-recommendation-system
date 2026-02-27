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

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load data and models
@st.cache_resource
def load_data_and_models():
    df_master_loaded = pd.read_pickle("df_master.pkl")
    faiss_index_loaded = faiss.read_index("faiss_index.bin")
    with open("word2vec_model.pkl", "rb") as f:
        word2vec_model_loaded = pickle.load(f)
    # indices_loaded = pd.read_pickle("indices.pkl") # Not directly used in recommend_faiss
    return df_master_loaded, faiss_index_loaded, word2vec_model_loaded

df_master, index, word2vec_model = load_data_and_models()

# Initialize NLTK components for preprocessing
s_w = set(stopwords.words('english'))
lem = WordNetLemmatizer()

# Define preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    words = text.split()
    words = [i for i in words if i not in s_w]
    words = [lem.lemmatize(i) for i in words]
    return \" \".join(words)

# Define function to get product embedding for a query
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

# Define recommendation function
def recommend_faiss(query, n=10):
    tokenized_query = preprocess_text(query).split()
    query_embedding = get_product_embedding(tokenized_query)
    query_embedding_faiss = query_embedding.reshape(1, -1).astype('float32')

    distances, indices = index.search(query_embedding_faiss, n)
    similar_indices = indices.flatten()
    recommended_products = df_master.iloc[similar_indices].copy()

    similarity_scores = 1 / (1 + distances.flatten())
    recommended_products['similarity score'] = similarity_scores

    return recommended_products[['title', 'brand', 'price', 'rating', 'platform', 'image_url', 'product_url', 'similarity score']]

# Streamlit app layout
st.title('Product Recommendation System')

user_query = st.text_input('Enter your query (e.g., "bluetooth headphones"):')

if st.button('Get Recommendations'):
    if user_query:
        st.write(f"Showing recommendations for: **{user_query}**")
        recommendations_df = recommend_faiss(user_query, n=10)

        # Function to make URLs clickable
        def make_clickable(url):
            if pd.isna(url) or url == '':
                return ''
            return f'<a href="{url}" target="_blank">Link</a>'

        # Display recommendations with clickable URLs and images
        for i, row in recommendations_df.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                if row['image_url']:
                    st.markdown(f"<img src='{row['image_url']}' width='100'>", unsafe_allow_html=True)
                else:
                    st.write("No Image")
            with col2:
                st.markdown(f"### {row['title']}")
                st.write(f"**Brand:** {row['brand']}")
                st.write(f"**Price:** ₹{row['price']:.2f}")
                st.write(f"**Rating:** {row['rating']:.1f}")
                st.write(f"**Platform:** {row['platform']}")
                st.write(f"**Similarity Score:** {row['similarity score']:.4f}")
                st.markdown(f"**Product URL:** {make_clickable(row['product_url'])}", unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.warning('Please enter a query to get recommendations.')
