from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from gensim.models import Word2Vec
import nltk
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from starlette.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables to hold loaded models and dataframes
df_master = None
faiss_index = None
word2vec_model = None
s_w = None
lem = None

# Function to load data and models on startup
@app.on_event("startup")
async def load_data_and_models():
    global df_master, faiss_index, word2vec_model, s_w, lem
    
    # Download NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    df_master = pd.read_pickle("df_master.pkl")
    faiss_index = faiss.read_index("faiss_index.bin")
    with open("word2vec_model.pkl", "rb") as f:
        word2vec_model = pickle.load(f)
    
    # Initialize NLTK components for preprocessing
    s_w = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    
    print("Models and data loaded successfully.")

# Define preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [i for i in words if i not in s_w]
    words = [lem.lemmatize(i) for i in words]
    return " ".join(words)

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
def recommend_faiss(query: str, n: int = 10):
    if not word2vec_model or not faiss_index or not df_master:
        # This case should ideally not be reached if startup event works correctly
        return pd.DataFrame() 

    tokenized_query = preprocess_text(query).split()
    query_embedding = get_product_embedding(tokenized_query)
    query_embedding_faiss = query_embedding.reshape(1, -1).astype('float32')

    distances, indices = faiss_index.search(query_embedding_faiss, n)
    similar_indices = indices.flatten()
    recommended_products = df_master.iloc[similar_indices].copy()

    similarity_scores = 1 / (1 + distances.flatten())
    recommended_products['similarity_score'] = similarity_scores

    return recommended_products[['title', 'brand', 'price', 'rating', 'platform', 'image_url', 'product_url', 'similarity_score']]

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str
    n: int = 10

# Define POST endpoint for recommendations
@app.post("/recommend")
async def get_recommendations(request: QueryRequest):
    recommendations_df = recommend_faiss(request.query, request.n)
    if recommendations_df.empty:
        return {"message": "No recommendations found.", "query": request.query}
    
    # Convert DataFrame to list of dictionaries for JSON response
    recommendations_list = recommendations_df.to_dict(orient='records')
    return {"query": request.query, "recommendations": recommendations_list}
