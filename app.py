%%writefile streamlit_app.py
import streamlit as st
import requests # Import the requests library

# Define the FastAPI endpoint URL
FASTAPI_URL = "http://localhost:8000/recommend"

# Streamlit app layout
st.title('Product Recommendation System (Client)')

user_query = st.text_input('Enter your query (e.g., "bluetooth headphones"):')

if st.button('Get Recommendations'):
    if user_query:
        st.write(f"Fetching recommendations for: **{user_query}**")

        try:
            # Make a POST request to the FastAPI endpoint
            response = requests.post(
                FASTAPI_URL,
                json={"query": user_query, "n": 10} # Send query and n as JSON payload
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()

            if data and "recommendations" in data and len(data["recommendations"]) > 0:
                recommendations_list = data["recommendations"]

                # Display recommendations
                for i, row in enumerate(recommendations_list):
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if row.get('image_url'):
                            st.markdown(f"<img src='{row['image_url']}' width='100'>", unsafe_allow_html=True)
                        else:
                            st.write("No Image")
                    with col2:
                        st.markdown(f"### {row.get('title', 'N/A')}")
                        st.write(f"**Brand:** {row.get('brand', 'N/A')}")
                        st.write(f"**Price:** ₹{row.get('price', 0.0):.2f}")
                        st.write(f"**Rating:** {row.get('rating', 0.0):.1f}")
                        st.write(f"**Platform:** {row.get('platform', 'N/A')}")
                        st.write(f"**Similarity Score:** {row.get('similarity_score', 0.0):.4f}")
                        product_url = row.get('product_url', '')
                        if product_url:
                            st.markdown(f"**Product URL:** <a href=\"{product_url}\" target=\"_blank\">Link</a>", unsafe_allow_html=True)
                        else:
                            st.write("**Product URL:** N/A")
                        st.markdown("---")
            else:
                st.warning(f"No recommendations found for '{user_query}'.")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to FastAPI backend at {FASTAPI_URL}. Please ensure the server is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching recommendations: {e}")
    else:
        st.warning('Please enter a query to get recommendations.')
