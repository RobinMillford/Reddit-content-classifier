import streamlit as st
from joblib import load
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

# --- Page Configuration ---
st.set_page_config(
    page_title="Reddit Content Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #0d0c22;
        background-image: 
            radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0);
        background-size: 25px 25px;
    }
    .st-emotion-cache-10trblm { color: #e5e7eb; }
    .stButton>button {
        background: linear-gradient(90deg, #d53a9d 0%, #7c3aed 100%);
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(192, 38, 211, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(192, 38, 211, 0.7);
    }
    .stTextArea textarea {
        background-color: rgba(17, 24, 39, 0.8);
        border: 1px solid #4b5563;
        color: #e5e7eb;
    }
    .stAlert {
        background: rgba(23, 22, 49, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e5e7eb !important;
    }
    .stAlert[data-baseweb="notification"][data-kind="success"] {
        border-left: 6px solid #22d3ee !important;
        box-shadow: 0 0 30px rgba(6, 182, 212, 0.3);
    }
    .stAlert[data-baseweb="notification"][data-kind="error"] {
        border-left: 6px solid #d946ef !important;
        box-shadow: 0 0 30px rgba(217, 70, 239, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- Caching the Model Loading ---
@st.cache_resource
def load_model_and_vectorizer():
    """Loads the saved champion model and vectorizer from the repository."""
    try:
        model = load('champion_model.pkl')
        vectorizer = load('vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer not found. This can happen during the first deployment. If the error persists, please check the repository files.")
        return None, None

model, vectorizer = load_model_and_vectorizer()

# --- UI Layout ---
st.markdown(
    '<h1 style="text-align: center; font-size: 3.5rem; font-weight: bold; background-image: linear-gradient(90deg, #d53a9d, #22d3ee); -webkit-background-clip: text; color: transparent;">Content Classifier</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align: center; font-size: 1.25rem; color: #9ca3af;">ML-powered analysis of SFW vs. NSFW text.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

col1, col2 = st.columns([2, 1.2], gap="large")

with col1:
    input_text = st.text_area("Enter post title and body here:", height=250, placeholder="e.g., 'This is a title. This is the body of the post...'")
    
    if st.button("Classify Content", use_container_width=True):
        if model and vectorizer and input_text:
            with st.spinner("Analyzing..."):
                # First, vectorize the raw text
                vectorized_text = vectorizer.transform([input_text])
                
                # *** THIS IS THE KEY FIX ***
                # Check if the loaded model is an MLPClassifier or an ensemble that might contain one.
                # If so, convert the data to the dense format it requires.
                is_mlp_or_ensemble = isinstance(model, (MLPClassifier, VotingClassifier))
                
                if is_mlp_or_ensemble:
                    prediction_input = vectorized_text.toarray()
                else:
                    prediction_input = vectorized_text

                prediction = model.predict(prediction_input)
                
                if prediction[0] == 1: # NSFW
                    st.error("**Result: NSFW Content**", icon="🚨")
                    st.markdown("This text has been classified as **Not-Safe-For-Work**.")
                else: # SFW
                    st.success("**Result: SFW Content**", icon="✅")
                    st.markdown("This text has been classified as **Safe-For-Work**.")
        elif not input_text:
            st.warning("Please enter some text to classify.")

with col2:
    st.info("**About this Project**")
    st.markdown("""
    This application is a complete end-to-end MLOps project that demonstrates a full CI/CD pipeline.
    
    - **Automated Training:** A weekly GitHub Action automatically fetches the latest data from Reddit and retrains multiple classification models.
    - **Model Selection:** The best-performing model (including an ensemble) is automatically selected and saved.
    - **CI/CD:** Pushing the new model to the `main` branch of the GitHub repository automatically triggers a re-deployment of this Streamlit app.
    """)
