import streamlit as st
from joblib import load
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Reddit Content Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for the "Cyber-Glow" Theme ---
st.markdown("""
<style>
    /* Main background and font */
    .stApp {
        background-color: #0d0c22;
        background-image: 
            radial-gradient(circle at 1px 1px, rgba(255,255,255,0.05) 1px, transparent 0);
        background-size: 25px 25px;
        color: #e5e7eb;
    }

    /* --- Header --- */
    /* This targets the main title element */
    .st-emotion-cache-10trblm {
        padding-bottom: 1rem;
    }

    /* --- Main Card Elements --- */
    /* Style for the text area */
    .stTextArea textarea {
        background-color: rgba(23, 22, 49, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 0.75rem;
        color: #e5e7eb;
        transition: box-shadow 0.3s ease;
        box-shadow: 0 0 20px rgba(192, 38, 211, 0.1);
    }
    .stTextArea textarea:focus {
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.5);
    }

    /* Style for the button */
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

    /* --- Result Cards --- */
    /* Base style for both success and error boxes */
    .stAlert {
        background: rgba(23, 22, 49, 0.6);
        backdrop-filter: blur(12px);
        border-radius: 1.5rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e5e7eb !important;
    }
    
    /* SFW Result Box (Success) */
    .stAlert[data-baseweb="notification"][data-kind="success"] {
        border-left: 6px solid #22d3ee !important; /* cyan-400 */
        box-shadow: 0 0 30px rgba(6, 182, 212, 0.3);
    }
    
    /* NSFW Result Box (Error) */
    .stAlert[data-baseweb="notification"][data-kind="error"] {
        border-left: 6px solid #d946ef !important; /* fuchsia-500 */
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
                input_df = pd.DataFrame([input_text])
                prediction = model.predict(input_df)
                
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
    - **Model Selection:** The best-performing model is automatically selected based on its F1-score and saved as the new production model.
    - **CI/CD:** Pushing the new model to the `main` branch of the GitHub repository automatically triggers a re-deployment of this Streamlit application.
    """)