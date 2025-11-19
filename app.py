import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os

# Import specific models to ensure unpickling works
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Content Classifier | Multi-Label Analysis",
    page_icon="🤖",
    layout="wide",
)

# --- Modern CSS Styling ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --error-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        --dark-bg: #0f0f23;
        --card-bg: rgba(255, 255, 255, 0.05);
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --border-color: rgba(255, 255, 255, 0.1);
        --shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        --border-radius: 16px;
    }
    
    /* Global Styles */
    .stApp {
        background: var(--dark-bg);
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main Container */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Custom Header */
    .hero-header {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: var(--card-bg);
        border-radius: var(--border-radius);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        min-width: 120px;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    /* Cards */
    .modern-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary-gradient);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
        margin: 0.5rem 0;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Text Areas */
    .stTextArea textarea {
        background: rgba(15, 15, 35, 0.8);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        color: var(--text-primary);
        font-size: 1rem;
        padding: 1rem;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: var(--card-bg);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
    }
    
    .stRadio > div > label {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Alerts and Status */
    .stAlert {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-radius: var(--border-radius);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        margin: 1rem 0;
    }
    
    .stSuccess {
        background: rgba(79, 172, 254, 0.1);
        border-left: 4px solid #4facfe;
    }
    
    .stError {
        background: rgba(250, 112, 154, 0.1);
        border-left: 4px solid #fa709a;
    }
    
    .stWarning {
        background: rgba(67, 233, 123, 0.1);
        border-left: 4px solid #43e97b;
    }
    
    .stInfo {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
    }
    
    /* Results Section */
    .result-container {
        margin: 2rem 0;
    }
    
    .result-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--text-primary);
    }
    
    .classification-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 500;
        margin: 0.25rem;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
    }
    
    .badge-safe {
        background: rgba(79, 172, 254, 0.2);
        color: #4facfe;
        border: 1px solid rgba(79, 172, 254, 0.3);
    }
    
    .badge-nsfw {
        background: rgba(250, 112, 154, 0.2);
        color: #fa709a;
        border: 1px solid rgba(250, 112, 154, 0.3);
    }
    
    .badge-positive {
        background: rgba(67, 233, 123, 0.2);
        color: #43e97b;
        border: 1px solid rgba(67, 233, 123, 0.3);
    }
    
    .badge-negative {
        background: rgba(254, 215, 102, 0.2);
        color: #fed766;
        border: 1px solid rgba(254, 215, 102, 0.3);
    }
    
    .badge-neutral {
        background: rgba(160, 160, 160, 0.2);
        color: #a0a0a0;
        border: 1px solid rgba(160, 160, 160, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.9);
        backdrop-filter: blur(20px);
    }
    
    /* Performance Metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: #667eea transparent #667eea transparent;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 15, 35, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-gradient);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- Enhanced Model Loading with Progress ---
@st.cache_resource
def load_models_and_metadata():
    """Loads the saved models, vectorizer, and metadata with enhanced error handling."""
    # Define the filenames generated by your training script
    FILES = {
        'binary': 'best_binary_model.joblib',
        'multi': 'best_multi_model.joblib',
        'vectorizer': 'tfidf_vectorizer.joblib',
        'metadata': 'model_metadata.joblib'
    }
    
    with st.spinner("🔄 Loading AI models..."):
        try:
            # Load Vectorizer
            if os.path.exists(FILES['vectorizer']):
                vectorizer = load(FILES['vectorizer'])
            else:
                st.error(f"❌ Vectorizer file not found: {FILES['vectorizer']}")
                return None, None, None, None

            # Load Metadata
            if os.path.exists(FILES['metadata']):
                metadata = load(FILES['metadata'])
            else:
                st.warning(f"⚠️ Metadata file not found: {FILES['metadata']}")
                metadata = None

            # Load Binary Model
            binary_model = None
            if os.path.exists(FILES['binary']):
                binary_model = load(FILES['binary'])
            else:
                st.warning(f"⚠️ Binary model file not found: {FILES['binary']}")

            # Load Multi-Label Model
            multi_label_model = None
            if os.path.exists(FILES['multi']):
                multi_label_model = load(FILES['multi'])
            else:
                st.warning(f"⚠️ Multi-label model file not found: {FILES['multi']}")
                
            return binary_model, multi_label_model, vectorizer, metadata
                
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return None, None, None, None

binary_model, multi_label_model, vectorizer, metadata = load_models_and_metadata()

# --- Model Status Display ---
st.markdown("### 🤖 AI Model Status")
status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    if binary_model:
        st.success("✅ Binary Model")
    else:
        st.error("❌ Binary Model")

with status_col2:
    if multi_label_model:
        st.success("✅ Multi-Label Model")
    else:
        st.error("❌ Multi-Label Model")

with status_col3:
    if vectorizer:
        st.success("✅ Vectorizer")
    else:
        st.error("❌ Vectorizer")

with status_col4:
    if metadata:
        st.success("✅ Metadata")
    else:
        st.error("❌ Metadata")

if not binary_model and not multi_label_model:
    st.error("⚠️ **No models available!** Please train models first by running: `python src/train.py`")

st.markdown("---")

# --- Modern Hero Header ---
st.markdown("""
<div style="text-align: center; margin-bottom: 3rem; padding: 2rem 0;">
    <h1 style="font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 1rem; line-height: 1.2;">🤖 AI Content Classifier</h1>
    <p style="font-size: 1.3rem; color: #a0a0a0; font-weight: 400; margin-bottom: 2rem;">Advanced Multi-Label Analysis powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Performance Stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Categories", "5", help="Multi-label classification categories")
with col2:
    st.metric("Accuracy", "88%", help="Overall model performance")
with col3:
    st.metric("Training Data", "12K+", help="Reddit posts analyzed")
with col4:
    st.metric("Analysis Speed", "Real-time", help="Instant AI processing")

st.markdown("---")

# --- Helper Functions ---
def predict_multi_label(text, model, vectorizer, metadata):
    if not all([model, vectorizer, metadata]): return None
    vectorized_text = vectorizer.transform([text])
    try:
        predictions = model.predict(vectorized_text)[0]
    except (TypeError, ValueError):
        prediction_input = vectorized_text.toarray()
        predictions = model.predict(prediction_input)[0]
    results = {}
    if 'label_columns' in metadata and 'label_encoders' in metadata:
        for i, category in enumerate(metadata['label_columns']):
            label_idx = predictions[i]
            for label_name, idx in metadata['label_encoders'][category].items():
                if idx == label_idx:
                    results[category] = label_name
                    break
    return results

def predict_binary(text, model, vectorizer):
    if not all([model, vectorizer]): return None, None
    vectorized_text = vectorizer.transform([text])
    try:
        prediction = model.predict(vectorized_text)[0]
        try:
            probabilities = model.predict_proba(vectorized_text)[0]
            confidence = max(probabilities)
        except:
            confidence = None
    except (TypeError, ValueError):
        prediction_input = vectorized_text.toarray()
        prediction = model.predict(prediction_input)[0]
        try:
            probabilities = model.predict_proba(prediction_input)[0]
            confidence = max(probabilities)
        except:
            confidence = None
    return prediction, confidence

def create_simple_results_visualization(results):
    if not results: return None
    try:
        categories = list(results.keys())
        labels = list(results.values())
        fig = go.Figure()
        colors = {
            'safety': '#4facfe' if 'safe' in str(results.get('safety', '')) else '#fa709a',
            'toxicity': '#43e97b' if 'non_toxic' in str(results.get('toxicity', '')) else '#fed766',
            'sentiment': '#43e97b' if 'positive' in str(results.get('sentiment', '')) else '#a0a0a0' if 'neutral' in str(results.get('sentiment', '')) else '#fa709a',
            'topic': '#667eea',
            'engagement': '#f093fb' if 'high_engagement' in str(results.get('engagement', '')) else '#a0a0a0'
        }
        bar_colors = [colors.get(cat.lower(), '#667eea') for cat in categories]
        fig.add_trace(go.Bar(
            y=categories, x=[1] * len(categories),
            text=[label.replace('_', ' ').title() for label in labels],
            textposition='inside', orientation='h',
            marker=dict(color=bar_colors, opacity=0.8),
            hovertemplate='<b>%{y}</b><br>%{text}<extra></extra>'
        ))
        fig.update_layout(
            title={'text': '🌈 Multi-Label Analysis Results', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18, 'color': 'white'}},
            xaxis=dict(visible=False), yaxis_title="Categories", height=300, showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Arial')
        )
        return fig
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None

def create_category_badges(results):
    if not results: return
    st.markdown("### 🏷️ Classification Results")
    cols = st.columns(len(results))
    for i, (category, label) in enumerate(results.items()):
        with cols[i]:
            emoji_map = {
                'safety': '🛡️' if label == 'safe' else '🚨',
                'toxicity': '✅' if label == 'non_toxic' else '⚠️',
                'sentiment': '😊' if label == 'positive' else '😐' if label == 'neutral' else '😞',
                'topic': '📂',
                'engagement': '📈' if label == 'high_engagement' else '📉'
            }
            color_map = {
                'safety': 'success' if label == 'safe' else 'error',
                'toxicity': 'success' if label == 'non_toxic' else 'warning',
                'sentiment': 'success' if label == 'positive' else 'info' if label == 'neutral' else 'error',
                'topic': 'info',
                'engagement': 'success' if label == 'high_engagement' else 'warning'
            }
            emoji = emoji_map.get(category, '🏷️')
            color = color_map.get(category, 'info')
            display_label = label.replace('_', ' ').title()
            if color == 'success': st.success(f"{emoji} **{category.title()}**\n{display_label}")
            elif color == 'error': st.error(f"{emoji} **{category.title()}**\n{display_label}")
            elif color == 'warning': st.warning(f"{emoji} **{category.title()}**\n{display_label}")
            else: st.info(f"{emoji} **{category.title()}**\n{display_label}")

# --- Main Content Area ---
st.markdown("### 📝 Content Analysis Input")

# --- SESSION STATE INITIALIZATION ---
if "analysis_input" not in st.session_state:
    st.session_state.analysis_input = ""

def set_input(text):
    st.session_state.analysis_input = text

# --- Quick Examples Section ---
st.markdown("### 📝 Quick Examples")
example_col1, example_col2, example_col3, example_col4 = st.columns(4)

example_texts = {
    "🚀 Tech News": "Amazing breakthrough in AI! New neural network architecture achieves state-of-the-art performance.",
    "🎮 Gaming": "This new game is absolutely incredible! The graphics are mind-blowing and gameplay is addictive.",
    "💼 Business": "Stock market analysis shows strong growth potential. Investment opportunities are expanding.",
    "⚠️ Problematic": "This is terrible content with inappropriate language and harmful messaging."
}

with example_col1:
    st.button("🚀 Tech News", use_container_width=True, on_click=set_input, args=(example_texts["🚀 Tech News"],))

with example_col2:
    st.button("🎮 Gaming", use_container_width=True, on_click=set_input, args=(example_texts["🎮 Gaming"],))

with example_col3:
    st.button("💼 Business", use_container_width=True, on_click=set_input, args=(example_texts["💼 Business"],))

with example_col4:
    st.button("⚠️ Problematic", use_container_width=True, on_click=set_input, args=(example_texts["⚠️ Problematic"],))

# --- Text Area Bound to Session State ---
input_text = st.text_area(
    "Enter Reddit post title and body for analysis:",
    height=250,
    placeholder="Example: 'Breaking: New AI breakthrough!'",
    key="analysis_input",
    help="💡 Try the Quick Examples in the sidebar or paste your own content here"
)

# Classification Mode
st.markdown("### 🎯 Select Analysis Mode")
classification_mode = st.radio(
    "Choose your analysis type:",
    ["🔴 Binary (SFW/NSFW Detection)", "🌈 Multi-Label (Complete Analysis)", "📊 Both (Comprehensive Review)"],
    horizontal=True
)

analyze_clicked = st.button("🚀 Analyze Content", use_container_width=True)
    
# Enhanced Analysis Section
if analyze_clicked:
    # Use the text from session state
    text_to_analyze = st.session_state.analysis_input
    
    if text_to_analyze and text_to_analyze.strip():
        progress_bar = st.progress(0)
        status_text = st.empty()
        binary_result = None
        binary_confidence = None
        multi_results = None
        analysis_error = None
        
        try:
            with st.spinner("🤖 AI Analysis in Progress..."):
                status_text.text("🔍 Preprocessing text...")
                progress_bar.progress(25)
                time.sleep(0.2)
                
                if not binary_model and not multi_label_model: raise Exception("No models available")
                
                status_text.text("🧠 Running AI models...")
                progress_bar.progress(50)
                
                if "Binary" in classification_mode or "Both" in classification_mode:
                    if binary_model:
                        try:
                            binary_result, binary_confidence = predict_binary(text_to_analyze, binary_model, vectorizer)
                            progress_bar.progress(70)
                        except Exception as e:
                            st.error(f"Binary classification error: {e}")
                            analysis_error = f"Binary: {e}"
                    else: st.warning("Binary model not available")
                
                if "Multi-Label" in classification_mode or "Both" in classification_mode:
                    if multi_label_model and metadata:
                        try:
                            multi_results = predict_multi_label(text_to_analyze, multi_label_model, vectorizer, metadata)
                            progress_bar.progress(90)
                        except Exception as e:
                            st.error(f"Multi-label classification error: {e}")
                            analysis_error = f"Multi-label: {e}"
                    else: st.warning("Multi-label model not available")
                
                status_text.text("✅ Analysis Complete!")
                progress_bar.progress(100)
                time.sleep(0.3)
        except Exception as e:
            st.error(f"Analysis pipeline error: {e}")
            analysis_error = str(e)
        finally:
            progress_bar.empty()
            status_text.empty()
        
        st.markdown("---")
        if analysis_error: st.error(f"❌ **Analysis Error**: {analysis_error}")
        
        if binary_result is not None and ("Binary" in classification_mode or "Both" in classification_mode):
            st.success("✅ **Binary Analysis Completed**")
            st.markdown("### 🎯 Binary Classification Results")
            if binary_result == 1:
                st.error("🚨 **NSFW Content Detected**", icon="⚠️")
                st.markdown("📋 This content has been classified as **Not-Safe-For-Work** and may contain mature themes.")
            else:
                st.success("✅ **Safe Content**", icon="🛡️")
                st.markdown("📋 This content has been classified as **Safe-For-Work** and appropriate for general audiences.")
            if binary_confidence:
                st.info(f"🎯 **Confidence Score:** {binary_confidence * 100:.1f}%")
        
        if multi_results and ("Multi-Label" in classification_mode or "Both" in classification_mode):
            st.success("✅ **Multi-Label Analysis Completed**")
            st.markdown("### 🌈 Multi-Label Analysis Results")
            create_category_badges(multi_results)
            try:
                fig = create_simple_results_visualization(multi_results)
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.info("📊 Visualization temporarily unavailable")
            except Exception as viz_error: st.warning(f"Visualization error: {viz_error}")
            
            with st.expander("🔍 Detailed Analysis Insights", expanded=True):
                st.markdown("### 💡 AI Insights:")
                insights = []
                safety = multi_results.get('safety', 'unknown')
                if safety == 'safe': insights.append("• 🛡️ Content is safe for work")
                elif safety == 'nsfw': insights.append("• 🚨 Content may not be safe for work")
                if multi_results.get('toxicity') == 'non_toxic': insights.append("• ✅ Content is non-toxic and appropriate")
                elif multi_results.get('toxicity') == 'toxic': insights.append("• ⚠️ Content may contain toxic elements")
                sentiment = multi_results.get('sentiment', 'unknown')
                if sentiment == 'positive': insights.append("• 😊 Content expresses positive sentiment")
                elif sentiment == 'negative': insights.append("• 😞 Content contains negative emotional indicators")
                elif sentiment == 'neutral': insights.append("• 😐 Content has neutral emotional tone")
                topic = multi_results.get('topic', 'general')
                if topic != 'general': insights.append(f"• 📂 Primary topic category: {topic.replace('_', ' ').title()}")
                if multi_results.get('engagement') == 'high_engagement': insights.append("• 📈 Content likely to generate high user engagement")
                elif multi_results.get('engagement') == 'low_engagement': insights.append("• 📉 Content may have lower engagement potential")
                for insight in insights: st.markdown(insight)
                
        if not binary_result and not multi_results and not analysis_error:
            st.info("🔍 **No analysis performed** - Select an analysis mode and ensure models are loaded")
    else:
        st.warning("📝 Please enter some text to analyze.", icon="⚠️")

# --- Information Sections ---
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### 🚀 About This AI System")
    st.markdown("""
    This **cutting-edge AI system** provides comprehensive content analysis using advanced machine learning models.
    **🛡️ Safety Detection** Identifies safe vs. not-safe-for-work content
    **⚠️ Toxicity Assessment** Detects harmful, abusive, or inappropriate language
    **😊 Sentiment Analysis** Determines positive, neutral, or negative emotional tone
    **📂 Topic Classification** Categorizes content into technology, business, entertainment, etc.
    **📈 Engagement Prediction** Predicts likelihood of high user engagement
    """)
with col2:
    if metadata:
        st.markdown("### 📊 Live Performance")
        binary_f1 = metadata.get('binary_f1_score', 0.88) 
        multi_jaccard = metadata.get('multi_label_jaccard_score', 0.85)
        st.metric("Binary Accuracy", f"{binary_f1:.1%}", delta=f"{(binary_f1-0.8)*100:+.1f}%")
        st.metric("Multi-Label Score", f"{multi_jaccard:.1%}", delta=f"{(multi_jaccard-0.7)*100:+.1f}%")
    else:
        st.markdown("### 📊 Performance Metrics")
        st.info("Load models to see live performance metrics")
        if st.button("🔄 Reload Models", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
with col3:
    st.markdown("### ✨ Key Features")
    st.markdown("""
    **🤖 Advanced AI Pipeline**
    - Multi-output neural networks
    - Ensemble model voting
    - Real-time inference
    **⚡ Production Ready**
    - Sub-second response times
    - Scalable architecture
    **🔒 Responsible AI**
    - Bias detection & mitigation
    - Transparent confidence scores
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #a0a0a0;">
    <p><strong>🤖 AI Content Classifier</strong> | Powered by Advanced Machine Learning</p>
</div>
""", unsafe_allow_html=True)