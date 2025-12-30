import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Page Configuration
st.set_page_config(page_title="Skill Drift Monitor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Montserrat:wght@700&display=swap');

    /* Global Typography */
    html, body {
    font-family: 'Inter', sans-serif;
    background-color: #0e1117;
    color: #ffffff;
    }

    /* Apply font only to main content — NOT internal Streamlit UI */
    section.main * {
        font-family: 'Inter', sans-serif;
    }


    /* Professional Headings */
    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif !important;
        letter-spacing: -0.5px;
    }

    /* Mode Indicator Badge */
    .status-header {
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .demo-mode { background: rgba(52, 152, 219, 0.1); color: #3498db; border-left: 5px solid #3498db; }
    .audit-mode { background: rgba(155, 89, 182, 0.1); color: #9b59b6; border-left: 5px solid #9b59b6; }

    /* Glassmorphism Card Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Sidebar Glass & Font Style */
    [data-testid="stSidebar"] {
        background: rgba(26, 28, 36, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Sidebar Profile Section */
    .profile-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 30px 10px;
        background: rgba(255, 255, 255, 0.03); 
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 10px;
        text-align: center;
    }

    .profile-pic {
        border-radius: 50%;
        padding: 4px;
        background: linear-gradient(45deg, #2ecc71, #27ae60); 
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.6);
        margin-bottom: 15px;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    .admin-name {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .status-tag {
        font-family: 'Inter', sans-serif;
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        padding: 3px 15px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(46, 204, 113, 0.4);
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Custom Suggestion Buttons */
    div.stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background-color: rgba(255, 75, 75, 0.05);
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
        border-radius: 25px;
        padding: 0.5rem 1.2rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #ff4b4b;
        color: white;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)