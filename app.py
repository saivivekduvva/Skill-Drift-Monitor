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

# Injecting Google Fonts, Glassmorphism, and Glowing Profile CSS
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

# Data Processing
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z ]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)


# Sidebar & Mode Selection
with st.sidebar:
    st.markdown(f"""
        <div class="profile-container">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" class="profile-pic" width="100">
            <div class="admin-name">System Administrator</div>
            <div class="status-tag">● Session Active</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.title("Settings")
    # Added Icons to Radio labels for visual distinction
    mode_options = {
        "📊 Demo Dataset": "Demo Dataset",
        "🏛 Institution Audit": "Institution Audit"
    }
    selected_label = st.radio("Analysis Mode", list(mode_options.keys()), index=0)
    analysis_mode = mode_options[selected_label]
    st.divider()


# Main Content
st.title("🎓 Academic–Industry Skill Drift Monitor")

# Visual feedback for current mode
if analysis_mode == "Demo Dataset":
    st.markdown('<div class="status-header demo-mode">📊 CURRENT MODE: <b>SANDBOX / DEMO DATASET</b></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-header audit-mode">🏛 CURRENT MODE: <b>OFFICIAL INSTITUTION AUDIT</b></div>', unsafe_allow_html=True)

st.markdown("<style>div[data-testid='stVerticalBlock'] > div:empty { display: none; }</style>", unsafe_allow_html=True)

try:
    jobs_df = pd.read_csv("Glassdoor_Salary_Cleaned_Version.csv")
    
    if analysis_mode == "Demo Dataset":
        academic_df = pd.read_csv("synthetic_academic_syllabus_1000_rows.csv")
    else:
        uploaded = st.sidebar.file_uploader("Upload Institution Syllabus (CSV)", type=["csv"])
        if uploaded: 
            academic_df = pd.read_csv(uploaded)
        else: 
            st.info("👋 Waiting for file upload. Please upload your institutional syllabus CSV via the sidebar to begin the audit.")
            st.stop()

    # Pre-processing & Similarity
    with st.spinner("Analyzing skill alignment..."):
        jobs_df["clean_text"] = (jobs_df["Job Title"] + " " + jobs_df["Job Description"]).apply(clean_text)
        academic_df["clean_text"] = academic_df.astype(str).agg(" ".join, axis=1).apply(clean_text)

        vectorizer = TfidfVectorizer(max_features=1000)
        all_text = pd.concat([academic_df["clean_text"], jobs_df["clean_text"]])
        tfidf_matrix = vectorizer.fit_transform(all_text)
        
        acad_vec = tfidf_matrix[:len(academic_df)]
        job_vec = tfidf_matrix[len(academic_df):]
        
        sim_scores = cosine_similarity(job_vec, acad_vec).max(axis=1)
        jobs_df["alignment_score"] = sim_scores
        
        def get_label(s):
            if s > 0.30: return "High Alignment"
            if s > 0.12: return "Partial Alignment"
            return "Low Alignment"
        jobs_df["Level"] = jobs_df["alignment_score"].apply(get_label)

# --------------------------------------------------
    # STRATEGIC EXPLORER
    # --------------------------------------------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Strategic Role Explorer")
    
    all_titles = jobs_df["Job Title"].unique().tolist()
    search_col, sort_col = st.columns([3, 1])
    
    if 'search_val' not in st.session_state:
        st.session_state.search_val = ""

    search_query = search_col.text_input("Quick Search Job Roles", value=st.session_state.search_val)
    sort_order = sort_col.selectbox("Sort Priority", ["Highest Alignment", "Lowest Alignment", "Alphabetical"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Filtering & "Did You Mean"
    display_df = jobs_df[jobs_df["Job Title"].str.contains(search_query, case=False, na=False)]

    if display_df.empty and search_query != "":
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.warning(f"No exact matches for '{search_query}'.")
        suggestions = difflib.get_close_matches(search_query, all_titles, n=6, cutoff=0.3)
        if suggestions:
            st.markdown("#### Suggested Roles:")
            cols = st.columns(3)
            for i, suggestion in enumerate(suggestions):
                if cols[i % 3].button(suggestion, key=f"sug_{i}"):
                    st.session_state.search_val = suggestion
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --------------------------------------------------
    # ANALYTICS DASHBOARD
    # --------------------------------------------------
    if not display_df.empty:

        avg_sim = display_df['alignment_score'].mean()
        drift_pct = (len(display_df[display_df["Level"] == "Low Alignment"]) / len(display_df)) * 100

        # ================= HERO EXECUTIVE SNAPSHOT =================
        st.markdown("## 📌 Executive Curriculum Health Snapshot")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📉 Curriculum Risk", f"{drift_pct:.1f}%",
                    help="Percentage of job roles with low syllabus relevance")

        with col2:
            st.metric("📊 Average Alignment", f"{avg_sim:.2%}",
                    help="Overall curriculum relevance to industry")

        with col3:
            st.metric("📁 Roles Analyzed", len(display_df))
        # ===========================================================

        if sort_order == "Highest Alignment":
            display_df = display_df.sort_values("alignment_score", ascending=False)
        elif sort_order == "Lowest Alignment":
            display_df = display_df.sort_values("alignment_score", ascending=True)
        else:
            display_df = display_df.sort_values("Job Title")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown("### 🔎 Strategic Role Explorer")

        with st.expander("🧭 How to read this chart"):
            st.markdown("""
            - *Each bar represents one job role*
            - *Bar length shows curriculum–industry relevance score*
            - *Score range is from 0 to 1*
            """)

