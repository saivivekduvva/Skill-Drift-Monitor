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

# Page Configuration
st.set_page_config(page_title="Skill Drift Monitor", layout="wide", initial_sidebar_state="expanded")

# Injecting Google Fonts, Glassmorphism, and Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Montserrat:wght@700&display=swap');

    html, body {
        font-family: 'Inter', sans-serif;
        background-color: #0e1117;
        color: #ffffff;
    }

    section.main * { font-family: 'Inter', sans-serif; }

    h1, h2, h3, h4 {
        font-family: 'Montserrat', sans-serif !important;
        letter-spacing: -0.5px;
    }

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

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        margin-bottom: 25px;
    }

    [data-testid="stSidebar"] {
        background: rgba(26, 28, 36, 0.8) !important;
        backdrop-filter: blur(15px);
    }

    .profile-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 30px 10px;
        background: rgba(255, 255, 255, 0.03); 
        border-radius: 20px;
        margin: 10px;
    }

    .profile-pic {
        border-radius: 50%;
        padding: 4px;
        background: linear-gradient(45deg, #2ecc71, #27ae60); 
        box-shadow: 0 0 15px rgba(46, 204, 113, 0.6);
        margin-bottom: 15px;
    }

    .status-tag {
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        padding: 3px 15px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(46, 204, 113, 0.4);
    }

    div.stButton > button {
        border-radius: 25px;
        transition: all 0.3s ease;
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
            <div style="font-weight: 700; color: white;">System Administrator</div>
            <div class="status-tag">● Session Active</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.title("Settings")
    mode_options = {"📊 Demo Dataset": "Demo Dataset", "🏛 Institution Audit": "Institution Audit"}
    selected_label = st.radio("Analysis Mode", list(mode_options.keys()), index=0)
    analysis_mode = mode_options[selected_label]

# Main Content
st.title("🎓 Academic–Industry Skill Drift Monitor")

if analysis_mode == "Demo Dataset":
    st.markdown('<div class="status-header demo-mode">📊 CURRENT MODE: <b>SANDBOX / DEMO DATASET</b></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-header audit-mode">🏛 CURRENT MODE: <b>OFFICIAL INSTITUTION AUDIT</b></div>', unsafe_allow_html=True)

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

    # 1. STRATEGIC EXPLORER (SEARCH & FILTERS)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 Strategic Role Explorer")
    all_titles = jobs_df["Job Title"].unique().tolist()
    search_col, sort_col = st.columns([3, 1])
    
    if 'search_val' not in st.session_state: st.session_state.search_val = ""
    search_query = search_col.text_input("Quick Search Job Roles", value=st.session_state.search_val)
    sort_order = sort_col.selectbox("Sort Priority", ["Highest Alignment", "Lowest Alignment", "Alphabetical"])
    st.markdown('</div>', unsafe_allow_html=True)

    display_df = jobs_df[jobs_df["Job Title"].str.contains(search_query, case=False, na=False)]

    if not display_df.empty:
        avg_sim = display_df['alignment_score'].mean()
        drift_pct = (len(display_df[display_df["Level"] == "Low Alignment"]) / len(display_df)) * 100

        # 2. EDUCATIONAL IMPACT SUMMARY (METRICS)
        st.markdown("## 📌 Educational Impact Summary")
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px; border-left: 5px solid #3498db; margin-bottom: 20px;">
                <h5 style="margin-top:0;">📖 Key Terminology Guide</h5>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                    <p style="font-size: 0.85rem; color: #ccc;"><b>Curriculum Risk:</b> % of roles where the syllabus lacks necessary modern keywords.</p>
                    <p style="font-size: 0.85rem; color: #ccc;"><b>Syllabus Match:</b> Direct overlap score between syllabus and job duties.</p>
                    <p style="font-size: 0.85rem; color: #ccc;"><b>Roles Analyzed:</b> Unique career paths compared against your data.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("📉 Curriculum Risk", f"{drift_pct:.1f}%")
        col2.metric("🤝 Syllabus Match Score", f"{avg_sim:.2%}")
        col3.metric("📁 Roles Analyzed", len(display_df))

        # 3. BAR CHART SECTION
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔎 Job-to-Syllabus Relevance Breakdown")
        st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(5px); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 20px;">
                <h6 style="margin: 0 0 10px 0; color: #3498db;">🎨 Chart Legend</h6>
                <span style="color: #2ecc71; margin-right: 20px;">● <b>Green:</b> High Alignment (Up-to-date)</span>
                <span style="color: #f1c40f; margin-right: 20px;">● <b>Yellow:</b> Partial Alignment (Update Suggested)</span>
                <span style="color: #e74c3c;">● <b>Red:</b> Low Alignment (Critical Risk)</span>
            </div>
        """, unsafe_allow_html=True)

        if sort_order == "Highest Alignment": display_df = display_df.sort_values("alignment_score", ascending=False)
        elif sort_order == "Lowest Alignment": display_df = display_df.sort_values("alignment_score", ascending=True)
        else: display_df = display_df.sort_values("Job Title")

        fig_bar = px.bar(
            display_df.head(15), x='alignment_score', y='Job Title', orientation='h',
            color='alignment_score', color_continuous_scale='RdYlGn',
            labels={"alignment_score": "Match Quality"}, template="plotly_dark", height=450
        )
        fig_bar.update_xaxes(range=[0, 1])
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 4. SYLLABUS HEALTH SECTION (PIE CHART & INSTRUCTION MANUAL)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Syllabus Health Analysis")
        
        col_pie, col_manual = st.columns([1.2, 1])
        
        with col_pie:
            counts = display_df["Level"].value_counts()
            fig_pie = px.pie(
                names=counts.index, 
                values=counts.values, 
                hole=0.6, 
                template="plotly_dark",
                color_discrete_sequence=["#2ecc71", "#f1c40f", "#e74c3c"]
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_manual:
            st.markdown("""
                <div style="
                    background: rgba(255, 255, 255, 0.03); 
                    padding: 25px; 
                    border-radius: 12px; 
                    border: 1px solid rgba(255,255,255,0.05);
                    height: 100%;
                ">
                    <h5 style="margin-top:0; color: #ffffff;">📊 Instruction Manual</h5>
                    <p style="font-size: 0.9rem; color: #aaaaaa; line-height: 1.6;">
                        This chart shows the percentage of your analyzed job market categorized by <b>Syllabus Health</b>.
                    </p>
                    <ul style="font-size: 0.9rem; color: #dddddd; padding-left: 20px; line-height: 1.8;">
                        <li><b>Green Segment:</b> Roles where your syllabus is a strong match.</li>
                        <li><b>Yellow Segment:</b> Roles where you are missing roughly 50% of required skills.</li>
                        <li><b>Red Segment:</b> Roles that are currently outside your curriculum's reach.</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # 5. DATA TABLE SECTION (BELOW PIE & MANUAL)
        st.markdown("#### 📋 Detailed Data View")
        st.dataframe(display_df[["Job Title", "Level", "alignment_score"]], use_container_width=True, height=400)
        st.markdown('</div>', unsafe_allow_html=True)

        st.sidebar.download_button("📥 Export Drift Report", display_df.to_csv(), "drift_report.csv")

except Exception as e:
    st.error(f"Waiting for data input... (System Error: {e})")