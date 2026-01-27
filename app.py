# ==============================
# News Credibility Assistant
# FINAL STABLE VERSION
# ==============================

import streamlit as st
import joblib, re, nltk, time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline(
        "text2text-generation",
        model="facebook/bart-large-cnn"
    )


# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="News Credibility Assistant",
    page_icon="📰",
    layout="centered"
)

# ------------------------------
# DEFAULT THEME (DARK)
# ------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ------------------------------
# TOP-RIGHT THEME TOGGLE (FIXED)
# ------------------------------
toggle_col, _ = st.columns([2, 7])
with toggle_col:
    is_light = st.toggle("Theme 🌙⇄☀️", value=(st.session_state.theme == "light"))
    st.session_state.theme = "light" if is_light else "dark"

# ------------------------------
# THEME COLORS
# ------------------------------
if st.session_state.theme == "dark":
    BG = "linear-gradient(135deg,#0f2027,#203a43,#2c5364)"
    CARD = "rgba(255,255,255,0.12)"
    TXT = "#ffffff"
    SUBTXT = "#d0d0d0"
    INPUT_BG = "#1f2a33"
    INPUT_TXT = "#ffffff"
else:
    BG = "linear-gradient(135deg,#f5f7fa,#c3cfe2)"
    CARD = "rgba(255,255,255,0.98)"
    TXT = "#111111"
    SUBTXT = "#222222"
    INPUT_BG = "#ffffff"
    INPUT_TXT = "#000000"

# ------------------------------
# GLOBAL VISIBILITY CSS (FINAL)
# ------------------------------


st.markdown(f"""
<style>
.stApp {{
    background: {BG};
}}

.glass {{
    background: {CARD};
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 24px;
    margin-top: 24px;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {TXT} !important;
}}

p, li, ul, ol {{
    color: {TXT} !important;
}}

label, span, div {{
    color: {TXT} !important;
}}

textarea {{
    background-color: {INPUT_BG} !important;
    color: {INPUT_TXT} !important;
}}

textarea::placeholder {{
    color: #777777 !important;
}}

.footer {{
    text-align:center;
    color:{SUBTXT};
    font-size:13px;
    margin-top:40px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# SIDEBAR VISIBILITY FIX (LIGHT + DARK)
# ------------------------------
st.markdown("""
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1e1e2f;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Light mode override */
body[data-theme="light"] [data-testid="stSidebar"] {
    background-color: #f2f4f8;
}

body[data-theme="light"] [data-testid="stSidebar"] * {
    color: #111111 !important;
}

/* Sidebar radio labels */
[data-testid="stSidebar"] label {
    color: inherit !important;
}
</style>
""", unsafe_allow_html=True)
# ------------------------------
# SIDEBAR (COLLAPSIBLE MENU STYLE)
# ------------------------------
with st.sidebar:
    st.markdown("## 🎓 News Credibility Assistant")

    with st.expander("📌 What can you do here?", expanded=True):
        st.markdown("""
- Analyze news credibility  
- Detect bias & emotional tone  
- Compare two news articles  
- Read neutral AI summaries  
- Learn how misinformation works  
""")

    with st.expander("🎯 Designed for"):
        st.markdown("""
- University students  
- Competitive exam aspirants  
- Researchers & educators  
""")

    with st.expander("🧭 How to use"):
        st.markdown("""
1. Paste a news article  
2. Click **Analyze**  
3. Review credibility & bias  
4. Read AI summary  
5. Decide responsibly  
""")

    with st.expander("⚠️ Important"):
        st.markdown("""
This tool does **not declare truth**.  
It highlights **credibility patterns**  
to support **critical thinking**.
""")

    with st.expander("🚀 Coming next"):
        st.markdown("""
- Fact-check API integration  
- Browser extension  
- Multilingual support  
- Source URL verification  
""")
# ------------------------------
# TEXTAREA TEXT VISIBILITY FIX (LIGHT MODE ONLY)
# ------------------------------
st.markdown("""
<style>
/* Force textarea text to be visible in light mode */
body[data-theme="light"] textarea {
    color: #000000 !important;
}

/* Keep placeholder readable */
body[data-theme="light"] textarea::placeholder {
    color: #666666 !important;
}
</style>
""", unsafe_allow_html=True)
# ------------------------------
# BUTTON TEXT VISIBILITY FIX (LIGHT MODE)
# ------------------------------
st.markdown("""
<style>
/* Force Streamlit button text & icons visible in light mode */
body[data-theme="light"] button[kind="primary"],
body[data-theme="light"] button[kind="secondary"] {
    color: #ffffff !important;
}

/* Ensure SVG icons inside buttons are visible */
body[data-theme="light"] button svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

/* Optional: ensure contrast stays good */
body[data-theme="light"] button {
    background-color: #0f172a !important;
}
</style>
""", unsafe_allow_html=True)



# ------------------------------
# LOAD NLTK
# ------------------------------
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

load_nltk()



# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_ml():
    return joblib.load("fake_news_model.pkl"), joblib.load("tfidf_vectorizer.pkl")

model, vectorizer = load_ml()

# ------------------------------
# LOAD SUMMARIZER
# ------------------------------
if "summarizer" not in st.session_state:
    with st.spinner("🔄 Loading AI summarizer..."):
        st.session_state.summarizer = pipeline(
            "text2text-generation",
            model="facebook/bart-large-cnn",
            device=-1
        )

summarizer = st.session_state.summarizer

# ------------------------------
# NLP SETUP
# ------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text.lower())
    text = re.sub(r"[^a-z\s]", "", text)
    return " ".join(lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words)

# ------------------------------
# HELPERS
# ------------------------------
TRUSTED = ["reuters","bbc","cnn","who","government","official"]
SENSATIONAL = ["shocking","breaking","secret","exposed"]
OPINION = ["clearly","obviously","no doubt","everyone knows"]

def source_trust(text):
    return min(0.5 + 0.1 * sum(s in text.lower() for s in TRUSTED), 1.0)

def detect_bias(text):
    sens = "High" if sum(w in text.lower() for w in SENSATIONAL) >= 2 else "Low"
    tone = "Opinionated" if any(w in text.lower() for w in OPINION) else "Neutral"
    return sens, tone

def explain_score(conf, src):
    reasons = []
    if conf < 0.6: reasons.append("Unclear or neutral language patterns detected")
    if src < 0.6: reasons.append("Trusted sources not clearly referenced")
    if not reasons: reasons.append("Language and source cues match reliable reporting")
    return reasons

def student_actions(score):
    if score < 0.5:
        return ["Avoid sharing", "Verify with official sources", "Cross-check multiple outlets"]
    elif score < 0.7:
        return ["Compare with trusted news outlets", "Check official statements"]
    else:
        return ["Still verify key facts", "Use information responsibly"]

def analyze(text):
    X = vectorizer.transform([clean_text(text)])
    conf = model.predict_proba(X)[0].max()
    src = source_trust(text)
    return conf, src, (0.7*conf + 0.3*src)

def summarize(text):
    if len(text.split()) < 50:
        return "Text too short to summarize."

    result = summarizer(
        text[:1024],
        max_length=80,
        min_length=30,
        do_sample=False
    )

    return result[0]["generated_text"]


#def summarize(text):
#    if len(text.split()) < 50:
 #       return "Text too short to summarize."
   # return summarizer(text[:1024], max_length=80, min_length=30, do_sample=False)[0]["summary_text"]

# ------------------------------
# SESSION HISTORY
# ------------------------------
st.session_state.setdefault("history", [])

# ------------------------------
# HEADER
# ------------------------------
st.markdown("<h1>📰 AI-Powered News Credibility Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze • Compare • Learn Responsibly</p>", unsafe_allow_html=True)

# ------------------------------
# MODE SELECT
# ------------------------------
mode = st.radio("Mode:", ["Single Article", "Compare Two Articles"], horizontal=True)

# ==============================
# SINGLE ARTICLE MODE
# ==============================
if mode == "Single Article":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    text = st.text_area("Paste news article:", height=220, placeholder="Paste full news article text here...")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Analyze"):
        with st.spinner("🧠 Analyzing article..."):
            conf, src, final = analyze(text)
            sens, tone = detect_bias(text)

        st.session_state.history.append(final)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.progress(final)
        st.markdown(f"**Credibility:** {round(final*100,2)}%")

        st.markdown("### 📣 Bias & Tone")
        st.write(f"- Sensational Language: {sens}")
        st.write(f"- Emotional Tone: {tone}")

        st.markdown("### 🤔 Why this score?")
        for r in explain_score(conf, src):
            st.write(f"- {r}")

        st.markdown("### 🧭 What should you do next?")
        for a in student_actions(final):
            st.write(f"• {a}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📝 AI Summary")
        with st.spinner("✍️ Generating summary..."):
            st.info(summarize(text))
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# COMPARE MODE
# ==============================
else:
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.text_area("Article A", height=180)
    with c2:
        t2 = st.text_area("Article B", height=180)

    if st.button("🔄 Compare"):
        with st.spinner("Comparing articles..."):
            _, _, s1 = analyze(t1)
            _, _, s2 = analyze(t2)

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.write(f"📰 Article A Credibility: **{round(s1*100,2)}%**")
        st.write(f"📰 Article B Credibility: **{round(s2*100,2)}%**")
        st.caption("Comparison highlights relative credibility, not absolute truth.")
        st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# LEARNING JOURNEY
# ------------------------------
if st.session_state.history:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("🎓 Your Learning Journey")
    avg = sum(st.session_state.history)/len(st.session_state.history)
    st.write(f"Articles analyzed: {len(st.session_state.history)}")
    st.write(f"Average credibility: {round(avg*100,2)}%")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# TRANSPARENCY
# ------------------------------
with st.expander("📘 How this system works"):
    st.write(
        "- ML model trained on real & fake news datasets\n"
        "- Combines language patterns with source cues\n"
        "- Provides explanations instead of black-box decisions\n"
        "- Designed for education, not fact declaration"
    )

with st.expander("⚙️ Technical Highlights"):
    st.write(
        "- Kaggle + LIAR datasets\n"
        "- TF-IDF + classical ML\n"
        "- Hybrid scoring approach\n"
        "- Ethical AI with uncertainty handling"
    )

# ------------------------------
# FOOTER
# ------------------------------
st.markdown('<div class="footer">⚠️ Educational AI assistant. Encourages critical thinking.</div>', unsafe_allow_html=True)




