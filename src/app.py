import streamlit as st
import time

from rag_pipeline import generate_answer
from hallucination import detect_hallucination

# ==========================================
# 🌌 SPARKLING ICE-BLUE CONSTELLATION THEME
# ==========================================
st.set_page_config(page_title="ScholarRadar AI", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    /* 1. Moving Constellation Background */
    @keyframes moveStars {
        from { background-position: 0 0; }
        to { background-position: -1000px 500px; }
    }

    .stApp {
        background-color: #02050a; /* Deep eternal dark space */
        background-image: 
            radial-gradient(circle at top right, rgba(10, 20, 35, 0.8), transparent),
            url("https://www.transparenttextures.com/patterns/stardust.png");
        animation: moveStars 120s linear infinite;
        color: #e6f1ff; 
    }

    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* 2. Smooth, Rounded Inputs with Ice Blue Glow */
    .stChatInputContainer textarea {
        background-color: rgba(5, 12, 25, 0.8) !important;
        color: #7DF9FF !important; /* Sparkling Electric Blue typing text */
        border: 1px solid rgba(125, 249, 255, 0.3) !important; 
        border-radius: 25px !important; 
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5), inset 0 0 8px rgba(125, 249, 255, 0.05);
        padding: 15px 20px !important;
    }

    /* 3. Transmitting Radar Animation (Whitish Blue) */
    @keyframes radar-transmit-blue {
        0% { box-shadow: 0 0 0 0px rgba(125, 249, 255, 0.7), inset 0 0 0 0px rgba(125, 249, 255, 0.4); }
        50% { box-shadow: 0 0 0 15px rgba(125, 249, 255, 0), inset 0 0 10px 2px rgba(125, 249, 255, 0.2); }
        100% { box-shadow: 0 0 0 0px rgba(125, 249, 255, 0), inset 0 0 0 0px rgba(125, 249, 255, 0); }
    }
    .radar-logo {
        width: 70px; height: 70px;
        border: 2px solid #7DF9FF;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        animation: radar-transmit-blue 1.5s infinite ease-out;
        color: #E0FFFF; /* Whitish blue icon */
        font-size: 30px;
        margin-bottom: 25px;
        background: radial-gradient(circle, rgba(125,249,255,0.15) 0%, transparent 60%);
    }

    /* 4. Smooth Glassmorphism Source Cards */
    .source-card {
        background: rgba(10, 18, 30, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(125, 249, 255, 0.15); 
        border-bottom: 2px solid #7DF9FF; /* Ice Blue underline */
        border-radius: 16px; 
        padding: 12px 18px;
        margin-top: 10px;
        font-size: 13px;
        color: #E0FFFF;
        display: inline-block;
        margin-right: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .source-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(125, 249, 255, 0.2);
    }

    /* Ice Blue Typography Accents with "Sparkle" Text-Shadow */
    h1, h2, h3, .sidebar-text {
        color: #7DF9FF !important;
        font-weight: 600;
        text-shadow: 0px 0px 10px rgba(125, 249, 255, 0.3); /* The Sparkle Effect */
    }

    /* 5. Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(3, 8, 15, 0.85) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(125, 249, 255, 0.05);
        box-shadow: 2px 0 20px rgba(0,0,0,0.5);
    }
    </style>

    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """, unsafe_allow_html=True)

# ==========================================
# 🛰️ SIDEBAR (COMMAND CENTER)
# ==========================================
with st.sidebar:
    # Transmitting Radar Logo
    st.markdown('<div class="radar-logo"><i class="fas fa-satellite-dish"></i></div>', unsafe_allow_html=True)
    st.markdown("<h1>ScholarRadar</h1>", unsafe_allow_html=True)
    st.caption("Deep Research Assistant")
    st.write("---")

    st.info("**Engine:** Mistral-7B (Local)\n\n**Retriever:** FAISS + Cross-Encoder")

    st.write("---")

    # Updated Trust Score Label
    st.markdown("<h3>🛡️ Confidence</h3>", unsafe_allow_html=True)
    reliability_placeholder = st.empty()
    reliability_placeholder.markdown("<p style='color: gray'>Awaiting Query...</p>", unsafe_allow_html=True)

# ==========================================
# 💬 MAIN CHAT INTERFACE
# ==========================================
st.markdown("<h1>What would you like to research?</h1>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your research query..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.status("📡 Radar Transmitting & Scanning...", expanded=True) as status:
            st.write("🔍 Extracting Top 50 chunks...")
            time.sleep(0.3)
            st.write("🎯 Applying Cross-Encoder Re-ranking...")

            try:
                answer, sources, context = generate_answer(prompt)

                st.write("🛡️ Evaluating Hallucination Risk...")
                score, risk_status = detect_hallucination(answer, context)

                status.update(label="✅ Transmission Complete", state="complete", expanded=False)
                success = True
            except Exception as e:
                status.update(label="❌ Signal Lost", state="error", expanded=False)
                st.error(f"Error details: {e}")
                success = False

        if success:
            # Smooth Source Cards with Whitish Blue Icons
            st.markdown("**Sources Analyzed:**")
            source_html = "".join(
                [f'<div class="source-card"><i class="fas fa-file-pdf" style="color: #7DF9FF;"></i> {src}</div>' for src
                 in sources])
            st.markdown(source_html, unsafe_allow_html=True)
            st.write("")

            # Formatted Streaming Output
            placeholder = st.empty()
            full_response = ""
            lines = answer.split('\n')
            for line in lines:
                for chunk in line.split(' '):
                    full_response += chunk + " "
                    time.sleep(0.04)
                    placeholder.markdown(full_response + "▌")
                full_response += "\n\n"
            placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

            # 🔥 Update: Confidence Format (Ice Blue for good, Amber for warning)
            color = "#7DF9FF" if score >= 0.75 else "#ffb703"
            reliability_placeholder.markdown(
                f"<h2 style='color:{color}'>{score:.2f}</h2><p style='color:{color}'>{risk_status}</p>",
                unsafe_allow_html=True
            )