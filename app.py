import streamlit as st
from PIL import Image
from predict_utils import analyze_image

st.set_page_config(page_title="Image Authenticity Verification", layout="centered")

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .sub-text {
        font-size: 1rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }
    .card {
        background: rgba(255, 255, 255, 0.03);
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 12px;
    }
    .verdict-box {
        padding: 14px;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin-top: 10px;
        border: 1px solid rgba(255, 255, 255, 0.10);
        background: rgba(124, 58, 237, 0.12);
    }
    .confidence-text {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown('<div class="main-title">🧠 Image Authenticity Verification System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Upload an image to detect whether it is <b>Real</b>, <b>AI Generated</b>, or <b>Morphed / Edited</b>.</div>',
    unsafe_allow_html=True
)

# -------------------- Upload Section --------------------
uploaded_file = st.file_uploader("📌 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🔍 Analyzing image... Please wait"):
        result = analyze_image(img)

    # -------------------- Final Result --------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader(" Final Result")

    st.markdown(
        f'<div class="verdict-box">Final Verdict: {result["verdict"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="confidence-text"> Confidence</div>', unsafe_allow_html=True)
    st.write(f"**Fake / AI Content:** {result['fake_percent']:.2f}%")
    st.write(f"**Real Content:** {result['real_percent']:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------- Debug / Advanced --------------------
    with st.expander(" Advanced Details (Patch Analysis)"):
        st.write(f"**Morph Score (Patch Variance):** {result['morph_score']:.4f}")
        st.write("Patch-wise Fake Probability:")
        st.write(result["patch_probs"])

else:
    st.info("👆 Upload an image to begin analysis.")