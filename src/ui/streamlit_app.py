"""Streamlit web interface for Arabic OCR."""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inference import ArabicOCRPredictor

# Page config
st.set_page_config(
    page_title="Arabic OCR System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.main-title {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.result-box {
    padding: 1.5rem;
    background-color: #f0f2f6;
    border-radius: 10px;
    margin: 1rem 0;
    direction: rtl;
    text-align: right;
    font-size: 1.5rem;
}
.confidence-score {
    font-size: 1.2rem;
    color: #28a745;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load OCR model (cached)."""
    try:
        predictor = ArabicOCRPredictor(
            model_path="models/best_model.pth",
            device="auto",
        )
        return predictor, None
    except Exception as e:
        return None, str(e)


def main():
    """Main application."""

    # Title
    st.markdown('<h1 class="main-title">📝 Arabic OCR System</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    mode = st.sidebar.radio(
        "Mode",
        ["Upload Image", "Live Camera", "Batch Processing"],
        help="Select input mode"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Parameters")

    beam_width = st.sidebar.slider(
        "Beam Width",
        min_value=1,
        max_value=10,
        value=5,
        help="Higher values = better accuracy but slower"
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Minimum confidence to display results"
    )

    show_alternatives = st.sidebar.checkbox(
        "Show Alternatives",
        value=False,
        help="Display alternative predictions"
    )

    # Load model
    with st.spinner("Loading model..."):
        predictor, error = load_model()

    if predictor is None:
        st.error(f"❌ Failed to load model: {error}")
        st.info("Please ensure the model is trained and saved to 'models/best_model.pth'")
        return

    st.sidebar.success("✅ Model loaded successfully!")
    st.sidebar.info(f"Device: {predictor.device}")
    st.sidebar.info(f"Vocabulary: {predictor.vocabulary.vocab_size} tokens")

    # Main content
    if mode == "Upload Image":
        upload_mode(predictor, beam_width, confidence_threshold, show_alternatives)
    elif mode == "Live Camera":
        camera_mode(predictor, beam_width, confidence_threshold)
    elif mode == "Batch Processing":
        batch_mode(predictor, beam_width)


def upload_mode(predictor, beam_width, confidence_threshold, show_alternatives):
    """Upload image mode."""
    st.header("📤 Upload Image")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image containing Arabic text"
        )

        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if uploaded_file is not None:
            with st.spinner("🔍 Recognizing text..."):
                # Predict
                predictor.beam_width = beam_width
                result = predictor.predict(
                    image,
                    return_confidence=True,
                    return_alternatives=show_alternatives,
                )

            # Display results
            st.subheader("Recognition Result")

            if result.get("confidence", 1.0) >= confidence_threshold:
                st.markdown(
                    f'<div class="result-box">{result["text"]}</div>',
                    unsafe_allow_html=True
                )

                # Metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
                with col_b:
                    st.metric("Processing Time", f"{result['processing_time']:.3f}s")

                # Alternatives
                if show_alternatives and result.get("alternatives"):
                    st.subheader("Alternative Predictions")
                    for i, alt in enumerate(result["alternatives"], 1):
                        st.write(f"{i}. {alt['text']} ({alt['confidence']:.2%})")

                # Download button
                st.download_button(
                    label="📥 Download Text",
                    data=result["text"],
                    file_name="ocr_result.txt",
                    mime="text/plain",
                )
            else:
                st.warning("⚠️ Confidence below threshold. No result displayed.")


def camera_mode(predictor, beam_width, confidence_threshold):
    """Live camera mode."""
    st.header("📷 Live Camera")

    st.info("📸 Click 'Take Photo' to capture and recognize text")

    camera_input = st.camera_input("Take a photo")

    if camera_input is not None:
        # Convert to PIL Image
        image = Image.open(camera_input)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Captured Image", use_column_width=True)

        with col2:
            with st.spinner("🔍 Recognizing text..."):
                predictor.beam_width = beam_width
                result = predictor.predict(image, return_confidence=True)

            if result.get("confidence", 1.0) >= confidence_threshold:
                st.subheader("Recognition Result")
                st.markdown(
                    f'<div class="result-box">{result["text"]}</div>',
                    unsafe_allow_html=True
                )

                st.metric("Confidence", f"{result.get('confidence', 0):.2%}")
                st.metric("Processing Time", f"{result['processing_time']:.3f}s")
            else:
                st.warning("⚠️ Confidence below threshold")


def batch_mode(predictor, beam_width):
    """Batch processing mode."""
    st.header("📚 Batch Processing")

    uploaded_files = st.file_uploader(
        "Choose multiple images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload multiple images for batch processing"
    )

    if uploaded_files:
        st.info(f"📊 Processing {len(uploaded_files)} images...")

        if st.button("🚀 Start Batch Processing"):
            # Load images
            images = [Image.open(f) for f in uploaded_files]

            # Process
            start_time = time.time()
            with st.spinner("Processing..."):
                predictor.beam_width = beam_width
                results = predictor.predict_batch(images, batch_size=8)
            total_time = time.time() - start_time

            # Display results
            st.success(f"✅ Completed in {total_time:.2f}s")
            st.metric("Average Time per Image", f"{total_time/len(results):.3f}s")

            # Results table
            st.subheader("Results")

            for i, (file, result) in enumerate(zip(uploaded_files, results), 1):
                with st.expander(f"📄 Image {i}: {file.name}"):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(Image.open(file), use_column_width=True)

                    with col2:
                        st.markdown(
                            f'<div class="result-box">{result["text"]}</div>',
                            unsafe_allow_html=True
                        )

            # Download all results
            all_text = "\n\n".join([
                f"Image {i}: {file.name}\n{result['text']}"
                for i, (file, result) in enumerate(zip(uploaded_files, results), 1)
            ])

            st.download_button(
                label="📥 Download All Results",
                data=all_text,
                file_name="batch_ocr_results.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
