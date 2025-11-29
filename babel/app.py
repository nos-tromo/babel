import sys
from pathlib import Path
from typing import Any

import streamlit as st
from streamlit.web import cli as st_cli

from babel.core import (
    detect_language,
    get_language_name,
    load_classifier,
    load_whisper_model,
    predict_dialect,
    save_uploaded_file,
    set_device,
    slice_audio,
)


def display_results(predictions: list[dict[str, Any]]) -> None:
    """
    Display the classification results.

    Args:
        predictions (list[dict[str, Any]]): List of prediction dictionaries.
    """
    st.subheader("Results")
    st.caption("The model's predictions are listed below, sorted by confidence.")
    for pred in predictions:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{pred['label']}**")
        with col2:
            st.progress(pred["score"])
            st.caption(f"Confidence: {pred['score']:.4f}")


def main() -> None:
    """
    Main function to run the Streamlit app.

    Args:
        model_id (str): The identifier of the model to load.
    """
    st.set_page_config(page_title="Babel", layout="wide")
    st.title("Babel")

    st.markdown(
        """
        This application uses deep learning models to identify Arabic dialects from audio recordings.
        Upload an audio file, and the model will predict the most likely dialect with a confidence score.
        """
    )
    st.caption("Supported formats: MP3, M4A, WAV, OGG, FLAC, MP4, MKV, AVI, MOV, WEBM")

    with st.spinner("Initializing models..."):
        device = set_device()
        whisper_model = load_whisper_model(device)
        classifier = load_classifier(device)

    if not classifier:
        st.error("Failed to load classifier. Please try again.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose an audio or video file",
        type=["mp3", "m4a", "wav", "ogg", "flac", "mp4", "mkv", "avi", "mov", "webm"],
    )

    st.markdown(
            """
            <hr style="margin-top:2rem;margin-bottom:1rem;">
            <p style="text-align:center;">
                🔗 <a href="https://github.com/nos-tromo/babel" target="_blank">GitHub</a>
            </p>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        st.divider()
        st.subheader("Analysis Settings")
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.text_input("Start Time (seconds or hh:mm:ss)", value="0")
        with col2:
            duration = st.number_input(
                "Duration (seconds)", min_value=1.0, value=30.0, step=1.0
            )

        if st.button("Analyze Segment"):
            tmp_file_path = save_uploaded_file(uploaded_file)
            sliced_file_path = None

            try:
                with st.spinner("Slicing audio..."):
                    sliced_file_path = slice_audio(tmp_file_path, start_time, duration)
                    if not sliced_file_path:
                        st.error(
                            "Failed to slice audio. Please check the input parameters."
                        )
                        st.stop()

                with st.spinner("Detecting language..."):
                    language = detect_language(whisper_model, sliced_file_path)
                    if not language:
                        st.error("Failed to detect language. Please try again.")
                        st.stop()

                if language != "ar":
                    language_name = get_language_name(language)
                    st.warning(
                        f"Detected language is {language_name.capitalize()}, not Arabic. Please upload an Arabic audio file."
                    )
                    st.stop()

                with st.spinner("Analyzing..."):
                    predictions = predict_dialect(classifier, sliced_file_path)
                    display_results(predictions)
                    if not predictions:
                        st.error("No predictions were made. Please try again.")
                        st.stop()

            finally:
                # Cleanup sliced file
                if sliced_file_path:
                    Path(sliced_file_path).unlink(missing_ok=True)
                # Cleanup original temp file
                if tmp_file_path:
                    Path(tmp_file_path).unlink(missing_ok=True)


# ---- Streamlit CLI wrapper ----------------------------------------------- #
def run() -> None:
    """
    CLI entry point for the Streamlit app. This function is used to run the app from the command
    line. It sets up the command line arguments as if the user typed them. For example: `streamlit
    run app.py <any extra args>`.
    """
    sys.argv = ["streamlit", "run", __file__] + sys.argv[1:]
    sys.exit(st_cli.main())


if __name__ == "__main__":
    main()
