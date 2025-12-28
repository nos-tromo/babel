import sys
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger
from streamlit.runtime import exists
from streamlit.web import cli as st_cli

from babel.core import Babel
from babel.utils.logging_cfg import setup_logging
from babel.utils.env_cfg import set_offline_env

setup_logging()
set_offline_env()


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


@st.cache_resource
def get_babel() -> Babel:
    """
    Get the Babel instance.

    Returns:
        Babel: The Babel instance.
    """
    return Babel()


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    logger.info("Starting Babel Streamlit app")
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
        babel = get_babel()
        # Access properties to trigger loading within the spinner context
        if not hasattr(babel, "classifier") or not hasattr(babel, "whisper_model"):
            logger.error("Failed to load models")
            st.error("Failed to load models. Please try again.")
            st.stop()
        logger.info("Models initialized successfully")

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
        logger.info(f"File uploaded: {uploaded_file.name}")
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
            logger.info(
                f"Starting analysis for {uploaded_file.name} (start={start_time}, duration={duration})"
            )
            tmp_file_path = babel.save_uploaded_file(uploaded_file)
            sliced_file_path = None

            try:
                with st.spinner("Slicing audio..."):
                    sliced_file_path = babel.slice_audio(
                        tmp_file_path, start_time, duration
                    )
                    if not sliced_file_path:
                        logger.error("Audio slicing failed")
                        st.error(
                            "Failed to slice audio. Please check the input parameters."
                        )
                        st.stop()

                with st.spinner("Detecting language..."):
                    language = babel.detect_language(sliced_file_path)
                    if not language:
                        logger.error("Language detection failed")
                        st.error("Failed to detect language. Please try again.")
                        st.stop()
                    logger.info(f"Detected language: {language}")

                if language != "ar":
                    language_name = babel.get_language_name(language)
                    logger.warning(f"Non-Arabic language detected: {language_name}")
                    st.warning(
                        f"Detected language is {language_name.capitalize()}, not Arabic. Please upload an Arabic audio file."
                    )
                    st.stop()

                with st.spinner("Analyzing..."):
                    predictions = babel.predict_dialect(sliced_file_path)
                    display_results(predictions)
                    if not predictions:
                        logger.error("Dialect prediction returned no results")
                        st.error("No predictions were made. Please try again.")
                        st.stop()
                    logger.info(
                        f"Analysis complete. Top prediction: {predictions[0]['label']}"
                    )

            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")
                st.error(f"An unexpected error occurred: {e}")
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
    app_path = Path(__file__).resolve()
    sys.argv = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(st_cli.main())


if __name__ == "__main__":
    try:
        if exists():
            main()
        else:
            run()
    except ImportError as e:
        logger.exception(f"Failed to run the Streamlit app: {e}")
        run()
