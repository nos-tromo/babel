import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import whisper  # type: ignore
from transformers import Pipeline, pipeline
from whisper.tokenizer import LANGUAGES  # type: ignore


class Babel:
    def __init__(self) -> None:
        self.device = self.get_device()
        self.dialect_model_id = os.getenv(
            "DIALECT_MODEL", "badrex/mms-300m-arabic-dialect-identifier"
        )
        self.whisper_model_id = os.getenv("WHISPER_MODEL", "turbo")

    @staticmethod
    def get_device() -> str:
        """
        Determine the appropriate device for model inference.

        Returns:
            str: The device to be used for inference.
        """
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    @staticmethod
    @st.cache_resource
    def load_whisper_model(model_id: str, device: str) -> Any:
        """
        Load the Whisper model for language detection.

        Args:
            model_id (str): The identifier of the model to load.
            device (str): The device to load the model onto.

        Returns:
            Any: The loaded Whisper model.
        """
        return whisper.load_model(name=model_id, device=device)

    @staticmethod
    @st.cache_resource
    def load_classifier(model_id: str, device: str) -> Pipeline:
        """
        Load the audio classification model.

        Args:
            model_id (str): The identifier of the model to load.
            device (str): The device to load the model onto.

        Returns:
            Pipeline: The loaded audio classification pipeline.
        """
        return pipeline(
            task="audio-classification",
            model=model_id,
            device=device,
        )

    @property
    def whisper_model(self) -> Any:
        """
        Get the Whisper model for language detection.

        Returns:
            Any: The loaded Whisper model.
        """
        return self.load_whisper_model(self.whisper_model_id, self.device)

    @property
    def classifier(self) -> Pipeline:
        """
        Get the audio classification pipeline.

        Returns:
            Pipeline: The loaded audio classification pipeline.
        """
        return self.load_classifier(self.dialect_model_id, self.device)

    def detect_language(self, audio_path: str) -> str:
        """
        Detect the language of the audio file using Whisper.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            str: Detected language code.

        Raises:
            ValueError: If language detection fails.
        """
        audio = whisper.load_audio(audio_path)
        mel = whisper.log_mel_spectrogram(
            audio, n_mels=self.whisper_model.dims.n_mels
        ).to(self.whisper_model.device)
        _, probs = self.whisper_model.detect_language(mel)
        if not probs:
            raise ValueError("Language detection failed; no probabilities returned.")
        return str(max(probs, key=probs.get))

    def predict_dialect(self, file: str) -> list[dict[str, Any]]:
        """
        Detect the dialect of the given audio file.

        Args:
            file (str): Path to the audio file.

        Returns:
            list[dict[str, Any]]: The classification predictions.
        """
        predictions = self.classifier(file)
        if not isinstance(predictions, list):
            if isinstance(predictions, (list, tuple)):
                predictions = list(predictions)
            else:
                predictions = [predictions]
        return [
            dict(prediction)
            for prediction in predictions
            if isinstance(prediction, dict)
        ]

    @staticmethod
    def get_language_name(language_code: str) -> str:
        """
        Get the full name of a language from its code.

        Args:
            language_code (str): The ISO 639-1 language code.

        Returns:
            str: The full name of the language.
        """
        return LANGUAGES.get(language_code, "Unknown").title()

    @staticmethod
    def save_uploaded_file(uploaded_file: Any) -> str:
        """
        Save uploaded file to a temporary file and return the path.

        Args:
            uploaded_file (Any): The uploaded file object from Streamlit.

        Returns:
            str: The path to the saved temporary file.
        """
        suffix = Path(uploaded_file.name).suffix
        if not suffix:
            suffix = ".mp3"  # Default to mp3 if no extension found

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name

    @staticmethod
    def slice_audio(input_path: str, start_time: str | float, duration: float) -> str:
        """
        Slice audio file using ffmpeg and convert to WAV.

        Args:
            input_path (str): Path to the input audio file.
            start_time (str | float): Start time in seconds or "hh:mm:ss" format.
            duration (float): Duration in seconds.

        Returns:
            str: Path to the sliced temporary audio file (WAV format).

        Raises:
            ValueError: If the sliced audio file is empty.
        """
        # Force .wav extension for the output to ensure compatibility
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            output_path = tmp_file.name

        # ffmpeg -ss <start> -t <duration> -i <input> -vn -acodec pcm_s16le -ar 16000 <output>
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-i",
            input_path,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # PCM 16-bit little endian
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # Mono
            output_path,
        ]

        subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if Path(output_path).stat().st_size == 0:
            raise ValueError(
                "Sliced audio file is empty. Check start time and duration."
            )

        return output_path
