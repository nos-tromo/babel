import sys
import os
from unittest.mock import MagicMock, patch
import pytest

# Add the project root to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from babel.core import (
    get_language_name,
    set_device,
    save_uploaded_file,
    slice_audio,
    predict_dialect,
)


def test_get_language_name():
    assert get_language_name("en") == "English"
    assert get_language_name("ar") == "Arabic"
    assert get_language_name("xyz") == "Unknown"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_set_device_cuda(mock_mps, mock_cuda):
    mock_cuda.return_value = True
    assert set_device() == "cuda"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_set_device_mps(mock_mps, mock_cuda):
    mock_cuda.return_value = False
    mock_mps.return_value = True
    assert set_device() == "mps"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_set_device_cpu(mock_mps, mock_cuda):
    mock_cuda.return_value = False
    mock_mps.return_value = False
    assert set_device() == "cpu"


def test_save_uploaded_file():
    mock_file = MagicMock()
    mock_file.name = "test_audio.mp3"
    mock_file.getvalue.return_value = b"fake audio content"

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_audio.mp3"
        mock_temp.return_value.__enter__.return_value = mock_temp_file

        path = save_uploaded_file(mock_file)

        assert path == "/tmp/test_audio.mp3"
        mock_temp_file.write.assert_called_once_with(b"fake audio content")


@patch("subprocess.run")
@patch("pathlib.Path.stat")
def test_slice_audio(mock_stat, mock_run):
    # Mock file size check
    mock_stat_obj = MagicMock()
    mock_stat_obj.st_size = 1024  # Non-zero size
    mock_stat.return_value = mock_stat_obj

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/sliced.wav"
        mock_temp.return_value.__enter__.return_value = mock_temp_file

        output_path = slice_audio("input.mp3", 10.0, 5.0)

        assert output_path == "/tmp/sliced.wav"
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        command = args[0]
        assert command[0] == "ffmpeg"
        assert "-ss" in command
        assert "10.0" in command
        assert "-t" in command
        assert "5.0" in command


@patch("subprocess.run")
@patch("pathlib.Path.stat")
def test_slice_audio_empty_output(mock_stat, mock_run):
    # Mock file size check to return 0
    mock_stat_obj = MagicMock()
    mock_stat_obj.st_size = 0
    mock_stat.return_value = mock_stat_obj

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/sliced.wav"
        mock_temp.return_value.__enter__.return_value = mock_temp_file

        with pytest.raises(ValueError):
            slice_audio("input.mp3", 0, 5)


def test_predict_dialect():
    mock_classifier = MagicMock()
    # Case 1: List of dicts
    mock_classifier.return_value = [{"label": "EGY", "score": 0.9}]
    result = predict_dialect(mock_classifier, "file.wav")
    assert len(result) == 1
    assert result[0]["label"] == "EGY"

    # Case 2: Single dict (not in list, though pipeline usually returns list)
    mock_classifier.return_value = {"label": "LEV", "score": 0.8}
    result = predict_dialect(mock_classifier, "file.wav")
    assert len(result) == 1
    assert result[0]["label"] == "LEV"
