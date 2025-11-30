import sys
import os
from unittest.mock import MagicMock, patch
import pytest

# Add the project root to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from babel.core import Babel


def test_get_language_name() -> None:
    """
    Test the get_language_name method.
    """
    assert Babel.get_language_name("en") == "English"
    assert Babel.get_language_name("ar") == "Arabic"
    assert Babel.get_language_name("xyz") == "Unknown"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_get_device_cuda(mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
    """
    Test the get_device method for CUDA.

    Args:
        mock_mps (MagicMock): Mock for MPS availability check.
        mock_cuda (MagicMock): Mock for CUDA availability check
    """
    mock_cuda.return_value = True
    assert Babel.get_device() == "cuda"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_get_device_mps(mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
    """
    Test the get_device method for MPS.

    Args:
        mock_mps (MagicMock): Mock for MPS availability check.
        mock_cuda (MagicMock): Mock for CUDA availability check.
    """
    mock_cuda.return_value = False
    mock_mps.return_value = True
    assert Babel.get_device() == "mps"


@patch("torch.cuda.is_available")
@patch("torch.backends.mps.is_available")
def test_get_device_cpu(mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
    """
    Test the get_device method for CPU.

    Args:
        mock_mps (MagicMock): Mock for MPS availability check.
        mock_cuda (MagicMock): Mock for CUDA availability check.
    """
    mock_cuda.return_value = False
    mock_mps.return_value = False
    assert Babel.get_device() == "cpu"


def test_save_uploaded_file() -> None:
    """
    Test the save_uploaded_file method.
    """
    mock_file = MagicMock()
    mock_file.name = "test_audio.mp3"
    mock_file.getvalue.return_value = b"fake audio content"

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/test_audio.mp3"
        mock_temp.return_value.__enter__.return_value = mock_temp_file

        path = Babel.save_uploaded_file(mock_file)

        assert path == "/tmp/test_audio.mp3"
        mock_temp_file.write.assert_called_once_with(b"fake audio content")


@patch("subprocess.run")
@patch("pathlib.Path.stat")
def test_slice_audio(mock_stat: MagicMock, mock_run: MagicMock) -> None:
    """
    Test the slice_audio method.

    Args:
        mock_stat (MagicMock): Mock for file stat.
        mock_run (MagicMock): Mock for subprocess.run.
    """
    # Mock file size check
    mock_stat_obj = MagicMock()
    mock_stat_obj.st_size = 1024  # Non-zero size
    mock_stat.return_value = mock_stat_obj

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/sliced.wav"
        mock_temp.return_value.__enter__.return_value = mock_temp_file

        output_path = Babel.slice_audio("input.mp3", 10.0, 5.0)

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
def test_slice_audio_empty_output(mock_stat: MagicMock, mock_run: MagicMock) -> None:
    """
    Test the slice_audio method for empty output.

    Args:
        mock_stat (MagicMock): Mock for file stat.
        mock_run (MagicMock): Mock for subprocess.run.
    """
    # Mock file size check to return 0
    mock_stat_obj = MagicMock()
    mock_stat_obj.st_size = 0
    mock_stat.return_value = mock_stat_obj

    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/sliced.wav"
        mock_temp.return_value.__enter__.return_value = mock_temp_file

        with pytest.raises(ValueError):
            Babel.slice_audio("input.mp3", 0, 5)


def test_predict_dialect():
    """
    Test the predict_dialect method.
    """
    # Mock the Babel instance and its classifier
    with (
        patch("babel.core.Babel.load_whisper_model"),
        patch("babel.core.Babel.load_classifier") as mock_load_classifier,
        patch("babel.core.Babel.get_device"),
    ):
        mock_classifier = MagicMock()
        mock_load_classifier.return_value = mock_classifier

        babel = Babel()

        # Case 1: List of dicts
        mock_classifier.return_value = [{"label": "EGY", "score": 0.9}]
        result = babel.predict_dialect("file.wav")
        assert len(result) == 1
        assert result[0]["label"] == "EGY"

        # Case 2: Single dict (not in list, though pipeline usually returns list)
        mock_classifier.return_value = {"label": "LEV", "score": 0.8}
        result = babel.predict_dialect("file.wav")
        assert len(result) == 1
        assert result[0]["label"] == "LEV"
