"""
Shared fixtures and configuration for ML component tests.
Following the albumentations testing pattern.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
import pytest
import torch
import torchaudio
from attr import define

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test data constants
TEST_AUDIO_SAMPLE_RATE = 16000
TEST_AUDIO_DURATION = 2.0  # seconds
TEST_TEXT = "This is a test sentence for speech processing."
TEST_LANGUAGES = ["en", "es", "fr", "de"]


@pytest.fixture
def mock_device():
    """Mock device for testing."""
    return "cpu"


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    num_samples = int(TEST_AUDIO_SAMPLE_RATE * TEST_AUDIO_DURATION)
    # Generate a simple sine wave
    t = np.linspace(0, TEST_AUDIO_DURATION, num_samples)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440Hz sine wave
    return audio


@pytest.fixture
def sample_audio_file(sample_audio_data, tmp_path):
    """Create a temporary audio file for testing."""
    audio_file = tmp_path / "test_audio.wav"
    audio_tensor = torch.from_numpy(sample_audio_data).unsqueeze(0)
    torchaudio.save(str(audio_file), audio_tensor, TEST_AUDIO_SAMPLE_RATE)
    return str(audio_file)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return TEST_TEXT


@pytest.fixture
def sample_texts():
    """List of sample texts for batch testing."""
    return [
        "First test sentence.",
        "Second test sentence with more words.",
        "Short text.",
        "A longer test sentence that contains multiple words and should be suitable for testing various text processing scenarios."
    ]


@pytest.fixture
def mock_file_repository():
    """Mock file repository for testing."""
    repo = Mock()
    repo.directory = tempfile.mkdtemp()
    repo.get_file = Mock(side_effect=lambda name: Mock(file_path=os.path.join(repo.directory, name)))
    repo.save_file = Mock(return_value=Mock())
    return repo


@pytest.fixture(params=TEST_LANGUAGES)
def language(request):
    """Parametrized language fixture."""
    return request.param


@pytest.fixture
def mock_whisper_model():
    """Mock WhisperX model for ASR testing."""
    model = Mock()
    model.transcribe = Mock(return_value={
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": TEST_TEXT,
                "words": [
                    {"word": "This", "start": 0.0, "end": 0.3},
                    {"word": "is", "start": 0.3, "end": 0.4},
                    {"word": "a", "start": 0.4, "end": 0.5},
                    {"word": "test", "start": 0.5, "end": 0.8},
                ]
            }
        ],
        "language": "en"
    })
    return model


@pytest.fixture
def mock_diarization_model():
    """Mock diarization model for ASR testing."""
    model = Mock()
    model.return_value = {
        "segments": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    }
    return model


@pytest.fixture
def mock_llm_model():
    """Mock LLM model for translation testing."""
    model = Mock()
    model.create_chat_completion = Mock(return_value={
        "choices": [
            {
                "message": {"content": "Translated text", "role": "assistant"}
            }
        ]
    })
    return model


@pytest.fixture
def mock_tts_model():
    """Mock TTS model for speech synthesis testing."""
    model = Mock()
    model.tts_to_file = Mock()
    model.to = Mock(return_value=model)  # For device switching
    return model


def convert_audio_format(audio_data, target_format="mono"):
    """Helper function to convert audio format for testing."""
    if target_format == "mono" and len(audio_data.shape) > 1:
        return np.mean(audio_data, axis=1)
    return audio_data


def create_mock_segment(start=0.0, end=1.0, text="test", speaker="SPEAKER_00"):
    """Helper function to create mock segment objects."""
    segment = Mock()
    segment.start = start
    segment.end = end
    segment.text = text
    segment.speaker = speaker
    segment.words = []
    return segment


@pytest.fixture
def sample_segments():
    """Create sample segments for testing."""
    return [
        create_mock_segment(0.0, 1.0, "First segment", "SPEAKER_00"),
        create_mock_segment(1.0, 2.5, "Second segment", "SPEAKER_01"),
        create_mock_segment(2.5, 4.0, "Third segment", "SPEAKER_00"),
    ]


@pytest.fixture
def sample_video_path() -> str:
    """Path to a small synthetic test video in tests/fixtures/media/.

    If the file hasn't been generated yet, creates it on the fly with ffmpeg.
    """
    media_dir = Path(__file__).parent / "fixtures" / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    video = media_dir / "test_video.mp4"
    if not video.exists():
        import subprocess
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", "testsrc=duration=3:size=320x180:rate=12",
                "-f", "lavfi",
                "-i", "sine=frequency=440:duration=3",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                "-c:a", "aac", "-shortest",
                str(video),
            ],
            capture_output=True, timeout=20,
        )
    return str(video)


@pytest.fixture
def sample_video_url() -> str:
    """Public S3 URL of a test video for full-pipeline integration tests."""
    return "https://storage.yandexcloud.net/langswap-public/ru_source/1.mp4" 