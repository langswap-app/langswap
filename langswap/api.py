"""Public API for langswap package."""
import uuid
import time
from contextlib import contextmanager
from dotenv import load_dotenv
import boto3
import os
import logging

from langswap.translation_pipeline import VideoTranslationPipeline, ChangeManager
from langswap.pipeline_models.models import TraslationUpdate
from langswap.pipeline_models.models import TranslationPipelineConfig, load_config_from_json
from langswap.file_repository import RemoteFile, RemoteFileRepository, download_s3_directory

load_dotenv()

# Set the global logging level to WARNING to hide INFO messages
logging.disable(logging.DEBUG)

BASE_DIR = "data"

def init_s3_client():
    s3 = boto3.client('s3',
        endpoint_url            = 'https://storage.yandexcloud.net',
        aws_access_key_id       = os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key   = os.environ['AWS_SECRET_ACCESS_KEY']
    )
    return s3

def get_file(repo, s3_url):
    remote_file = RemoteFile(
            s3_url = s3_url,
            name="source.mp4"
    )
    remote_file = repo.materialize_file(remote_file)
    return remote_file.file_path

def process_translation(input, progress_callback=None):
    """
    Core translation processing function that can be used both by RunPod handler
    and local testing
    
    Args:
        input: Dictionary with translation parameters
        progress_callback: Optional function to report progress
    """
    # Helper function for progress updates
    def update_progress(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)

    # Stage-level wall-clock timing.  The init-vs-inference split was previously
    # only inferred from logs; this records it directly so optimization is
    # data-driven.  Times are accumulated and printed as a summary at the end.
    stage_times = {}

    @contextmanager
    def timed(stage):
        t0 = time.perf_counter()
        update_progress(f"[timing] {stage}: start")
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            stage_times[stage] = dt
            update_progress(f"[timing] {stage}: {dt:.1f}s")

    assert 'target_language' in input.keys(), "target language is missing from input"
    assert 'tts_engine' in input.keys(), "tts engine is missing from input"

    # First progress update - Initialization
    update_progress("0% Initializing translation pipeline")

    s3_client = init_s3_client()

    public_id = input.get('public_id', str(uuid.uuid4()))
    repo = RemoteFileRepository(public_id, BASE_DIR, s3_client)
    with timed("download_input"):
        file_path = get_file(repo, input.get("s3_video_url"))
    
    tts_engine = input.get("tts_engine", "omnivoice")

    config = TranslationPipelineConfig(
        source_lang=input.get('source_language', None),
        target_lang=input.get('target_language'),
        name=public_id,
        public_id=public_id,
        num_speakers=input.get('count_speakers', None),
        source_video_path=file_path,
        base_dir=BASE_DIR,
        device='cuda',
        tts_model=tts_engine,
        dubbing_algo=input.get("dubbing_algo", "speedup"),
        watermark=input.get("watermark", True),
        skip_diarization=input.get("skip_diarization", False),
        # Default to the VAD backend: faster-whisper + Silero VAD segmentation,
        # no forced aligner and no per-language model — lightest and fastest, and
        # VAD places segment boundaries as tightly as the Qwen aligner.  Override
        # per-job with "asr_backend" ("qwen_onnx" / "qwen" / "whisperx") when a
        # forced aligner or the larger ASR is needed (e.g. proper-noun accuracy).
        asr_backend=input.get("asr_backend", "vad"),
        translation_backend=input.get("translation_backend", "llamacpp"),
    )
    
    with timed("pipeline_init"):
        pipeline = VideoTranslationPipeline(config=config, file_repository=repo)

    # Second progress update - Transcription
    update_progress("20% Starting transcription (Speech-to-Text)")
    with timed("asr"):
        pipeline._generate_asr()

    # Third progress update - Translation
    update_progress("40% Starting translation")
    with timed("translation"):
        pipeline._generate_translation()

    # Fourth progress update - Text-to-Speech
    update_progress("60% Starting Text-to-Speech synthesis")
    with timed("tts"):
        pipeline._generate_speech()

    # Fifth progress update - Audio separation and merging
    update_progress("80% Starting audio separation and enhancement")
    with timed("merge"):
        video_translation = pipeline._merge(pipeline.config.dubbing_algo)

    # Generate VTT files
    update_progress("95% Generating subtitle files")
    with timed("vtt"):
        source_vtt, translated_vtt = pipeline.generate_vtt_files()

    # Final progress update
    total = sum(stage_times.values())
    summary = " | ".join(f"{k}={v:.1f}s" for k, v in stage_times.items())
    update_progress(f"[timing] TOTAL={total:.1f}s :: {summary}")
    update_progress("100% Translation pipeline completed successfully")

    result_video = video_translation.processed_video
    
    # Return URLs for both the video and the VTT files
    return {
        's3_result_video_url': f'{result_video.s3_url}',
        's3_source_transcript_url': f'{source_vtt.s3_url}',
        's3_translated_transcript_url': f'{translated_vtt.s3_url}'
    }

def process_update_translation(input, progress_callback=None):
    """
    Core translation processing function that can be used both by RunPod handler
    and local testing
    
    Args:
        input: Dictionary with translation parameters
        progress_callback: Optional function to report progress
    """
    # Helper function for progress updates
    def update_progress(message):
        if progress_callback:
            progress_callback(message)
        else:
            print(message)
    
    public_id = input.get('public_id')
    s3_video_url = input.get("s3_video_url")
    update_translation_collection = input.get("update_translation")
    
    # First progress update - Initialization
    update_progress("0% Initializing translation pipeline")

    s3_client = init_s3_client()
    repo = RemoteFileRepository(public_id, BASE_DIR, s3_client)
    get_file(repo, s3_video_url)
    bucket = os.getenv('BUCKET')
    download_s3_directory(s3_client, bucket, f"{BASE_DIR}/{public_id}", f"{BASE_DIR}/{public_id}")
    config_file = repo.get_file("config.json")
    config = load_config_from_json(config_file.file_path)
    
    pipeline = VideoTranslationPipeline(config=config, file_repository=repo)
    # Second progress update - Transcription
    update_progress("60% Loading cache")
    video_translation = pipeline.translate_video()

  
    update_progress("60% Initializing change manager")
    change_menager = ChangeManager(pipeline, video_translation)
    
    video_translation = change_menager.apply_update_translations(TraslationUpdate.from_pairs(update_translation_collection))
    update_progress("90% Apply changes")
    update_progress("95% Generating subtitle files")
    pipeline.video_translation = video_translation
    source_vtt, translated_vtt = pipeline.generate_vtt_files()
    
    result_video = video_translation.processed_video
    
    
    # Return URLs for both the video and the VTT files
    return {
        's3_result_video_url': f'{result_video.s3_url}',
        's3_source_transcript_url': f'{source_vtt.s3_url}',
        's3_translated_transcript_url': f'{translated_vtt.s3_url}'
    }


def test_video_translation_local(input_file="tests/fixtures/test_input.json"):
    """
    This function builds a sample event and calls the handler.
    Adjust the values for testing based on your environment.
    """
    import json
    with open(input_file, "r") as f:
        test_event = json.load(f)
    try:
        result = process_translation(test_event['input'])
        print("Translation Test Successful:")
        print(result)
        return result
    except Exception as e:
        print("Translation Test Failed:")
        print(e)
        raise

if __name__ == "__main__":
    test_video_translation_local()
