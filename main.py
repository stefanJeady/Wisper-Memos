"""Audio Transcription Tool using Faster Whisper - See README.md for full documentation."""

import os
import subprocess
import tempfile
import logging
from faster_whisper import WhisperModel
import torch
import platform

# Configure logging format for clear output tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Platform detection for model configuration
is_mac = platform.system() == "Darwin"

# Log system information for debugging
logging.info(f"Running on {'Mac (CPU mode)' if is_mac else 'CUDA mode'}")
logging.info(f"CUDA Available: {torch.cuda.is_available()}")
logging.info(f"cuDNN Version: {torch.backends.cudnn.version()}")

# ANSI color codes for enhanced terminal output
BLUE_TEXT = "\033[94m"
GREEN_TEXT = "\033[92m"
RESET_TEXT = "\033[0m"


def get_audio_length(input_audio):
    """Get audio duration in seconds using FFprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                input_audio
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        # Check if we got valid output
        duration_str = result.stdout.strip()
        if not duration_str:
            raise ValueError(f"FFprobe returned empty duration for file: {input_audio}")
        
        return float(duration_str)
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "Unknown FFmpeg error"
        raise RuntimeError(f"FFmpeg failed: {error_msg}. Make sure FFmpeg is installed and the audio file exists.")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
    except ValueError as e:
        if "could not convert string to float" in str(e):
            raise ValueError(f"Invalid duration format from FFmpeg for file: {input_audio}")
        else:
            raise


def split_audio_ffmpeg(input_audio, temp_audio_dir, chunk_length_sec=30):
    """Split audio into chunks using FFmpeg for optimal Whisper processing."""
    audio_chunks = []

    subprocess.run([
        "ffmpeg",
        "-i", input_audio,
        "-f", "segment",
        "-segment_time", str(chunk_length_sec),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        os.path.join(temp_audio_dir, "chunk_%04d.wav"),
        "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for file in sorted(os.listdir(temp_audio_dir)):
        if file.endswith(".wav"):
            audio_chunks.append(os.path.join(temp_audio_dir, file))

    logging.info(f"{GREEN_TEXT}Chunks created: {len(audio_chunks)}{RESET_TEXT}")

    return audio_chunks


def transcribe_chunk(model, chunk_path):
    """Transcribe a single audio chunk using Whisper with beam search."""
    segments, _ = model.transcribe(chunk_path, beam_size=5, language="en")
    return " ".join(segment.text.strip() for segment in segments)


def main(input_audio, output_text):
    """Main function to transcribe audio file to text using chunked processing."""
    
    # Check if input file exists
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio file not found: {input_audio}")
    
    audio_length_sec = get_audio_length(input_audio)
    logging.info(f"{GREEN_TEXT}Audio length: {audio_length_sec:.2f} seconds. This may take some time. Starting to split audio into chunks for best Whisper performance.{RESET_TEXT}")

    if is_mac:
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    else:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    with tempfile.TemporaryDirectory(prefix="audio_chunks_") as temp_audio_dir, \
            tempfile.TemporaryDirectory(prefix="text_chunks_") as temp_text_dir:

        audio_chunks = split_audio_ffmpeg(input_audio, temp_audio_dir)

        if not audio_chunks:
            logging.error("No chunks created. Check your input audio and ffmpeg output.")
            return

        transcription_files = []
        total_chunks = len(audio_chunks)

        for idx, chunk in enumerate(audio_chunks, 1):
            logging.info(f"Transcribing {chunk}...")
            text = transcribe_chunk(model, chunk)
            logging.info(f"Transcription completed for {chunk}: {BLUE_TEXT}{text}{RESET_TEXT}")
            logging.info(f"{GREEN_TEXT}Chunk {idx} of {total_chunks} complete{RESET_TEXT}")

            txt_file = os.path.join(temp_text_dir, os.path.basename(chunk).replace(".wav", ".txt"))
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(text)
            transcription_files.append(txt_file)

        with open(output_text, "w", encoding="utf-8") as outfile:
            for txt_file in transcription_files:
                with open(txt_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read() + " ")

    logging.info(f"{GREEN_TEXT}Transcription complete. Result saved to {output_text}{RESET_TEXT}")


if __name__ == "__main__":
    # Configuration for the transcription process
    # These paths can be modified to match your specific files
    input_audio = "input.m4a"  # Path to the audio file to transcribe
    output_text = "final_transcription.txt"  # Path where transcription will be saved
    
    try:
        # Start the transcription process
        main(input_audio, output_text)
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        print(f"\n{BLUE_TEXT}Please ensure:{RESET_TEXT}")
        print(f"1. Your audio file '{input_audio}' exists in the current directory")
        print(f"2. FFmpeg is installed and available in your PATH")
        print(f"3. The audio file format is supported by FFmpeg")
        exit(1)