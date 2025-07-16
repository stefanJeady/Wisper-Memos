import os
import subprocess
import tempfile
import logging
from faster_whisper import WhisperModel
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

logging.info(f"CUDA Available: {torch.cuda.is_available()}")
logging.info(f"cuDNN Version: {torch.backends.cudnn.version()}")


def split_audio_ffmpeg(input_audio, chunk_length_sec=30):
    audio_chunks = []
    temp_audio_dir = tempfile.mkdtemp(prefix="audio_chunks_")

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

    logging.info(f"Chunks created: {audio_chunks}")

    return audio_chunks


def transcribe_chunk(model, chunk_path):
    segments, _ = model.transcribe(chunk_path, beam_size=5, language="en")
    return " ".join(segment.text.strip() for segment in segments)


def main(input_audio, output_text):
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    audio_chunks = split_audio_ffmpeg(input_audio)

    if not audio_chunks:
        logging.error("No chunks created. Check your input audio and ffmpeg output.")
        return

    temp_text_dir = tempfile.mkdtemp(prefix="text_chunks_")
    transcription_files = []

    for chunk in audio_chunks:
        logging.info(f"Transcribing {chunk}...")
        text = transcribe_chunk(model, chunk)
        logging.info(f"Transcription completed for {chunk}: {text}")

        txt_file = os.path.join(temp_text_dir, os.path.basename(chunk).replace(".wav", ".txt"))
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)
        transcription_files.append(txt_file)

    with open(output_text, "w", encoding="utf-8") as outfile:
        for txt_file in transcription_files:
            with open(txt_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + " ")

    logging.info(f"Transcription complete. Result saved to {output_text}")


if __name__ == "__main__":
    input_audio = "input.m4a"
    output_text = "final_transcription.txt"
    main(input_audio, output_text)
