import os
import subprocess
from faster_whisper import WhisperModel
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")


def split_audio_ffmpeg(input_audio, chunk_length_sec=30):
    audio_dir = "audio_chunks"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    subprocess.run([
        "ffmpeg",
        "-i", input_audio,
        "-f", "segment",
        "-segment_time", str(chunk_length_sec),
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        os.path.join(audio_dir, "chunk_%04d.wav"),
        "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    chunks = sorted([
        os.path.join(audio_dir, file)
        for file in os.listdir(audio_dir)
        if file.endswith(".wav")
    ])

    print(f"Chunks created: {chunks}")

    return chunks


def transcribe_chunk(model, chunk_path):
    segments, _ = model.transcribe(chunk_path, beam_size=5, language="en")
    text = " ".join([segment.text.strip() for segment in segments])
    return text


def main(input_audio, output_text):
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    audio_chunks = split_audio_ffmpeg(input_audio)

    if not audio_chunks:
        print("No chunks created. Check your input audio and ffmpeg output.")
        return

    text_dir = "text_chunks"
    if not os.path.exists(text_dir):
        os.makedirs(text_dir)

    transcription_files = []

    for chunk in audio_chunks:
        print(f"Transcribing {chunk}...")
        text = transcribe_chunk(model, chunk)
        print(f"Transcription completed for {chunk}: {text}\n")

        txt_file = os.path.join(text_dir, os.path.basename(chunk).replace(".wav", ".txt"))
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)
        transcription_files.append(txt_file)

    with open(output_text, "w", encoding="utf-8") as outfile:
        for txt_file in transcription_files:
            with open(txt_file, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + " ")

    print(f"Transcription complete. Result saved to {output_text}")


if __name__ == "__main__":
    input_audio = "input.m4a"
    output_text = "final_transcription.txt"
    main(input_audio, output_text)