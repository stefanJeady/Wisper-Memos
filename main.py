import os
import subprocess
import glob
from faster_whisper import WhisperModel

# Paths
input_file = "input.m4a"
output_dir = "chunks"
os.makedirs(output_dir, exist_ok=True)
output_transcription = "transcription.txt"

# Split MFA file into 30-second chunks
subprocess.run([
    "ffmpeg", "-i", input_file,
    "-f", "segment", "-segment_time", "30",
    "-c", "copy", f"{output_dir}/chunk_%03d.m4a"
])

# Initialize Faster Whisper with slowest, most exact options
model = WhisperModel("large-v3", device="cpu", compute_type="int8")

# Transcribe chunks
chunk_files = sorted(glob.glob(f"{output_dir}/chunk_*.m4a"))

with open(output_transcription, "w") as outfile:
    for idx, chunk_file in enumerate(chunk_files, 1):
        print(f"Transcribing chunk {idx}/{len(chunk_files)}: {chunk_file}")
        segments, _ = model.transcribe(chunk_file, beam_size=5, word_timestamps=True)
        for segment in segments:
            text = segment.text.strip()
            print(f"  >> {text}")
            outfile.write(text + " ")
        outfile.write("\n")

print(f"Transcription complete. Output written to {output_transcription}")