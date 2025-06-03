# Requires FFMPEG and Faster Whisper installed

import os
import subprocess
import glob
import time
import sys
import shutil
from faster_whisper import WhisperModel

# Settings
UseCuda = False
FileExtension = ".m4a"
FileName = "input" + FileExtension
Output_dir = "chunks"
output_transcription = "transcription.txt"

# Terminal color codes
GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Clear output directory if it exists, then create it
if os.path.exists(Output_dir):
    print(f"{ORANGE}Found existing chunks directory. Clearing it...{RESET}")
    shutil.rmtree(Output_dir)  

# Always create the directory (whether it existed before or not)
os.makedirs(Output_dir, exist_ok=True)

# Split audio file into 30-second chunks to prevent model collapse
subprocess.run([
    "ffmpeg", "-i", FileName,
    "-f", "segment", "-segment_time", "30",
    "-c", "copy", "-start_number", "1", 
    f"{Output_dir}/chunk_%03d" + FileExtension,
    "-loglevel", "error"
], stderr=subprocess.PIPE)  

# Initialize Faster Whisper
if not UseCuda:
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
else:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# Transcribe chunks
chunk_files = sorted(glob.glob(f"{Output_dir}/chunk_*" + FileExtension))
total_chunks = len(chunk_files)
start_time = time.time()
total_words = 0

try:
    with open(output_transcription, "w") as outfile:
        for idx, chunk_file in enumerate(chunk_files, 1):
            print(f"{GREEN}Chunk {RESET}{idx}/{total_chunks}{GREEN}: {RESET}{os.path.basename(chunk_file)}")
            
            segments, _ = model.transcribe(chunk_file, beam_size=5, word_timestamps=True)
            chunk_words = 0
            
            for segment in segments:
                text = segment.text.strip()
                words = text.split()
                chunk_words += len(words)
                total_words += len(words)
                print(f"  {BLUE}█{RESET} {text}")
                outfile.write(text + " ")
            outfile.write("\n\n")
            
            # Update progress
            percent_complete = idx / total_chunks * 100
            elapsed = time.time() - start_time
            
            words_per_minute = int(total_words / (elapsed / 60)) if elapsed > 0 else 0
            
            if idx > 0:
                avg_time_per_chunk = elapsed / idx
                remaining_chunks = total_chunks - idx
                estimated_remaining = avg_time_per_chunk * remaining_chunks
                
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining))
                
                print(f"{GREEN}Progress: {RESET}{percent_complete:.1f}%{GREEN} | Time: {RESET}{elapsed_str}{GREEN} elapsed / {RESET}{remaining_str}{GREEN} remaining | WPM: {RESET}{words_per_minute}")
except Exception as e:
    print(f"{ORANGE}An error occurred during transcription: {RESET}{str(e)}")

# Final progress update
total_time = time.time() - start_time
total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
words_per_minute = int(total_words / (total_time / 60)) if total_time > 0 else 0

print(f"\n{GREEN}Transcription complete. Output written to {RESET}{output_transcription}")
print(f"{GREEN}Total time: {RESET}{total_time_str}{GREEN} | Total words: {RESET}{total_words}{GREEN} | Words per minute: {RESET}{words_per_minute}")