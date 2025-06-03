# Requires FFMPEG and Faster Whisper installed

import os
import subprocess
import glob
import time
import shutil
import threading
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor, as_completed

# Settings
UseCuda = False
FileExtension = ".m4a"
FileName = "input" + FileExtension
Output_dir = "chunk_audio"
Text_output_dir = "chunk_text" 
output_transcription = "transcription.txt"
num_threads = max(1, os.cpu_count() - 8)

# Terminal color codes
GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
BLUE = "\033[94m"
RESET = "\033[0m"

# Clear output directories if they exist, then create them
for directory in [Output_dir, Text_output_dir]:
    if os.path.exists(directory):
        print(f"{ORANGE}Found existing {directory} directory. Clearing it...{RESET}")
        shutil.rmtree(directory)  
    
    # Always create the directory (whether it existed before or not)
    os.makedirs(directory, exist_ok=True)

# Split audio file into 15-second chunks to prevent model collapse
subprocess.run([
    "ffmpeg", "-i", FileName,
    "-f", "segment", "-segment_time", "15",
    "-c", "copy", "-start_number", "0", 
    f"{Output_dir}/chunk_%03d" + FileExtension,
    "-loglevel", "error"
], stderr=subprocess.PIPE)  

# Thread-safe counters and locks
transcription_lock = threading.Lock()
progress_lock = threading.Lock()
total_words = 0
processed_chunks = 0

# Function to transcribe a single chunk
def transcribe_chunk(chunk_data):
    global total_words, processed_chunks
    chunk_file, idx, total_chunks, model = chunk_data
    
    chunk_words = 0
    result_text = ""
    chunk_output_file = f"{Text_output_dir}/chunk_{idx:03d}.txt"
    
    try:
        with progress_lock:
            print(f"{GREEN}Starting Chunk {RESET}{idx}/{total_chunks}")
        
        segments, _ = model.transcribe(
            chunk_file, 
            beam_size=15, 
            word_timestamps=False, 
            temperature=0,
            condition_on_previous_text=False
        )
        
        for segment in segments:
            text = segment.text.strip()
            words = text.split()
            chunk_words += len(words)
            # Add each segment as a new line instead of joining with spaces
            result_text += text + "\n"
            
            with progress_lock:
                print(f"  {BLUE}█{RESET} {text}")
        
        result_text += "\n"  # Add an extra line break between chunks
        
        # Save this chunk's transcription to its own file in the text folder
        with open(chunk_output_file, "w") as chunk_file:
            chunk_file.write(result_text)
        
        with progress_lock:
            processed_chunks += 1
            total_words += chunk_words
            
            # Update progress
            percent_complete = processed_chunks / total_chunks * 100
            elapsed = time.time() - start_time
            
            words_per_minute = int(total_words / (elapsed / 60)) if elapsed > 0 else 0
            
            if processed_chunks > 0:
                avg_time_per_chunk = elapsed / processed_chunks
                remaining_chunks = total_chunks - processed_chunks
                estimated_remaining = avg_time_per_chunk * remaining_chunks
                
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                remaining_str = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining))
                
                print(f"{GREEN}Progress: {RESET}{percent_complete:.1f}%{GREEN} | Time: {RESET}{elapsed_str}{GREEN} elapsed / {RESET}{remaining_str}{GREEN} remaining | WPM: {RESET}{words_per_minute}")
        
        return idx, chunk_output_file
    except Exception as e:
        with progress_lock:
            print(f"{ORANGE}Error processing chunk {idx}: {RESET}{str(e)}")
        # Save error information to the chunk file
        with open(chunk_output_file, "w") as chunk_file:
            chunk_file.write(f"[Transcription error for chunk {idx}]\n\n")
        return idx, chunk_output_file

# Create model instances for each thread
def create_model():
    if not UseCuda:
        return WhisperModel("large-v3", device="cpu", compute_type="int8")
    else:
        return WhisperModel("large-v3", device="cuda", compute_type="float16")

# Main execution
try:
    # Get all chunk files and sort them
    chunk_files = sorted(glob.glob(f"{Output_dir}/chunk_*" + FileExtension))
    total_chunks = len(chunk_files)
    
    print(f"{GREEN}Found {RESET}{total_chunks} {GREEN}chunks to process with {RESET}{num_threads} {GREEN}threads")
    start_time = time.time()
    
    # Prepare task data
    tasks = []
    results = {}  # Dictionary to store results in order
    
    # Initialize the thread pool
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create models for each thread (will be reused)
        models = [create_model() for _ in range(num_threads)]
        
        # Submit all tasks
        futures = {}
        for i, chunk_file in enumerate(chunk_files, 1):
            model = models[i % num_threads]  # Assign model in round-robin fashion
            future = executor.submit(transcribe_chunk, (chunk_file, i, total_chunks, model))
            futures[future] = i
        
        # Process results as they complete
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text
    
    # Write results to file in the correct order
    with open(output_transcription, "w") as outfile:
        for i in range(1, total_chunks + 1):
            if i in results:
                outfile.write(results[i])
    
except Exception as e:
    print(f"{ORANGE}An error occurred during transcription: {RESET}{str(e)}")

# Final progress update
total_time = time.time() - start_time
total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
words_per_minute = int(total_words / (total_time / 60)) if total_time > 0 else 0

print(f"\n{GREEN}Transcription complete. Output written to {RESET}{output_transcription}")
print(f"{GREEN}Total time: {RESET}{total_time_str}{GREEN} | Total words: {RESET}{total_words}{GREEN} | Words per minute: {RESET}{words_per_minute}")
print(f"{GREEN}Used {num_threads} parallel threads for processing{RESET}")