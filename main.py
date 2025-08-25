"""Audio Transcription Tool using Faster Whisper - See README.md for full documentation."""

import os
import subprocess
import tempfile
import logging
import time
import sys
from faster_whisper import WhisperModel
import torch
import platform

# Disable default logging to create custom output
logging.basicConfig(level=logging.CRITICAL)

# Platform detection for model configuration
is_mac = platform.system() == "Darwin"

# Enhanced ANSI color codes and styles
class Colors:
    # Text colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_BLUE = "\033[44m"
    
    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    
    # Reset
    RESET = "\033[0m"
    
    # Special effects
    RAINBOW = [RED, YELLOW, GREEN, CYAN, BLUE, MAGENTA]


def print_banner():
    """Print a sexy startup banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                    WHISPER TRANSCRIPTION                     ║
║                     AI-Powered Audio Magic                   ║
╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}
    """
    print(banner)


def print_system_info():
    """Print system information"""
    print(f"{Colors.BOLD}{Colors.BLUE}SYSTEM INFORMATION{Colors.RESET}")
    print(f"{'─' * 50}")
    
    platform_info = "Mac (CPU mode)" if is_mac else "Windows/Linux (CUDA mode)"
    
    # Check CTranslate2 CUDA instead of PyTorch CUDA
    try:
        from faster_whisper import WhisperModel
        # Try to create a small model on CUDA
        test_model = WhisperModel("tiny", device="cuda")
        cuda_status = "Available (CTranslate2)"
    except:
        cuda_status = "Not Available"
    
    print(f"{Colors.CYAN}Platform:{Colors.RESET} {platform_info}")
    print(f"{Colors.CYAN}CUDA:{Colors.RESET} {cuda_status}")
    print()


def print_progress_bar(current, total, prefix="Progress", bar_length=40):
    """Print a colorful progress bar that updates in place."""
    percent = current / total
    filled_length = int(bar_length * percent)
    
    # Create green progress bar
    bar = f"{Colors.GREEN}{'█' * filled_length}{Colors.RESET}"
    bar += f"{Colors.DIM}{'░' * (bar_length - filled_length)}{Colors.RESET}"
    
    # Move up 4 lines to position above chunk status, clear and write progress
    sys.stdout.write(f"\033[4A")  # Move up 4 lines
    sys.stdout.write(f"\r{' ' * 100}\r")  # Clear line
    sys.stdout.write(f"{Colors.BOLD}{prefix}:{Colors.RESET} [{bar}] {Colors.GREEN}{percent:.1%}{Colors.RESET} ({current}/{total})")
    sys.stdout.write(f"\033[4B")  # Move back down 4 lines
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write(f"\033[4A")  # Move up to progress line
        sys.stdout.write(f"\r{Colors.BOLD}{prefix}:{Colors.RESET} [{bar}] {Colors.GREEN}{percent:.1%}{Colors.RESET} ({current}/{total}) {Colors.GREEN}COMPLETE{Colors.RESET}")
        sys.stdout.write(f"\033[4B")  # Move back down
        sys.stdout.flush()


def print_chunk_status_inline(chunk_idx, total_chunks, chunk_name, text_preview):
    """Print chunk transcription status that updates in place."""
    # Show a preview of the transcribed text
    preview = text_preview[:60] + "..." if len(text_preview) > 60 else text_preview
    
    # Move to line 2 (chunk info), clear and write
    sys.stdout.write(f"\033[2A")  # Move up 2 lines
    sys.stdout.write(f"\r{' ' * 100}\r")  # Clear line
    sys.stdout.write(f"{Colors.BOLD}Processing Chunk {chunk_idx}/{total_chunks}{Colors.RESET}")
    
    # Move to line 3 (filename), clear and write
    sys.stdout.write(f"\033[1B\r{' ' * 100}\r")  # Move down 1, clear line
    sys.stdout.write(f"{Colors.DIM}File: {chunk_name}{Colors.RESET}")
    
    # Move to line 4 (transcription), clear and write
    sys.stdout.write(f"\033[1B\r{' ' * 100}\r")  # Move down 1, clear line
    sys.stdout.write(f"{Colors.GREEN}Text: \"{preview}\"{Colors.RESET}")
    
    sys.stdout.flush()


def print_section_header(title, icon=""):
    """Print a styled section header."""
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}{title.upper()}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'═' * len(title)}{Colors.RESET}")


def print_audio_info(duration):
    """Print audio file information with style."""
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    
    print(f"{Colors.CYAN}Audio Duration:{Colors.RESET} {Colors.BOLD}{minutes}m {seconds}s{Colors.RESET}")
    
    if duration > 300:  # 5 minutes
        print(f"{Colors.YELLOW}Long audio detected - Using chunked processing for optimal performance{Colors.RESET}")
    print()


def print_completion_message(output_file):
    """Print a stylish completion message."""
    completion_banner = f"""
{Colors.GREEN}{Colors.BOLD}
TRANSCRIPTION COMPLETE!
{Colors.RESET}
{Colors.CYAN}Output saved to: {Colors.BOLD}{output_file}{Colors.RESET}

{Colors.YELLOW}Check AI Generated Transcript for accuracy and completeness{Colors.RESET}
    """
    print(completion_banner)


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
    print_section_header("Audio Processing")
    print(f"{Colors.CYAN}Splitting audio into optimized chunks...{Colors.RESET}")
    
    # Show chunking animation
    for i in range(4):
        sys.stdout.write(f"\r{Colors.YELLOW}{'⣾⣽⣻⢿⡿⣟⣯⣷'[i % 8]} Chunking audio")
        sys.stdout.flush()
        time.sleep(0.3)
    print(f"\r{Colors.GREEN}Audio chunking started{Colors.RESET}")
    
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

    print(f"{Colors.GREEN}Created {Colors.BOLD}{len(audio_chunks)}{Colors.RESET}{Colors.GREEN} audio chunks for processing{Colors.RESET}")

    return audio_chunks


def has_spoken_content(text):
    """Check if transcribed text contains meaningful spoken content."""
    if not text:
        return False
    
    # Remove common noise patterns and clean text
    cleaned_text = text.strip().lower()
    
    # Remove common Whisper hallucination patterns
    noise_patterns = [
        "thank you",
        "thanks for watching", 
        "like and subscribe",
        "music",
        "applause",
        "[music]",
        "[applause]",
        "♪",
        ".",
        ",",
        "!",
        "?",
        " "
    ]
    
    for pattern in noise_patterns:
        cleaned_text = cleaned_text.replace(pattern, "")
    
    # Check if there's substantial content left (at least 10 characters of meaningful text)
    return len(cleaned_text) >= 10


def transcribe_chunk_with_live_update(model, chunk_path, chunk_idx, total_chunks, chunk_name):
    """Transcribe a single audio chunk with live text updates, skipping empty chunks."""
    segments, _ = model.transcribe(chunk_path, beam_size=5, language="en")
    
    # Build text progressively and update display
    full_text = ""
    for segment in segments:
        segment_text = segment.text.strip()
        full_text += segment_text + " "
        
        # Update display with growing text
        preview = full_text[:80] + "..." if len(full_text) > 80 else full_text
        print_chunk_status_inline(chunk_idx, total_chunks, chunk_name, preview)
        
        # Small delay to show the progressive text building
        time.sleep(0.1)
    
    final_text = full_text.strip()
    
    # Check if chunk has meaningful spoken content
    if not has_spoken_content(final_text):
        print_chunk_status_inline(chunk_idx, total_chunks, chunk_name, f"{Colors.YELLOW}SKIPPED: No spoken content detected{Colors.RESET}")
        return None  # Return None to indicate this chunk should be skipped
    
    return final_text


def main(input_audio, output_text):
    """Main function to transcribe audio file to text using chunked processing."""
    
    # Print sexy startup banner
    print_banner()
    print_system_info()
    
    # Check if input file exists
    if not os.path.exists(input_audio):
        raise FileNotFoundError(f"Input audio file not found: {input_audio}")
    
    print_section_header("Audio Analysis")
    audio_length_sec = get_audio_length(input_audio)
    print_audio_info(audio_length_sec)

    print_section_header("Model Loading")
    print(f"{Colors.YELLOW}Loading Whisper AI model...{Colors.RESET}")
    
    if is_mac:
        model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        print(f"{Colors.GREEN}Model loaded on CPU (Mac optimized){Colors.RESET}")
    else:
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        print(f"{Colors.GREEN}Model loaded on CUDA (GPU accelerated){Colors.RESET}")

    with tempfile.TemporaryDirectory(prefix="audio_chunks_") as temp_audio_dir, \
            tempfile.TemporaryDirectory(prefix="text_chunks_") as temp_text_dir:

        audio_chunks = split_audio_ffmpeg(input_audio, temp_audio_dir)

        if not audio_chunks:
            print(f"{Colors.RED}❌ No chunks created. Check your input audio and ffmpeg output.{Colors.RESET}")
            return

        print_section_header("Transcription Processing")
        transcription_files = []
        total_chunks = len(audio_chunks)
        skipped_chunks = 0
        last_text = "Ready to start transcription..."  # Keep track of last displayed text
        
        # Create space for the display
        print()  # Space for progress bar
        print()  # Space for chunk info
        print()  # Space for filename
        print()  # Space for transcription text

        for idx, chunk in enumerate(audio_chunks, 1):
            # Show current chunk info with last text until new transcription is ready
            chunk_name = os.path.basename(chunk)
            print_chunk_status_inline(idx, total_chunks, chunk_name, last_text)
            
            # Update progress bar AFTER chunk status (so it shows above)
            print_progress_bar(idx - 1, total_chunks, "Overall Progress")
            
            # Process chunk with live text updates
            text = transcribe_chunk_with_live_update(model, chunk, idx, total_chunks, chunk_name)
            
            # Skip chunks with no spoken content
            if text is None:
                skipped_chunks += 1
                last_text = f"SKIPPED: No meaningful content in chunk {idx}"
                continue
            
            last_text = f"DONE: {text[:80]}{'...' if len(text) > 80 else ''}"  # Update last_text

            txt_file = os.path.join(temp_text_dir, os.path.basename(chunk).replace(".wav", ".txt"))
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(text)
            transcription_files.append(txt_file)

        # Final progress bar completion
        print()  # New line after the last chunk status
        print_progress_bar(total_chunks, total_chunks, "Overall Progress")
        
        # Show processing statistics
        processed_chunks = len(transcription_files)
        print(f"\n{Colors.CYAN}Processing Summary:{Colors.RESET}")
        print(f"  • Total chunks: {Colors.BOLD}{total_chunks}{Colors.RESET}")
        print(f"  • Processed: {Colors.GREEN}{Colors.BOLD}{processed_chunks}{Colors.RESET}")
        print(f"  • Skipped (no speech): {Colors.YELLOW}{Colors.BOLD}{skipped_chunks}{Colors.RESET}")

        print_section_header("Final Assembly")
        print(f"{Colors.YELLOW}Combining {processed_chunks} transcription chunks...{Colors.RESET}")
        
        with open(output_text, "w", encoding="utf-8") as outfile:
            for txt_file in transcription_files:
                with open(txt_file, "r", encoding="utf-8") as infile:
                    content = infile.read().strip()
                    if content:  # Only write non-empty content
                        outfile.write(content + " ")

    print_completion_message(output_text)


if __name__ == "__main__":
    # Configuration for the transcription process
    # These paths can be modified to match your specific files
    input_audio = "input.m4a"  # Path to the audio file to transcribe
    output_text = "final_transcription.txt"  # Path where transcription will be saved
    
    try:
        # Start the transcription process
        main(input_audio, output_text)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print(f"\n\n{Colors.YELLOW}{Colors.BOLD}TRANSCRIPTION INTERRUPTED{Colors.RESET}")
        print(f"{Colors.CYAN}Process cancelled by user (Ctrl+C){Colors.RESET}")
        print(f"{Colors.DIM}Cleaning up temporary files...{Colors.RESET}")
        exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}ERROR DURING TRANSCRIPTION{Colors.RESET}")
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        print(f"\n{Colors.BLUE}{Colors.BOLD}Please ensure:{Colors.RESET}")
        print(f"{Colors.CYAN}1. Your audio file '{input_audio}' exists in the current directory{Colors.RESET}")
        print(f"{Colors.CYAN}2. FFmpeg is installed and available in your PATH{Colors.RESET}")
        print(f"{Colors.CYAN}3. The audio file format is supported by FFmpeg{Colors.RESET}")
        print(f"{Colors.CYAN}4. You have sufficient disk space for temporary files{Colors.RESET}")
        print(f"\n{Colors.YELLOW}For troubleshooting, check that all dependencies are installed{Colors.RESET}")
        exit(1)