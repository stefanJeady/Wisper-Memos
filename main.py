"""Audio Transcription Tool using Faster Whisper - See README.md for full documentation."""

import os
import subprocess
import tempfile
import logging
import time
import sys
import yaml
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


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"{Colors.RED}Configuration file '{config_path}' not found. Using default settings.{Colors.RESET}")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"{Colors.RED}Error parsing YAML configuration: {e}. Using default settings.{Colors.RESET}")
        return get_default_config()

def get_default_config():
    """Return default configuration if YAML file is not available"""
    return {
        'whisper': {
            'input_audio': 'input.m4a',
            'output_text': 'final_transcription.txt',
            'model_size': 'large-v3',
            'beam_size': 20,
            'language': 'en',
            'include_timestamps': True,
            'chunk_length_sec': 30,
            'mac': {
                'device': 'cpu',
                'compute_type': 'int8'
            },
            'other': {
                'device': 'cuda',
                'compute_type': 'float16'
            }
        },
        'output': {
            'encoding': 'utf-8'
        }
    }


def print_banner():
    """Print a sexy startup banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                          WHISPER Memos                       ║
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
    
    # Check CUDA availability without creating a model
    try:
        import torch
        if torch.cuda.is_available():
            cuda_status = "Available (CTranslate2)"
        else:
            cuda_status = "Not Available"
    except:
        # Fallback check - try to import ctranslate2 directly
        try:
            import ctranslate2
            # Check if CUDA devices are available through ctranslate2
            if hasattr(ctranslate2, 'get_cuda_device_count') and ctranslate2.get_cuda_device_count() > 0:
                cuda_status = "Available (CTranslate2)"
            else:
                cuda_status = "Not Available"
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
    
    # Move up 2 lines to position above chunk status, clear and write progress
    sys.stdout.write(f"\033[2A")  # Move up 2 lines (reduced from 3)
    sys.stdout.write(f"\r{' ' * 100}\r")  # Clear line
    sys.stdout.write(f"{Colors.BOLD}{prefix}:{Colors.RESET} [{bar}] {Colors.GREEN}{percent:.1%}{Colors.RESET} ({current}/{total})")
    sys.stdout.write(f"\033[2B")  # Move back down 2 lines (reduced from 3)
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write(f"\033[2A")  # Move up to progress line (reduced from 3)
        sys.stdout.write(f"\r{Colors.BOLD}{prefix}:{Colors.RESET} [{bar}] {Colors.GREEN}{percent:.1%}{Colors.RESET} ({current}/{total}) {Colors.GREEN}COMPLETE{Colors.RESET}")
        sys.stdout.write(f"\033[2B")  # Move back down (reduced from 3)
        sys.stdout.flush()


def print_chunk_status_inline(chunk_idx, total_chunks, chunk_name, text_preview):
    """Print chunk transcription status that updates in place."""
    # Show a preview of the transcribed text
    preview = text_preview[:60] + "..." if len(text_preview) > 60 else text_preview
    
    # Move to line 2 (chunk info), clear and write
    sys.stdout.write(f"\033[1A")  # Move up 1 line (reduced from 2)
    sys.stdout.write(f"\r{' ' * 100}\r")  # Clear line
    sys.stdout.write(f"{Colors.BOLD}Processing Chunk {chunk_idx}/{total_chunks}{Colors.RESET}")
    
    # Move to line 3 (transcription), clear and write
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


def run_deepseek_analysis(config):
    """Run DeepSeek analysis automatically after successful transcription."""
    workflow_config = config.get('workflow', {})
    analysis_script = workflow_config.get('analysis_script', 'deepseek_analysis.py')
    analysis_delay = workflow_config.get('analysis_delay', 2)
    
    print_section_header("Auto-Analysis")
    print(f"{Colors.CYAN}Starting automated DeepSeek analysis...{Colors.RESET}")
    
    # Optional delay to allow file system to sync
    if analysis_delay > 0:
        print(f"{Colors.DIM}Waiting {analysis_delay} seconds for file sync...{Colors.RESET}")
        time.sleep(analysis_delay)
    
    try:
        # Check if analysis script exists
        if not os.path.exists(analysis_script):
            print(f"{Colors.RED}Analysis script not found: {analysis_script}{Colors.RESET}")
            return
        
        print(f"{Colors.YELLOW}Running: python {analysis_script}{Colors.RESET}")
        print(f"{Colors.DIM}This may take several minutes depending on transcription length...{Colors.RESET}")
        print()
        
        # Run the analysis script
        result = subprocess.run(
            [sys.executable, analysis_script],
            capture_output=False,  # Allow real-time output
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}ANALYSIS COMPLETE!{Colors.RESET}")
            print(f"{Colors.CYAN}DeepSeek analysis finished successfully{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}Analysis script exited with error code: {result.returncode}{Colors.RESET}")
            
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error running analysis: {e}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Unexpected error during analysis: {e}{Colors.RESET}")


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

    # Calculate actual chunk durations for accurate timestamps
    chunk_durations = []
    for chunk_path in audio_chunks:
        try:
            duration = get_audio_length(chunk_path)
            chunk_durations.append(duration)
        except:
            # Fallback to the target chunk length if we can't get the actual duration
            chunk_durations.append(chunk_length_sec)

    return audio_chunks, chunk_durations


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


def transcribe_chunk_with_live_update(model, chunk_path, chunk_idx, total_chunks, chunk_name, beam_size=5, language="en"):
    """Transcribe a single audio chunk with live text updates, skipping empty chunks."""
    segments, _ = model.transcribe(chunk_path, beam_size=beam_size, language=language)
    
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


def main(config=None):
    """Main function to transcribe audio file to text using chunked processing."""
    
    # Load configuration
    if config is None:
        config = load_config()
    
    # Extract settings from config
    whisper_config = config['whisper']
    input_audio = whisper_config['input_audio']
    output_text = whisper_config['output_text']
    model_size = whisper_config['model_size']
    beam_size = whisper_config['beam_size']
    language = whisper_config['language']
    include_timestamps = whisper_config['include_timestamps']
    chunk_length_sec = whisper_config['chunk_length_sec']
    
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
    
    # Warning for low beam size
    if beam_size < 15:
        print(f"{Colors.RED}{Colors.BOLD}LOW QUALITY BEAM SIZE WARNING{Colors.RESET}")
        print(f"{Colors.YELLOW}Beam size of {beam_size} is quite low. For better transcription quality,{Colors.RESET}")
        print(f"{Colors.YELLOW}consider using beam_size=15 or higher, especially for challenging audio.{Colors.RESET}")
        print(f"{Colors.CYAN}Current setting may result in lower accuracy transcriptions.{Colors.RESET}")
        print()
    
    # Warning for high beam size
    if beam_size > 20:
        print(f"{Colors.RED}{Colors.BOLD}HIGH BEAM SIZE WARNING{Colors.RESET}")
        print(f"{Colors.YELLOW}Beam size of {beam_size} is very high. Values above 20 have{Colors.RESET}")
        print(f"{Colors.YELLOW}diminishing returns and will slow processing drastically.{Colors.RESET}")
        print(f"{Colors.CYAN}Consider using beam_size=15-20 for optimal balance of quality and speed.{Colors.RESET}")
        print()
    
    if is_mac:
        device = whisper_config['mac']['device']
        compute_type = whisper_config['mac']['compute_type']
        print(f"{Colors.YELLOW}Loading Whisper AI model ({model_size}, {device}, {compute_type}, beam_size={beam_size}, language={language})...{Colors.RESET}")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"{Colors.GREEN}Model loaded on CPU (Mac optimized){Colors.RESET}")
    else:
        device = whisper_config['other']['device']
        compute_type = whisper_config['other']['compute_type']
        print(f"{Colors.YELLOW}Loading Whisper AI model ({model_size}, {device}, {compute_type}, beam_size={beam_size}, language={language})...{Colors.RESET}")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"{Colors.GREEN}Model loaded on CUDA (GPU accelerated){Colors.RESET}")

    with tempfile.TemporaryDirectory(prefix="audio_chunks_") as temp_audio_dir, \
            tempfile.TemporaryDirectory(prefix="text_chunks_") as temp_text_dir:

        audio_chunks, chunk_durations = split_audio_ffmpeg(input_audio, temp_audio_dir, chunk_length_sec)

        if not audio_chunks:
            print(f"{Colors.RED} No chunks created. Check your input audio and ffmpeg output.{Colors.RESET}")
            return

        print_section_header("Transcription Processing")
        transcription_files = []
        total_chunks = len(audio_chunks)
        skipped_chunks = 0
        
        # Create space for the display
        print()  # Space for progress bar
        print()  # Space for chunk info
        print()  # Space for transcription text

        for idx, chunk in enumerate(audio_chunks, 1):
            # Show current chunk info
            chunk_name = os.path.basename(chunk)
            
            # Calculate timestamp for this chunk using actual chunk durations
            chunk_start_time_sec = sum(chunk_durations[:idx-1])  # Sum of all previous chunks
            
            # Update progress bar AFTER chunk status (so it shows above)
            print_progress_bar(idx, total_chunks, "Overall Progress")
            
            # Process chunk with live text updates
            text = transcribe_chunk_with_live_update(model, chunk, idx, total_chunks, chunk_name, beam_size, language)
            
            # Skip chunks with no spoken content
            if text is None:
                skipped_chunks += 1
                continue
            
            # Add timestamp to text if enabled
            if include_timestamps:
                # Format timestamp as [MM:SS] or [HH:MM:SS] for longer audio
                minutes = int(chunk_start_time_sec // 60)
                seconds = int(chunk_start_time_sec % 60)
                if minutes >= 60:
                    hours = int(minutes // 60)
                    minutes = int(minutes % 60)
                    timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"
                else:
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                
                text = f"{timestamp} {text}"
            
            txt_file = os.path.join(temp_text_dir, os.path.basename(chunk).replace(".wav", ".txt"))
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(text)
            transcription_files.append(txt_file)

        # Clear the progress display and move to next section
        print()  # New line after the last chunk status
        
        # Show processing statistics
        processed_chunks = len(transcription_files)
        print(f"\n{Colors.CYAN}Processing Summary:{Colors.RESET}")
        print(f"  • Total chunks: {Colors.BOLD}{total_chunks}{Colors.RESET}")
        print(f"  • Processed: {Colors.GREEN}{Colors.BOLD}{processed_chunks}{Colors.RESET}")
        print(f"  • Skipped (no speech): {Colors.YELLOW}{Colors.BOLD}{skipped_chunks}{Colors.RESET}")

        print_section_header("Final Assembly")
        print(f"{Colors.YELLOW}Combining {processed_chunks} transcription chunks...{Colors.RESET}")
        
        with open(output_text, "w", encoding=config['output']['encoding']) as outfile:
            for txt_file in transcription_files:
                with open(txt_file, "r", encoding=config['output']['encoding']) as infile:
                    content = infile.read().strip()
                    if content:  # Only write non-empty content
                        if include_timestamps:
                            # Add each chunk on a new line when timestamps are included for better readability
                            outfile.write(content + "\n\n")
                        else:
                            # Original behavior: concatenate with spaces
                            outfile.write(content + " ")

    print_completion_message(output_text)
    
    # Workflow automation: Auto-run DeepSeek analysis if enabled
    if config.get('workflow', {}).get('auto_run_analysis', False):
        run_deepseek_analysis(config)


if __name__ == "__main__":
    # Load configuration from YAML file
    config = load_config()
    
    try:
        # Start the transcription process using configuration settings
        main(config)
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
        print(f"{Colors.CYAN}1. Your audio file exists in the current directory{Colors.RESET}")
        print(f"{Colors.CYAN}2. FFmpeg is installed and available in your PATH{Colors.RESET}")
        print(f"{Colors.CYAN}3. The audio file format is supported by FFmpeg{Colors.RESET}")
        print(f"{Colors.CYAN}4. You have sufficient disk space for temporary files{Colors.RESET}")
        print(f"{Colors.CYAN}5. The config.yaml file is properly formatted{Colors.RESET}")
        print(f"\n{Colors.YELLOW}For troubleshooting, check that all dependencies are installed{Colors.RESET}")
        print(f"{Colors.YELLOW}and verify your configuration in config.yaml{Colors.RESET}")
        exit(1)