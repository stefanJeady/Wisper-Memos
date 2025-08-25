# Audio Transcription Tool using Whisper

A powerful audio transcription tool that uses the Faster Whisper model to convert audio files to text. The tool automatically detects your platform and optimizes performance accordingly.
The speed of your GPU will determine how long this takes to complete.

## Features

- **Automatic platform detection** - Mac uses CPU mode, other platforms use CUDA if available
- **Audio chunking** - Splits large files into optimal chunks for better transcription quality
- **Multi-format support** - Supports various audio formats via FFmpeg
- **Comprehensive logging** - Track the transcription process with detailed logs
- **High accuracy** - Uses the large-v3 Whisper model with beam search for best results

## Prerequisites

### Required Software

- **Python 3.8+**
- **FFmpeg** - Must be installed and accessible from command line
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian)

### Python Dependencies

```bash
pip install faster_whisper torch
```

## Installation

1. Clone or download this repository
2. Install the required Python packages:

   ```bash
   pip install faster_whisper torch
   ```

3. Ensure FFmpeg is installed and available in your system PATH

## Usage

### Basic Usage

1. Place your audio file in the project directory (or update the path in `main.py`)
2. Run the script:

   ```bash
   python main.py
   ```

### Customizing Input/Output

Edit the configuration in `main.py`:

```python
if __name__ == "__main__":
    input_audio = "your_audio_file.mp3"  # Change this to your audio file
    output_text = "transcription.txt"    # Change this to your desired output file
    main(input_audio, output_text)
```

### Supported Audio Formats

- MP3, M4A, WAV, FLAC, OGG
- Any format supported by FFmpeg

## How It Works

1. **Audio Analysis** - The tool first analyzes the input audio to determine its duration
2. **Platform Detection** - Automatically detects if running on Mac (CPU mode) or other platforms (CUDA mode)
3. **Model Loading** - Loads the appropriate Whisper large-v3 model configuration
4. **Audio Chunking** - Splits the audio into 30-second chunks for optimal processing
5. **Transcription** - Each chunk is transcribed using beam search (beam_size=5)
6. **Assembly** - All transcriptions are combined into a single output file

## Configuration Options

### Chunk Length

You can modify the chunk length in the `split_audio_ffmpeg` function:

```python
# Default is 30 seconds, you can change this value
audio_chunks = split_audio_ffmpeg(input_audio, temp_audio_dir, chunk_length_sec=30)
```

### Model Settings

- **Mac**: Uses CPU with int8 compute type for compatibility
- **Other platforms**: Uses CUDA with float16 for better performance (if CUDA is available)

## Performance Notes

- **Processing Time**: Approximately 1:10 ratio (10 minutes to process 1 hour of audio)
- **Memory Usage**: Chunking keeps memory usage low even for very long audio files
- **GPU Acceleration**: Automatically uses CUDA if available for faster processing

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Ensure FFmpeg is installed and in your system PATH
   - Test with: `ffmpeg -version`

2. **CUDA errors**
   - The tool will fall back to CPU mode if CUDA is unavailable
   - Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Out of memory errors**
   - Reduce chunk length for very long audio files
   - Close other applications to free up system memory

4. **Poor transcription quality**
   - Ensure audio quality is good (clear speech, minimal background noise)
   - Try increasing beam_size in the `transcribe_chunk` function

## Output

The transcription is saved as a plain text file with all audio segments combined. The tool provides real-time progress updates showing:

- Audio duration and chunk count
- Progress through each chunk
- Transcription preview for each processed chunk

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Dependencies

- **faster_whisper** - AI transcription model
- **torch** - PyTorch for CUDA support  
- **ffmpeg** - Audio processing (must be installed separately)

## Author

Stefan Eady
[stefaneady.com](https://stefaneady.com)
