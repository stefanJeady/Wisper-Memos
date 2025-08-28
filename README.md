# Audio Transcription & Analysis Tool

A comprehensive toolkit that combines high-quality audio transcription using Faster Whisper with intelligent transcript analysis using DeepSeek R1. The speed of your GPU will determine how long transcription takes to complete.

## Table of Contents

### Transcription Tool

- [Transcription Features](#transcription-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Configuration](#configuration)
- [Workflow Automation](#workflow-automation-configuration)
- [Performance Notes](#performance-notes)
- [Troubleshooting](#troubleshooting)

### Analysis Tool

- [DeepSeek R1 Local Transcript Analysis](#deepseek-r1-local-transcript-analysis)
- [DeepSeek Quick Start](#deepseek-quick-start)
- [Analysis Features](#analysis-features)
- [DeepSeek Technical Details](#deepseek-technical-details)
- [DeepSeek Troubleshooting](#deepseek-troubleshooting)

### Configuration & Support

- [Configuration Files](#configuration-files)
- [Dependencies](#dependencies)
- [Privacy & Security](#privacy--security)

## Features

### Transcription Features

- **YAML Configuration** - All settings centralized in `config.yaml` for easy customization
- **Automatic platform detection** - Mac uses CPU mode, other platforms use CUDA if available
- **Audio chunking** - Splits large files into optimal chunks for better transcription quality
- **Timestamp support** - Optional timestamps showing when each chunk begins in the audio
- **Multi-format support** - Supports various audio formats via FFmpeg
- **Comprehensive logging** - Track the transcription process with detailed logs
- **High accuracy** - Uses the large-v3 Whisper model with beam search for best results
- **Workflow Automation** - Automatically run DeepSeek analysis after successful transcription

### Analysis Features

- **Local AI Analysis** - DeepSeek R1 model running locally via Ollama
- **Multiple Analysis Types** - Comprehensive, summary, action items, and participant analysis
- **Privacy First** - 100% local processing, no cloud services required
- **Intelligent Insights** - Meeting summaries, action items, and participant analysis
- **Customizable Prompts** - Modify analysis prompts for specific needs

## Prerequisites

### Required Software

- **Python 3.8+**
- **FFmpeg** - Must be installed and accessible from command line
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (Ubuntu/Debian)
- **CUDA Drivers & cuDNN** (Optional, for GPU acceleration)
  - **NVIDIA GPU**: Required for CUDA acceleration on Windows/Linux
  - **CUDA Toolkit**: Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
  - **cuDNN Library**: Download from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
  - **Compatible GPU**: NVIDIA GPU with CUDA Compute Capability 3.5 or higher
  - **Windows Installation**: Both CUDA and cuDNN must be added to your system PATH - restart required after installation
  - **Note**: Mac users will use CPU mode automatically (CUDA not supported on Apple Silicon)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install faster_whisper torch PyYAML
```

## Installation

1. Clone or download this repository
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure FFmpeg is installed and available in your system PATH

## Usage

### Basic Usage

1. Configure your settings in `config.yaml` (see Configuration section below)
2. Place your audio file in the project directory (or update the path in `config.yaml`)
3. Run the script:

   ```bash
   python main.py
   ```

**With Workflow Automation enabled**, this single command will:

- Complete audio transcription
- Automatically start DeepSeek analysis (if `auto_run_analysis: true`)
- Provide a complete audio → transcription → AI analysis workflow

### Configuration

All settings are now managed through the `config.yaml` file. This makes it easy to adjust parameters without modifying the code.

#### Configuration File Structure

```yaml
# Whisper Transcription Settings
whisper:
  # File paths
  input_audio: "input.m4a"                    # Path to the audio file to transcribe
  output_text: "final_transcription.txt"     # Path where transcription will be saved
  
  # Model configuration
  model_size: "large-v3"                     # Whisper model size
  beam_size: 20                              # Beam search size for accuracy
  language: "en"                             # Language code for transcription
  
  # Processing settings
  include_timestamps: true                   # Include timestamps in output
  chunk_length_sec: 30                       # Audio chunk length in seconds
```

#### Key Settings Explained

- **`model_size`**: Choose from `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`
- **`beam_size`**: Range 5-20. Higher values = better accuracy but slower processing. Use 15+ for challenging audio
- **`language`**: Use standard language codes (`en`, `es`, `fr`, `de`, etc.)
- **`include_timestamps`**: Add timestamps showing when each chunk begins
- **`chunk_length_sec`**: Controls segmentation size and timestamp precision

#### Timestamp Configuration

The tool supports optional timestamps that show when each chunk begins in the audio. This is now configured in `config.yaml`:

- **`include_timestamps`**: Set to `true` to add timestamps, `false` to disable
- **`chunk_length_sec`**: Controls audio segmentation size (default: 30 seconds)

**Timestamp Accuracy:**

The timestamps are calculated using the actual duration of each audio chunk, not estimated values. This ensures perfect accuracy even when:

- Chunks are slightly shorter/longer than the target duration
- The last chunk is shorter than the target duration
- FFmpeg makes boundary adjustments during segmentation

**Example output with timestamps:**

```text
[00:00] This is the beginning of the audio transcription.

[00:29] Here's what was said in the second chunk of audio.

[00:58] And this continues with the third chunk.

[01:27] For longer audio, timestamps can show hours: [01:05:30]
```

**Timestamp Format:**

- Short audio: `[MM:SS]` (e.g., `[05:30]`)
- Long audio: `[HH:MM:SS]` (e.g., `[01:05:30]`)

When timestamps are enabled, each chunk appears on a separate line for better readability. When disabled, all text is concatenated with spaces as before.

#### Workflow Automation Configuration

The tool supports automatic execution of DeepSeek analysis after successful transcription:

```yaml
# Workflow Automation Settings
workflow:
  auto_run_analysis: true                   # Enable automatic analysis after transcription
  analysis_script: "deepseek_analysis.py"  # Path to the analysis script
  analysis_delay: 2                         # Wait time before starting analysis (seconds)
```

**Workflow Settings Explained:**

- **`auto_run_analysis`**: Set to `true` to automatically run DeepSeek analysis after transcription completes successfully, `false` to disable
- **`analysis_script`**: Path to the analysis script (default: `deepseek_analysis.py`)
- **`analysis_delay`**: Optional delay in seconds before starting analysis (allows file system to sync)

**How it works:**

1. Run `python main.py` to start transcription
2. When transcription completes successfully, analysis automatically starts (if enabled)
3. Both processes complete seamlessly without manual intervention

This creates a complete audio → transcription → AI analysis workflow with a single command!

#### Available Model Sizes

- `tiny` - Fastest, lowest accuracy (~39 MB)
- `base` - Good balance of speed and accuracy (~74 MB)
- `small` - Better accuracy (~244 MB)
- `medium` - High accuracy (~769 MB)
- `large`, `large-v2`, `large-v3` - Best accuracy (~1550 MB)

#### Language Codes

Common language codes include:

- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean

#### Beam Size Settings

- `1` - Fastest transcription, lower accuracy
- `5` - Balanced accuracy and speed (recommended)
- `10` - Higher accuracy, slower processing
- `20` - Required for Challenging files, extremely slow, best accuracy

### Supported Audio Formats

- MP3, M4A, WAV, FLAC, OGG
- Any format supported by FFmpeg
- **Transcribing Apple Voice Memos (M4A files):**  
    To transcribe Apple Voice Memos, open the Voice Memos app on your Mac, then drag your desired recording from the Voice Memos sidebar directly into the root folder of this project.
    Rename the file to `input.m4a` before running the transcription tool. This ensures seamless transcription of your Apple Voice Memo audio files to text.

## How It Works

1. **Audio Analysis** - The tool first analyzes the input audio to determine its duration
2. **Platform Detection** - Automatically detects if running on Mac (CPU mode) or other platforms (CUDA mode)
3. **Model Loading** - Loads the configured Whisper model with your specified settings (displays actual model size, device, compute type, beam size, and language)
4. **Audio Chunking** - Splits the audio into configurable chunks (default: 30 seconds) for optimal processing
5. **Transcription** - Each chunk is transcribed using your configured beam search and language settings
6. **Timestamp Addition** - If enabled, timestamps are added to show when each chunk begins
7. **Assembly** - All transcriptions are combined into a single output file with optional formatting

## Configuration Options

### Model Configuration

All model settings are now centralized in the main configuration section at the bottom of `main.py`. You can easily modify:

- **Model Size**: Choose from tiny, base, small, medium, large, large-v2, or large-v3
- **Beam Size**: Adjust transcription accuracy vs speed (1-10, recommended: 5)
- **Language**: Set the target language code for transcription
- **File Paths**: Configure input audio and output text file locations

### Platform-Specific Settings

- **Mac**: Automatically uses CPU with int8 compute type for compatibility
- **Other platforms**: Uses CUDA with float16 for better performance (if CUDA is available)

### Chunk Length

The audio chunk length is configurable in the main configuration section and controls how FFmpeg segments the audio:

```python
if __name__ == "__main__":
    # Timestamp configuration
    include_timestamps = True    # Add timestamps to output
    chunk_length_sec = 30       # Length of each audio chunk in seconds
```

**Chunk Length Considerations:**

- **Shorter chunks (15-20s)**: More processing overhead, but may help with very challenging audio
- **Default (30s)**: Good balance of performance and transcription quality
- **Longer chunks (45-60s)**: Faster processing, fewer chunk boundaries
- **Very long chunks (>60s)**: May reduce transcription quality for some audio types

**Note:** Timestamps are calculated from actual chunk durations, not the target `chunk_length_sec` value, ensuring accuracy regardless of the segmentation setting.

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
   - Try increasing beam_size in the configuration (15-20 for challenging audio)
   - Experiment with different chunk lengths (15-45 seconds) for difficult audio
   - Use a larger model size (medium, large, or large-v3) for better accuracy

## Output

The transcription is saved as a plain text file with all audio segments combined. The tool provides real-time progress updates showing:

- Audio duration and chunk count
- Progress through each chunk
- Transcription preview for each processed chunk

## Configuration Files

This project uses YAML configuration files for easy customization:

- **`config.yaml`**: Main configuration file containing all Whisper and output settings
- **`requirements.txt`**: Python dependencies for the transcription tool
- **`requirements_deepseek.txt`**: Additional dependencies for the DeepSeek analysis tool

For detailed configuration options, see [CONFIGURATION.md](CONFIGURATION.md).

---

## DeepSeek R1 Local Transcript Analysis

This tool processes your transcripts using the DeepSeek R1 model running locally via Ollama. No cloud services or API keys required!

## DeepSeek Quick Start

### Prerequisites for Analysis

1. **Python 3.7+** - Already installed if you ran the main transcription tool
2. **Ollama** - Download from [https://ollama.ai](https://ollama.ai)

### DeepSeek Installation Steps

1. **Install Ollama** (if not already installed):
   - Visit [https://ollama.ai](https://ollama.ai)
   - Download and install for Windows
   - Restart your terminal/PowerShell

2. **Pull the DeepSeek R1 model**:

   ```powershell
   ollama pull deepseek-r1:latest
   ```

   **Note**: This model is large (~8-14GB) and may take time to download

3. **Install Python dependencies**:

   ```powershell
   pip install -r requirements_deepseek.txt
   ```

4. **Configure settings** (optional):
   - Settings are automatically loaded from `config.yaml`
   - You can modify Ollama URL, model name, and timeout values in the config file

### Running the Analysis

#### Option 1: Automated Setup (Recommended)

```powershell
# Run the PowerShell setup script
.\run_deepseek_analysis.ps1
```

#### Option 2: Manual Execution

```powershell
# Start Ollama service
ollama serve

# Run the analysis tool
python deepseek_analysis.py
```

## DeepSeek Configuration

The tool uses settings from `config.yaml` for easy customization:

```yaml
# DeepSeek R1 Analysis Settings
deepseek:
  # Ollama configuration
  ollama_url: "http://localhost:11434"       # Ollama server URL
  model_name: "deepseek-r1:latest"           # DeepSeek model name
  
  # Analysis settings
  timeout: 300                               # Request timeout in seconds (5 minutes)
  
  # Generation parameters
  generation:
    temperature: 0.7                         # Creativity/randomness (0.0-2.0, lower = more focused)
    top_p: 0.9                              # Nucleus sampling (0.0-1.0, controls diversity)
    max_tokens: 2048                        # Maximum response length
    stream: false                           # Stream response (true/false)
    request_timeout: 120                    # Individual request timeout in seconds
  
  # Prompts folder
  prompts_folder: "prompts"                  # Folder containing analysis prompts
```

You can modify these settings to:

- Change the Ollama server URL (if running on a different port/host)
- Use a different DeepSeek model variant
- Adjust timeout for longer analyses
- Point to a different prompts folder
- Fine-tune AI generation parameters:
  - **Temperature**: Controls creativity/randomness (0.0 = very focused, 2.0 = very creative)
  - **Top_p**: Controls diversity of word choices (0.1 = very focused, 1.0 = full vocabulary)
  - **Max_tokens**: Maximum length of the AI response
  - **Stream**: Whether to stream responses in real-time
  - **Request_timeout**: How long to wait for individual requests

## Analysis Features (Deepseek)

### Analysis Types Available

1. **Comprehensive Analysis** (Default)
   - Meeting summary
   - Key topics discussed
   - Participant identification
   - Action items and decisions
   - Important dates/deadlines
   - Tone assessment
   - Recommendations

2. **Summary Only**
   - Concise overview of main points
   - Key decisions made
   - Basic action items

3. **Action Items Only**
   - Extracted tasks and follow-ups
   - Decision points
   - Actionable items list

4. **Participants Analysis**
   - Speaker identification
   - Role analysis
   - Contribution assessment

### Output Options

- **Console Display**: Real-time results in your terminal
- **Markdown File**: Saved analysis with timestamp
- **Multiple Formats**: Structured, readable output

## DeepSeek Technical Details

### System Requirements for Analysis

- **RAM**: 8GB minimum (16GB recommended for larger models)
- **Storage**: 15-20GB free space for model
- **GPU**: Optional but recommended for faster processing

### Model Information

- **Model**: DeepSeek R1 (latest version)
- **Provider**: Ollama local inference
- **Privacy**: 100% local processing, no data sent to cloud
- **Languages**: Primarily English, with multilingual capabilities

### Analysis Performance

- **Small transcripts** (<1000 words): 30-60 seconds
- **Medium transcripts** (1000-5000 words): 1-3 minutes
- **Large transcripts** (5000+ words): 3-10 minutes
- *Times vary based on hardware and model size*

## DeepSeek Troubleshooting

### Common Analysis Issues

#### "Ollama not found" Error

```powershell
# Check if Ollama is installed
ollama --version

# If not found, reinstall from https://ollama.ai
# Make sure to restart your terminal after installation
```

#### "Model not found" Error

```powershell
# Pull the DeepSeek R1 model
ollama pull deepseek-r1:latest

# Check available models
ollama list
```

#### "Connection refused" Error

```powershell
# Start Ollama service
ollama serve

# Check if service is running
curl http://localhost:11434/api/tags
```

#### Out of Memory Issues

- Close other applications
- Try a smaller model variant:

  ```powershell
  ollama pull deepseek-r1:7b  # Smaller version
  ```

#### Slow Analysis Performance

- Ensure adequate RAM (8GB+ available)
- Consider GPU acceleration if available
- Break large transcripts into smaller chunks

### Network Issues

- The tool runs 100% locally - no internet required after model download
- If initial model download fails, check your internet connection
- Corporate firewalls may block Ollama model downloads

## Privacy & Security

- **100% Local Processing**: No data leaves your machine
- **No API Keys**: No cloud services or external accounts needed
- **Offline Capable**: Works without internet after initial setup
- **Your Data Stays Yours**: Complete privacy and control

## Tips for Best Analysis Results

1. **Clear Transcripts**: Ensure your transcript is well-formatted
2. **Specific Prompts**: The tool works best with meeting transcripts
3. **Hardware**: More RAM = faster processing
4. **Model Selection**: Larger models = better analysis quality
5. **Chunking**: For very long transcripts, consider breaking into sections

### Tuning AI Parameters

You can adjust the generation parameters in `config.yaml` for different use cases:

**For More Creative/Detailed Analysis:**

```yaml
generation:
  temperature: 1.0    # Higher creativity
  top_p: 0.95        # More diverse vocabulary
  max_tokens: 4096   # Longer responses
```

**For More Focused/Concise Analysis:**

```yaml
generation:
  temperature: 0.3   # More focused
  top_p: 0.8         # Limited vocabulary
  max_tokens: 1024   # Shorter responses
```

**For Faster Processing:**

```yaml
generation:
  stream: true       # See results as they generate
  request_timeout: 60 # Shorter timeout
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Dependencies

### Transcription Tool (Whisper)

- **faster_whisper** - AI transcription model
- **torch** - PyTorch for CUDA support
- **PyYAML** - YAML configuration file support
- **ffmpeg** - Audio processing (must be installed separately)

### Analysis Tool (DeepSeek)

- **requests** - HTTP requests for Ollama communication
- **PyYAML** - YAML configuration file support
- **Ollama** - Local AI model server (separate installation)

## Author

Stefan Eady
[stefaneady.com](https://stefaneady.com)
