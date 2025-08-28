"""
DeepSeek R1 Local Analysis Tool
Processes transcripts using local DeepSeek R1 model via Ollama
"""

import os
import json
import time
import requests
import yaml
import gc
from datetime import datetime
from pathlib import Path

class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

class DeepSeekAnalyzer:
    def __init__(self, config_path="config.yaml"):
        """Initialize the DeepSeek R1 analyzer with configuration from YAML file"""
        self.config = self.load_config(config_path)
        self.ollama_url = self.config['deepseek']['ollama_url']
        self.model_name = self.config['deepseek']['model_name']
        self.timeout = self.config['deepseek']['timeout']
        self.prompts_folder = self.config['deepseek']['prompts_folder']
    
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"{Colors.RED}Configuration file '{config_path}' not found. Using default settings.{Colors.RESET}")
            return self.get_default_config()
        except yaml.YAMLError as e:
            print(f"{Colors.RED}Error parsing YAML configuration: {e}. Using default settings.{Colors.RESET}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration if YAML file is not available"""
        return {
            'deepseek': {
                'ollama_url': 'http://localhost:11434',
                'model_name': 'deepseek-r1:latest',
                'timeout': 300,
                'generation': {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 2048,
                    'stream': False,
                    'request_timeout': 120
                },
                'prompts_folder': 'prompts'
            },
            'output': {
                'timestamp_format': '%Y%m%d_%H%M%S',
                'encoding': 'utf-8'
            }
        }
        
    def check_ollama_status(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
            return False
        except requests.exceptions.RequestException:
            return False
    
    def check_model_availability(self):
        """Check if DeepSeek R1 model is available locally"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                available_models = [model['name'] for model in models.get('models', [])]
                
                # Check for various DeepSeek R1 model names
                deepseek_models = [m for m in available_models if 'deepseek-r1' in m.lower()]
                
                if deepseek_models:
                    self.model_name = deepseek_models[0]  # Use the first available DeepSeek R1 model
                    return True
                return False
            return False
        except requests.exceptions.RequestException:
            return False
    
    def install_model(self):
        """Pull DeepSeek R1 model if not available"""
        print(f"{Colors.YELLOW}DeepSeek R1 model not found. Pulling model...{Colors.RESET}")
        print(f"{Colors.CYAN}This may take several minutes depending on your internet connection.{Colors.RESET}")
        
        try:
            # Try to pull the model
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                print(f"{Colors.GREEN}Model pull initiated. Please wait...{Colors.RESET}")
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if 'status' in data:
                                print(f"{Colors.BLUE}Status: {data['status']}{Colors.RESET}")
                        except json.JSONDecodeError:
                            continue
                return True
            else:
                print(f"{Colors.RED}Failed to pull model. Status code: {response.status_code}{Colors.RESET}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"{Colors.RED}Error pulling model: {e}{Colors.RESET}")
            return False
    
    def clear_model_memory(self):
        """Aggressively clear DeepSeek model memory and restart Ollama for fresh analysis"""
        print(f"{Colors.CYAN}Performing aggressive memory clearing for fresh analysis...{Colors.RESET}")
        
        try:
            import subprocess
            import platform
            import time
            
            # Step 1: Stop all models via API
            print(f"{Colors.YELLOW}Step 1: Stopping all loaded models...{Colors.RESET}")
            try:
                response = requests.get(f"{self.ollama_url}/api/ps", timeout=10)
                if response.status_code == 200:
                    running_models = response.json().get('models', [])
                    for model in running_models:
                        model_name = model.get('name')
                        print(f"{Colors.YELLOW}Stopping model: {model_name}{Colors.RESET}")
                        requests.post(f"{self.ollama_url}/api/stop", json={"name": model_name}, timeout=10)
            except:
                pass
            
            # Step 2: Force kill Ollama processes (Windows)
            print(f"{Colors.YELLOW}Step 2: Force stopping Ollama processes...{Colors.RESET}")
            if platform.system() == "Windows":
                try:
                    subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], 
                                 capture_output=True, timeout=10)
                except:
                    try:
                        # Alternative PowerShell method
                        subprocess.run(["powershell", "-Command", "Stop-Process -Name ollama -Force"], 
                                     capture_output=True, timeout=10)
                    except:
                        pass
            
            # Step 3: Wait for processes to fully terminate
            print(f"{Colors.YELLOW}Step 3: Waiting for processes to terminate...{Colors.RESET}")
            time.sleep(3)
            
            # Step 4: Restart Ollama
            print(f"{Colors.YELLOW}Step 4: Restarting Ollama service...{Colors.RESET}")
            try:
                if platform.system() == "Windows":
                    ollama_path = r"C:\Users\backup\AppData\Local\Programs\Ollama\ollama.exe"
                    subprocess.Popen([ollama_path, "serve"], 
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    subprocess.Popen(["ollama", "serve"])
            except Exception as e:
                print(f"{Colors.RED}Failed to restart Ollama: {e}{Colors.RESET}")
                return False
            
            # Step 5: Wait for Ollama to be ready
            print(f"{Colors.YELLOW}Step 5: Waiting for Ollama to be ready...{Colors.RESET}")
            max_retries = 15
            for i in range(max_retries):
                try:
                    response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
                    if response.status_code == 200:
                        print(f"{Colors.GREEN}Ollama is ready after {i+1} seconds{Colors.RESET}")
                        break
                except:
                    if i < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        print(f"{Colors.RED}Ollama failed to start properly{Colors.RESET}")
                        return False
            
            # Step 6: Verify no models are loaded
            print(f"{Colors.YELLOW}Step 6: Verifying clean memory state...{Colors.RESET}")
            try:
                response = requests.get(f"{self.ollama_url}/api/ps", timeout=10)
                if response.status_code == 200:
                    running_models = response.json().get('models', [])
                    if not running_models:
                        print(f"{Colors.GREEN}Memory successfully cleared - no models loaded{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}Warning: {len(running_models)} models still loaded{Colors.RESET}")
            except:
                pass
            
            # Step 7: Force Python garbage collection
            gc.collect()
            
            print(f"{Colors.GREEN}🧹 Aggressive memory clearing completed successfully{Colors.RESET}")
            return True
                
        except Exception as e:
            print(f"{Colors.RED}Error during aggressive memory clearing: {e}{Colors.RESET}")
            print(f"{Colors.YELLOW}Proceeding anyway, but results may be inconsistent{Colors.RESET}")
            return True
    
    def split_transcript_into_chunks(self, transcript, chunk_size=15000):
        """Split transcript into manageable chunks for LLM processing"""
        words = transcript.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def clean_ai_thinking(self, text):
        """Remove AI thinking tags and content from the output"""
        import re
        
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove any remaining thinking patterns
        text = re.sub(r'<\/think>', '', text)
        text = re.sub(r'</think>', '', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def analyze_transcript_chunks(self, transcript_text, prompt_text, prompt_name):
        """Analyze transcript in chunks and merge results"""
        chunks = self.split_transcript_into_chunks(transcript_text)
        chunk_results = []
        
        print(f"{Colors.CYAN}Processing {len(chunks)} chunks for {prompt_name} analysis...{Colors.RESET}")
        print(f"{Colors.YELLOW}Each chunk will be processed with fresh memory for optimal results.{Colors.RESET}\n")
        
        # Process each chunk
        for i, chunk in enumerate(chunks, 1):
            print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.BLUE}PROCESSING CHUNK {i}/{len(chunks)}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
            
            # Clear memory before each chunk for fresh analysis
            if i > 1:  # Don't clear on first chunk since we already cleared at start
                print(f"{Colors.CYAN}🧹 Clearing memory before chunk {i}...{Colors.RESET}")
                self.clear_model_memory()
            
            chunk_prompt = f"""You are analyzing part {i} of {len(chunks)} of a business meeting transcript. 

{prompt_text}

Focus on this chunk and provide key insights. Be concise and factual.

TRANSCRIPT CHUNK {i}:
{chunk}

Provide your analysis without any <think> tags or reasoning - just the key insights:"""

            print(f"{Colors.CYAN}📝 Analyzing chunk {i}...{Colors.RESET}")
            chunk_result = self.query_model(chunk_prompt, show_stream=True)
            
            if chunk_result:
                # Clean the result
                cleaned_result = self.clean_ai_thinking(chunk_result)
                chunk_results.append(f"**Chunk {i} Analysis:**\n{cleaned_result}")
                print(f"\n{Colors.GREEN}Chunk {i} completed successfully{Colors.RESET}\n")
            else:
                print(f"\n{Colors.RED}Chunk {i} failed{Colors.RESET}\n")
        
        # Clear memory before final merge
        print(f"{Colors.CYAN}🧹 Clearing memory before final merge...{Colors.RESET}")
        self.clear_model_memory()
        
        # Merge all chunk results
        if chunk_results:
            print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*60}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.MAGENTA}MERGING RESULTS INTO FINAL ANALYSIS{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.MAGENTA}{'='*60}{Colors.RESET}")
            return self.merge_chunk_results(chunk_results, prompt_name)
        else:
            return "Analysis failed - no chunks processed successfully."
    
    def merge_chunk_results(self, chunk_results, analysis_type):
        """Merge chunk results into a coherent final analysis"""
        combined_chunks = "\n\n".join(chunk_results)
        
        merge_prompt = f"""You are a professional business analyst. You have received analysis results from multiple chunks of a business meeting transcript about {analysis_type}.

Your task is to merge these chunk analyses into a single, coherent, professional report. 

Requirements:
- Create a well-structured, detailed human-readable analysis
- Remove any redundancy between chunks
- Organize information logically
- Use clear headings and bullet points
- Focus on actionable insights
- Write in a professional business tone
- Do NOT include any <think> tags or reasoning processes
- Do NOT mention chunks or the analysis process

CHUNK ANALYSES TO MERGE:
{combined_chunks}

Provide a final, polished business analysis report:"""

        print(f"{Colors.CYAN}Creating final merged analysis...{Colors.RESET}")
        final_result = self.query_model(merge_prompt, show_stream=True)
        
        if final_result:
            cleaned_result = self.clean_ai_thinking(final_result)
            print(f"\n{Colors.GREEN}Final analysis completed successfully{Colors.RESET}")
            return cleaned_result
        else:
            return "Failed to merge chunk results."
    
    def load_prompts(self):
        """Load all available prompts from the prompts folder"""
        prompts = {}
        prompts_dir = Path(self.prompts_folder)
        
        if not prompts_dir.exists():
            print(f"{Colors.YELLOW}Prompts folder not found. Using default prompts.{Colors.RESET}")
            return self.get_default_prompts()
        
        try:
            for prompt_file in prompts_dir.glob("*.txt"):
                prompt_name = prompt_file.stem
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompts[prompt_name] = f.read().strip()
            
            if not prompts:
                print(f"{Colors.YELLOW}No prompt files found. Using default prompts.{Colors.RESET}")
                return self.get_default_prompts()
                
            return prompts
            
        except Exception as e:
            print(f"{Colors.RED}Error loading prompts: {e}{Colors.RESET}")
            return self.get_default_prompts()
    
    def get_default_prompts(self):
        """Fallback default prompts if files aren't available"""
        return {
            "comprehensive": """Please analyze this meeting transcript comprehensively. Provide a summary, key topics, participants, action items, and recommendations.

Transcript:""",
            "summary": """Please provide a concise summary of this meeting transcript.

Transcript:"""
        }
    
    def analyze_transcript(self, transcript_text, prompt_content):
        """Analyze transcript using DeepSeek R1"""
        
        prompt = prompt_content + "\n\n" + transcript_text
        
        # Get generation parameters from config
        gen_config = self.config['deepseek']['generation']
        
        try:
            print(f"{Colors.YELLOW}Analyzing transcript with DeepSeek R1...{Colors.RESET}")
            print(f"{Colors.YELLOW}This will likely take a very long time{Colors.RESET}")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": gen_config['stream'],
                    "options": {
                        "temperature": gen_config['temperature'],
                        "top_p": gen_config['top_p'],
                        "max_tokens": gen_config['max_tokens']
                    }
                },
                timeout=gen_config['request_timeout'],
                stream=gen_config['stream']  # Set stream parameter for requests
            )
            
            if response.status_code == 200:
                if gen_config['stream']:
                    # Handle streaming response
                    print(f"{Colors.CYAN}📝 Streaming response:{Colors.RESET}\n")
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    chunk = data['response']
                                    full_response += chunk
                                    # Print chunks in real-time with color
                                    print(f"{Colors.WHITE}{chunk}{Colors.RESET}", end='', flush=True)
                                if data.get('done', False):
                                    print(f"\n\n{Colors.GREEN}Analysis complete!{Colors.RESET}")
                                    break
                            except json.JSONDecodeError as e:
                                # Skip malformed JSON lines
                                continue
                    return full_response if full_response else 'No response generated'
                else:
                    # Handle non-streaming response
                    result = response.json()
                    return result.get('response', 'No response generated')
            else:
                return f"Error: HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return f"Error communicating with DeepSeek R1: {e}"
        except json.JSONDecodeError as e:
            return f"Error parsing response from DeepSeek R1: {e}"
    
    def query_model(self, prompt, show_stream=True):
        """Simple query method for chunk processing with optional streaming"""
        try:
            gen_config = self.config['deepseek']['generation']
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": show_stream,  # Enable streaming for visual feedback
                    "options": {
                        "temperature": gen_config['temperature'],
                        "top_p": gen_config['top_p'],
                        "max_tokens": gen_config['max_tokens']
                    }
                },
                timeout=gen_config['request_timeout'],
                stream=show_stream
            )
            
            if response.status_code == 200:
                if show_stream:
                    # Handle streaming response with visual feedback
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    chunk = data['response']
                                    full_response += chunk
                                    # Print chunks in real-time
                                    print(f"{Colors.WHITE}{chunk}{Colors.RESET}", end='', flush=True)
                                if data.get('done', False):
                                    print()  # New line after completion
                                    break
                            except json.JSONDecodeError:
                                continue
                    return full_response if full_response else 'No response generated'
                else:
                    # Non-streaming response
                    result = response.json()
                    return result.get('response', 'No response generated')
            else:
                return None
                
        except Exception as e:
            print(f"{Colors.RED}Error in chunk query: {e}{Colors.RESET}")
            return None
    
    def save_analysis(self, analysis, output_file, prompt_name):
        """Save analysis to file"""
        timestamp = datetime.now().strftime(self.config['output']['timestamp_format'].replace('%Y%m%d_%H%M%S', '%Y-%m-%d %H:%M:%S'))
        
        with open(output_file, 'w', encoding=self.config['output']['encoding']) as f:
            f.write(f"# DeepSeek R1 Transcript Analysis\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Model:** {self.model_name}\n")
            f.write(f"**Prompt:** {prompt_name}\n\n")
            f.write("---\n\n")
            f.write(analysis)
        
        print(f"{Colors.GREEN}Analysis saved to: {output_file}{Colors.RESET}")

def print_banner():
    """Print application banner"""
    banner = f"""
{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════════╗
║                    DeepSeek R1 Local Analyzer                ║
║                      Transcript Analysis Tool                ║
╚══════════════════════════════════════════════════════════════╝
{Colors.RESET}
{Colors.YELLOW}Powered by local DeepSeek R1 via Ollama{Colors.RESET}
"""
    print(banner)

def main():
    print_banner()
    
    # Initialize analyzer
    analyzer = DeepSeekAnalyzer()
    
    # Check if Ollama is running
    print(f"{Colors.BLUE}Checking Ollama status...{Colors.RESET}")
    if not analyzer.check_ollama_status():
        print(f"{Colors.RED}Ollama is not running or not accessible at {analyzer.ollama_url}{Colors.RESET}")
        print(f"{Colors.YELLOW}Please ensure Ollama is installed and running.{Colors.RESET}")
        print(f"{Colors.CYAN}Visit https://ollama.ai to download and install Ollama.{Colors.RESET}")
        return
    
    print(f"{Colors.GREEN}Ollama is running{Colors.RESET}")
    
    # Check if DeepSeek R1 model is available
    print(f"{Colors.BLUE}Checking for DeepSeek R1 model...{Colors.RESET}")
    if not analyzer.check_model_availability():
        print(f"{Colors.YELLOW}DeepSeek R1 model not found locally{Colors.RESET}")
        install_choice = input(f"{Colors.CYAN}Would you like to install it now? (y/n): {Colors.RESET}").lower()
        
        if install_choice == 'y':
            if not analyzer.install_model():
                print(f"{Colors.RED}Failed to install model. Exiting.{Colors.RESET}")
                return
        else:
            print(f"{Colors.YELLOW}Cannot proceed without DeepSeek R1 model. Exiting.{Colors.RESET}")
            return
    
    print(f"{Colors.GREEN}DeepSeek R1 model available: {analyzer.model_name}{Colors.RESET}")
    
    # Clear model memory for fresh analysis
    analyzer.clear_model_memory()
    
    # Load transcript
    transcript_file = "final_transcription.txt"
    if not os.path.exists(transcript_file):
        print(f"{Colors.RED}Transcript file not found: {transcript_file}{Colors.RESET}")
        return
    
    try:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        print(f"{Colors.GREEN}Loaded transcript: {len(transcript_text)} characters{Colors.RESET}")
        
    except Exception as e:
        print(f"{Colors.RED}Error reading transcript file: {e}{Colors.RESET}")
        return
    
    # Load available prompts
    prompts = analyzer.load_prompts()
    
    # Display available prompts
    print(f"\n{Colors.BOLD}Available Analysis Prompts:{Colors.RESET}")
    prompt_names = list(prompts.keys())
    
    for i, prompt_name in enumerate(prompt_names, 1):
        # Create a friendly display name
        display_name = prompt_name.replace('_', ' ').title()
        print(f"{Colors.CYAN}{i}. {display_name}{Colors.RESET}")
    
    # Get user selection
    while True:
        try:
            choice = input(f"\n{Colors.YELLOW}Enter choice (1-{len(prompt_names)}, default: 1): {Colors.RESET}").strip()
            
            if choice == "":
                selected_index = 0  # default to first prompt
                break
            
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(prompt_names):
                break
            else:
                print(f"{Colors.RED}Please enter a number between 1 and {len(prompt_names)}{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Please enter a valid number{Colors.RESET}")
    
    selected_prompt_name = prompt_names[selected_index]
    selected_prompt_content = prompts[selected_prompt_name]
    
    # Perform analysis
    print(f"\n{Colors.BLUE}Starting {selected_prompt_name.replace('_', ' ')} analysis...{Colors.RESET}")
    start_time = time.time()
    
    # Use chunked analysis for better results
    analysis = analyzer.analyze_transcript_chunks(transcript_text, selected_prompt_content, selected_prompt_name)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"{Colors.GREEN}Analysis completed in {duration:.2f} seconds{Colors.RESET}")
    
    # Display results
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}═══ DEEPSEEK R1 ANALYSIS RESULTS ═══{Colors.RESET}\n")
    print(analysis)
    print(f"\n{Colors.BOLD}{Colors.MAGENTA}═══ END OF ANALYSIS ═══{Colors.RESET}\n")
    
    # Save to file
    timestamp = datetime.now().strftime(analyzer.config['output']['timestamp_format'])
    output_file = f"deepseek_analysis_{selected_prompt_name}_{timestamp}.md"
    
    save_choice = input(f"{Colors.CYAN}Save analysis to file? (Y/n): {Colors.RESET}").lower()
    if save_choice != 'n':
        analyzer.save_analysis(analysis, output_file, selected_prompt_name)
    
    print(f"\n{Colors.GREEN}Analysis complete!{Colors.RESET}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Analysis interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}An error occurred: {e}{Colors.RESET}")
