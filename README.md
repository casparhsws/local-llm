# Local LLM

A simple Python application that uses Ollama to run local language models with streaming output and performance metrics.

## Features

- 🚀 **Streaming responses** - See the model generate text in real-time
- ⏱️ **Performance metrics** - Time to first token and tokens per second
- 🎨 **Rich formatting** - Beautiful console output with colors and panels

## Setup

1. Install Ollama:
   ```bash
   brew install ollama
   ```

2. Start the Ollama server:
   ```bash
   ollama serve
   ```

3. Download the model:
   ```bash
   ollama pull gpt-oss:20b
   ```

4. Run the application:
   ```bash
   uv run main
   ```

## Requirements

- Python 3.x
- Ollama
- The `gpt-oss:20b` model
